"""The Green Governor — wires the four enforcement sites around an executor.

``GreenGovernor.run_action`` is the single async path an action travels:

1. **Pre-Action Gate (PAG)** — forecast and admit/reject/escalate.
2. **Action-Time Monitor (ATM)** — circuit-breaker check around execution.
3. **Post-Action Auditor (PAA)** — log predicted-vs-actual; retrain the estimator.
4. **Escalation Router (ER)** — route on budget/carbon exhaustion or a tripped breaker.

The governor is framework-agnostic: the caller supplies an ``execute`` coroutine
that actually runs the action and reports its real token usage.  This is the
seam the KAOS adapters plug into; nothing here imports KAOS.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from green_sarc.auditor import AuditRecord, PostActionAuditor
from green_sarc.estimator import Estimator
from green_sarc.escalation import EscalationEvent, EscalationReason, EscalationRouter
from green_sarc.forecast import GateDecision
from green_sarc.gate import PreActionGate
from green_sarc.monitor import ActionTimeMonitor, CircuitTripped
from green_sarc.pricing import CarbonModel, CostModel, carbon_for_tokens
from green_sarc.state import Action, Budget, GovernanceContext
from green_sarc.stores.base import AuditStore
from green_sarc.stores.memory import MemoryAuditStore

__all__ = [
    "ActionOutcome",
    "GovernedResult",
    "GateRejected",
    "GreenGovernor",
]


@dataclass
class ActionOutcome:
    """What an ``execute`` coroutine returns: the result and its real cost.

    ``actual_carbon`` may be omitted, in which case the governor computes it from
    ``actual_tokens`` via the carbon table.
    """

    result: Any
    actual_tokens: float
    actual_carbon: Optional[float] = None


@dataclass
class GovernedResult:
    """The outcome of a governed action that ran to completion."""

    result: Any
    decision: GateDecision
    audit: AuditRecord
    actual_cost: float
    actual_carbon: float
    circuit_tripped: bool = False


class GateRejected(Exception):
    """Raised when the Pre-Action Gate did not admit an action."""

    def __init__(self, decision: GateDecision) -> None:
        self.decision = decision
        super().__init__(f"action {decision.verdict.value}: {decision.reason}")


Executor = Callable[[Action], Awaitable[ActionOutcome]]


@dataclass
class GreenGovernor:
    """Governs an agent's actions through the four enforcement sites.

    Construct it with the live :class:`Budget`, an :class:`Estimator`, and the
    pricing/carbon tables; the gate, auditor, monitor, and router are wired up
    automatically (override ``monitor``/``router``/``store`` to customise).
    """

    budget: Budget
    estimator: Estimator
    cost_model: CostModel
    carbon_model: CarbonModel
    store: AuditStore = field(default_factory=MemoryAuditStore)
    monitor: ActionTimeMonitor = field(default_factory=ActionTimeMonitor)
    router: EscalationRouter = field(default_factory=EscalationRouter)
    gate: PreActionGate = field(init=False)
    auditor: PostActionAuditor = field(init=False)

    def __post_init__(self) -> None:
        self.gate = PreActionGate(self.estimator)
        self.auditor = PostActionAuditor(self.store, self.estimator)

    def _context(self, principal_id: str, session_id: str) -> GovernanceContext:
        return GovernanceContext(
            budget=self.budget,
            principal_id=principal_id,
            session_id=session_id,
        )

    async def run_action(
        self,
        action: Action,
        execute: Executor,
        *,
        principal_id: str = "",
        session_id: str = "",
    ) -> GovernedResult:
        """Run one action through PAG → ATM → PAA → ER.

        Returns a :class:`GovernedResult` on success.  Raises :class:`GateRejected`
        if the gate blocks the action, or
        :class:`~green_sarc.monitor.CircuitTripped` if the breaker kills the loop
        (in both cases an :class:`AuditRecord` is still written).
        """
        ctx = self._context(principal_id, session_id)
        action_id = uuid.uuid4().hex
        intensity = self.carbon_model.carbon_intensity(action.region, ctx.timestamp)

        # ---- SITE 1: Pre-Action Gate -------------------------------------
        decision = self.gate.evaluate(action, ctx)
        if not decision.admitted:
            escalated = decision.verdict.value == "escalate"
            self.auditor.record(
                action_id=action_id,
                action_kind=action.kind,
                model=action.model,
                region=action.region,
                forecast=decision.forecast,
                decision=decision,
                actual_cost=0.0,
                actual_carbon=0.0,
                budget_remaining_tokens=self.budget.remaining_tokens(),
                carbon_remaining=self.budget.remaining_carbon(),
                carbon_intensity=intensity,
                escalated=escalated,
            )
            if escalated:
                await self._escalate(
                    self._exhaustion_reason(),
                    action_id,
                    action,
                    decision.reason,
                )
            raise GateRejected(decision)

        # ---- SITE 2: Action-Time Monitor (pre-execution loop guard) ------
        try:
            self.monitor.before()
        except CircuitTripped as exc:
            self.auditor.record(
                action_id=action_id,
                action_kind=action.kind,
                model=action.model,
                region=action.region,
                forecast=decision.forecast,
                decision=decision,
                actual_cost=0.0,
                actual_carbon=0.0,
                budget_remaining_tokens=self.budget.remaining_tokens(),
                carbon_remaining=self.budget.remaining_carbon(),
                carbon_intensity=intensity,
                circuit_tripped=True,
                escalated=True,
            )
            await self._escalate(EscalationReason.CIRCUIT_TRIPPED, action_id, action, exc.reason)
            raise

        # ---- execute the admitted action ---------------------------------
        outcome = await execute(action)
        actual_cost = float(outcome.actual_tokens)
        actual_carbon = (
            float(outcome.actual_carbon)
            if outcome.actual_carbon is not None
            else carbon_for_tokens(
                self.cost_model,
                self.carbon_model,
                action.model,
                actual_cost,
                action.region,
                ctx.timestamp,
            )
        )
        self.budget.spend(actual_cost, actual_carbon)

        # ---- SITE 2: Action-Time Monitor (post-execution cost guard) -----
        circuit_exc: Optional[CircuitTripped] = None
        try:
            self.monitor.after(actual_cost)
        except CircuitTripped as exc:
            circuit_exc = exc

        # ---- SITE 3: Post-Action Auditor (log + retrain) -----------------
        record = self.auditor.record(
            action_id=action_id,
            action_kind=action.kind,
            model=action.model,
            region=action.region,
            forecast=decision.forecast,
            decision=decision,
            actual_cost=actual_cost,
            actual_carbon=actual_carbon,
            budget_remaining_tokens=self.budget.remaining_tokens(),
            carbon_remaining=self.budget.remaining_carbon(),
            carbon_intensity=intensity,
            circuit_tripped=circuit_exc is not None,
            escalated=circuit_exc is not None or self._is_exhausted(),
        )

        # ---- SITE 4: Escalation Router -----------------------------------
        if circuit_exc is not None:
            await self._escalate(
                EscalationReason.CIRCUIT_TRIPPED, action_id, action, circuit_exc.reason
            )
            raise circuit_exc
        if self._is_exhausted():
            await self._escalate(
                self._exhaustion_reason(),
                action_id,
                action,
                "budget or carbon ceiling reached after action",
            )

        return GovernedResult(
            result=outcome.result,
            decision=decision,
            audit=record,
            actual_cost=actual_cost,
            actual_carbon=actual_carbon,
            circuit_tripped=False,
        )

    # -- helpers ----------------------------------------------------------

    def _is_exhausted(self) -> bool:
        return self.budget.is_token_exhausted() or self.budget.is_carbon_exhausted()

    def _exhaustion_reason(self) -> EscalationReason:
        if self.budget.is_carbon_exhausted():
            return EscalationReason.CARBON_EXHAUSTED
        return EscalationReason.TOKEN_EXHAUSTED

    async def _escalate(
        self,
        reason: EscalationReason,
        action_id: str,
        action: Action,
        detail: str,
    ) -> None:
        await self.router.route(
            EscalationEvent(
                reason=reason,
                action_id=action_id,
                action_kind=action.kind,
                detail=detail,
                budget_remaining_tokens=self.budget.remaining_tokens(),
                carbon_remaining=self.budget.remaining_carbon(),
            )
        )
