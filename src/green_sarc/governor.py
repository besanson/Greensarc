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
from green_sarc.escalation import (
    EscalationEvent,
    EscalationReason,
    EscalationRouter,
    RouteOutcome,
)
from green_sarc.forecast import GateDecision, Verdict
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

    @classmethod
    def with_defaults(
        cls,
        *,
        token_budget: float,
        carbon_ceiling: float = 1.0e9,
        usd_budget: Optional[float] = None,
        delta: float = 0.05,
        cost_model: Optional[CostModel] = None,
        carbon_model: Optional[CarbonModel] = None,
        estimator: Optional[Estimator] = None,
        bootstrap_jsonl: Optional[str] = None,
        store: Optional[AuditStore] = None,
        max_loops: int = 50,
        max_total_cost: Optional[float] = None,
    ) -> "GreenGovernor":
        """Build a ready-to-use governor with sensible defaults.

        Wires a learning estimator plus the reference pricing/carbon tables
        (:mod:`green_sarc.data`) so you only specify your budgets::

            gov = GreenGovernor.with_defaults(token_budget=200_000, usd_budget=5.0)

        Pass your own ``cost_model`` / ``carbon_model`` for real provider prices
        and grid data, a pre-built ``estimator``, or a ``bootstrap_jsonl`` path to
        rehydrate the forecaster from a prior audit log.
        """
        from green_sarc.data import default_carbon, default_pricing
        from green_sarc.estimator import LearnedEstimator

        costs = cost_model if cost_model is not None else default_pricing()
        carbon = carbon_model if carbon_model is not None else default_carbon()
        est = estimator if estimator is not None else LearnedEstimator(costs, carbon)
        if bootstrap_jsonl is not None and hasattr(est, "bootstrap_from_jsonl"):
            est.bootstrap_from_jsonl(bootstrap_jsonl)
        kwargs: dict[str, Any] = {
            "budget": Budget(
                token_budget=token_budget,
                carbon_ceiling=carbon_ceiling,
                usd_budget=usd_budget,
                delta=delta,
            ),
            "estimator": est,
            "cost_model": costs,
            "carbon_model": carbon,
            "monitor": ActionTimeMonitor(max_loops=max_loops, max_total_cost=max_total_cost),
        }
        if store is not None:
            kwargs["store"] = store
        return cls(**kwargs)

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
            extra: dict[str, Any] = {}
            escalated = decision.verdict is Verdict.ESCALATE
            if escalated:
                outcome_er = await self._escalate(
                    self._exhaustion_reason(), action_id, action, decision.reason
                )
                extra["escalation_handled"] = outcome_er.handled
            self._record_blocked(action_id, action, decision, intensity, escalated, extra)
            raise GateRejected(decision)

        # ---- reserve the forecast upper bound atomically (audit P0-1) ----
        reserve_tokens = self.gate.cost_upper_bound(decision.forecast, self.budget.delta)
        reserve_carbon = decision.forecast.carbon_hat
        if not self.budget.reserve(reserve_tokens, reserve_carbon):
            # Raced out by a concurrent action between evaluate() and reserve().
            raced = GateDecision(
                verdict=Verdict.REJECT,
                forecast=decision.forecast,
                reason="insufficient budget after concurrent reservations",
            )
            self._record_blocked(action_id, action, raced, intensity, False, {})
            raise GateRejected(raced)

        # ---- SITE 2: Action-Time Monitor (pre-execution loop guard) ------
        try:
            self.monitor.before()
        except CircuitTripped as exc:
            self.budget.release(reserve_tokens, reserve_carbon)
            outcome_er = await self._escalate(
                EscalationReason.CIRCUIT_TRIPPED, action_id, action, exc.reason
            )
            self._record_blocked(
                action_id,
                action,
                decision,
                intensity,
                True,
                {"escalation_handled": outcome_er.handled},
                circuit_tripped=True,
            )
            raise

        # ---- execute the admitted action ---------------------------------
        try:
            outcome = await execute(action)
        except BaseException:
            self.budget.release(reserve_tokens, reserve_carbon)
            raise

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
        prompt = float(action.prompt_tokens or 0)
        actual_usd = self.cost_model.usd(action.model, prompt, max(0.0, actual_cost - prompt))
        # Release the reservation and spend the actuals in one atomic step.
        self.budget.commit(reserve_tokens, reserve_carbon, actual_cost, actual_carbon, actual_usd)

        # ---- SITE 2: Action-Time Monitor (post-execution cost guard) -----
        circuit_exc: Optional[CircuitTripped] = None
        try:
            self.monitor.after(actual_cost)
        except CircuitTripped as exc:
            circuit_exc = exc

        # ---- SITE 4: Escalation Router (route before recording so the audit
        #      can carry whether the escalation was handled) ---------------
        extra = {}
        exhausted = self._is_exhausted()
        escalated = circuit_exc is not None or exhausted
        if circuit_exc is not None:
            outcome_er = await self._escalate(
                EscalationReason.CIRCUIT_TRIPPED, action_id, action, circuit_exc.reason
            )
            extra["escalation_handled"] = outcome_er.handled
        elif exhausted:
            outcome_er = await self._escalate(
                self._exhaustion_reason(),
                action_id,
                action,
                "budget or carbon ceiling reached after action",
            )
            extra["escalation_handled"] = outcome_er.handled

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
            prompt_tokens=action.prompt_tokens or 0,
            actual_usd=actual_usd,
            circuit_tripped=circuit_exc is not None,
            escalated=escalated,
            session_id=session_id or None,
            extra=extra,
        )

        if circuit_exc is not None:
            raise circuit_exc

        return GovernedResult(
            result=outcome.result,
            decision=decision,
            audit=record,
            actual_cost=actual_cost,
            actual_carbon=actual_carbon,
            circuit_tripped=False,
        )

    # -- helpers ----------------------------------------------------------

    def _record_blocked(
        self,
        action_id: str,
        action: Action,
        decision: GateDecision,
        intensity: float,
        escalated: bool,
        extra: dict[str, Any],
        *,
        circuit_tripped: bool = False,
    ) -> None:
        """Write an audit record for an action that did not run (cost = 0)."""
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
            prompt_tokens=action.prompt_tokens or 0,
            circuit_tripped=circuit_tripped,
            escalated=escalated,
            extra=extra,
        )

    def _is_exhausted(self) -> bool:
        return self.budget.is_token_exhausted() or self.budget.is_carbon_exhausted()

    def _exhaustion_reason(self) -> EscalationReason:
        if self.budget.is_carbon_exhausted():
            return EscalationReason.CARBON_EXHAUSTED
        if self.budget.is_usd_exhausted():
            return EscalationReason.USD_EXHAUSTED
        return EscalationReason.TOKEN_EXHAUSTED

    async def _escalate(
        self,
        reason: EscalationReason,
        action_id: str,
        action: Action,
        detail: str,
    ) -> RouteOutcome:
        return await self.router.route(
            EscalationEvent(
                reason=reason,
                action_id=action_id,
                action_kind=action.kind,
                detail=detail,
                budget_remaining_tokens=self.budget.remaining_tokens(),
                carbon_remaining=self.budget.remaining_carbon(),
            )
        )
