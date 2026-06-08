"""SARC adapter — connect Green SARC into the SARC governance framework.

Green SARC borrows the four-enforcement-site architecture from SARC
(github.com/besanson/sarc-governance) and runs standalone.  This adapter makes
the relationship *concrete*: it expresses Green SARC's predictive cost/carbon
control as SARC :class:`~sarc_governance.Constraint` objects plugged into a
SARC :class:`~sarc_governance.GovernanceToolset`, so a single governed toolset
enforces **both** safety (SARC's own constraints) and **cost/carbon** (Green
SARC) at the *same* four sites.

Mapping onto SARC's sites:

- **Pre-Action Gate (PAG)** — a HARD constraint whose predicate runs Green
  SARC's :class:`~green_sarc.gate.PreActionGate`.  It *fires* (and SARC blocks
  the action with ``ConstraintViolation``) exactly when the forecast does not
  fit the remaining budget / carbon ceiling.
- **Post-Action Auditor (PAA)** — a SOFT constraint whose predicate reads the
  action's actual token usage from the tool result, writes the Green SARC
  predicted-vs-actual :class:`~green_sarc.auditor.AuditRecord`, spends the
  budget, and retrains the estimator.

This is the *composition* described in ``docs/relationship-to-sarc.md`` realised
in code.  The Green SARC core never imports SARC; only this adapter does, and it
imports it lazily (``pip install 'green-sarc[sarc]'``) so the rest of the
package stays dependency-free and usable standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from green_sarc._ttl_map import TTLMap
from green_sarc.auditor import PostActionAuditor
from green_sarc.estimator import Estimator
from green_sarc.forecast import GateDecision, Verdict
from green_sarc.gate import PreActionGate
from green_sarc.pricing import CarbonModel, CostModel, carbon_for_tokens
from green_sarc.state import Action, Budget, GovernanceContext
from green_sarc.stores.base import AuditStore
from green_sarc.stores.memory import MemoryAuditStore

__all__ = [
    "default_usage_extractor",
    "SarcCostCarbonGovernance",
    "wrap_toolset",
]


def _require_sarc() -> Any:
    try:
        import sarc_governance as sg
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the SARC adapter requires the 'sarc-governance' package; install green-sarc[sarc]"
        ) from exc
    return sg


def default_usage_extractor(result: Any) -> float:
    """Best-effort extraction of actual token usage from a tool result.

    Understands an OpenAI-style ``{"usage": {...}}`` dict, a plain
    ``actual_tokens``/``total_tokens`` field, or an object exposing those
    attributes.  Returns ``0.0`` when nothing usable is found (the caller should
    supply a custom extractor for its result shape).
    """
    if isinstance(result, dict):
        usage = result.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if total:
                return float(total)
            return float((usage.get("prompt_tokens") or 0) + (usage.get("completion_tokens") or 0))
        for key in ("actual_tokens", "total_tokens"):
            if key in result and result[key] is not None:
                return float(result[key])
        return 0.0
    for attr in ("actual_tokens", "total_tokens"):
        value = getattr(result, attr, None)
        if value is not None:
            return float(value)
    return 0.0


@dataclass
class SarcCostCarbonGovernance:
    """Builds the Green SARC cost/carbon constraints for a SARC toolset.

    Holds the live :class:`Budget`, the predictive gate, and the auditor, and
    produces the SARC :class:`Constraint` objects via :meth:`constraints`.  The
    gate forecasts at PAG; the auditor records actuals at PAA — mirroring the
    standalone :class:`~green_sarc.governor.GreenGovernor`, but expressed as SARC
    constraints so they compose with a caller's safety constraints.

    Override hooks (for tool shapes that differ from the defaults):

    - ``action_factory(tool, args) -> Action`` maps a SARC tool call to a Green
      SARC :class:`~green_sarc.state.Action`.  The default reads ``model`` /
      ``prompt_tokens`` / ``max_tokens`` from ``args``; supply your own if your
      tool arguments are shaped differently.
    - ``usage_extractor(result) -> float`` reads the actual token count from the
      tool result (default understands an OpenAI-style ``usage`` dict).

    Predicate context contract (what SARC must put in ``ctx``): the PAG predicate
    reads ``ctx["tool"]`` and ``ctx["args"]``; the PAA predicate additionally
    reads ``ctx["result"]``.  ``id(args)`` correlates the two within one
    ``call_tool`` invocation.
    """

    budget: Budget
    estimator: Estimator
    cost_model: CostModel
    carbon_model: CarbonModel
    region: str = ""
    action_factory: Optional[Callable[[str, Dict[str, Any]], Action]] = None
    usage_extractor: Callable[[Any], float] = default_usage_extractor
    store: AuditStore = field(default_factory=MemoryAuditStore)
    gate: PreActionGate = field(init=False)
    auditor: PostActionAuditor = field(init=False)
    _pending: TTLMap[int, Tuple[Action, GateDecision, float, float]] = field(
        default_factory=TTLMap
    )

    def __post_init__(self) -> None:
        self.gate = PreActionGate(self.estimator)
        self.auditor = PostActionAuditor(self.store, self.estimator)
        self._pending = TTLMap(on_evict=self._release_pending)
        if self.action_factory is None:
            self.action_factory = self._default_action

    def _release_pending(
        self, _key: int, value: Tuple[Action, GateDecision, float, float]
    ) -> None:
        """Return the budget reservation an orphaned (un-audited) gate held."""
        _action, _decision, reserve_tokens, reserve_carbon = value
        self.budget.release(reserve_tokens, reserve_carbon)

    def _default_action(self, tool: str, args: Dict[str, Any]) -> Action:
        return Action(
            kind=tool,
            model=str(args.get("model", "")),
            region=self.region,
            prompt_tokens=args.get("prompt_tokens"),
            max_tokens=args.get("max_tokens"),
        )

    # -- SARC predicates --------------------------------------------------

    def _pag_predicate(self, ctx: Dict[str, Any]) -> bool:
        """HARD/PAG predicate: fire (block) when the gate does not admit."""
        assert self.action_factory is not None
        args = ctx.get("args") or {}
        action = self.action_factory(str(ctx.get("tool", "")), args)
        decision = self.gate.evaluate(action, GovernanceContext(budget=self.budget))
        if decision.admitted:
            reserve_tokens = self.gate.cost_upper_bound(decision.forecast, self.budget.delta)
            reserve_carbon = decision.forecast.carbon_hat
            if self.budget.reserve(reserve_tokens, reserve_carbon):
                self._pending.put(id(args), (action, decision, reserve_tokens, reserve_carbon))
                return False  # do not fire -> SARC lets the action through
            # Raced out by a concurrent reservation: block as a reject.
            decision = GateDecision(
                verdict=Verdict.REJECT,
                forecast=decision.forecast,
                reason="insufficient budget after concurrent reservations",
            )
        # Rejected: log the blocked action, then fire so SARC raises.
        self._record(action, decision, actual_tokens=0.0)
        return True

    def _paa_predicate(self, ctx: Dict[str, Any]) -> bool:
        """SOFT/PAA predicate: record actuals; never fires (side effect only)."""
        args = ctx.get("args") or {}
        pending = self._pending.pop(id(args), None)
        if pending is None:
            return False
        action, decision, reserve_tokens, reserve_carbon = pending
        actual = float(self.usage_extractor(ctx.get("result")))
        self._record(
            action,
            decision,
            actual_tokens=actual,
            reserve_tokens=reserve_tokens,
            reserve_carbon=reserve_carbon,
        )
        return False

    def _record(
        self,
        action: Action,
        decision: GateDecision,
        *,
        actual_tokens: float,
        reserve_tokens: float = 0.0,
        reserve_carbon: float = 0.0,
    ) -> None:
        carbon = carbon_for_tokens(
            self.cost_model, self.carbon_model, action.model, actual_tokens, action.region
        )
        prompt = float(action.prompt_tokens or 0)
        actual_usd = self.cost_model.usd(action.model, prompt, max(0.0, actual_tokens - prompt))
        self.budget.commit(reserve_tokens, reserve_carbon, actual_tokens, carbon, actual_usd)
        self.auditor.record(
            action_id="",
            action_kind=action.kind,
            model=action.model,
            region=action.region,
            forecast=decision.forecast,
            decision=decision,
            actual_cost=actual_tokens,
            actual_carbon=carbon,
            budget_remaining_tokens=self.budget.remaining_tokens(),
            carbon_remaining=self.budget.remaining_carbon(),
            carbon_intensity=self.carbon_model.carbon_intensity(action.region),
            prompt_tokens=action.prompt_tokens or 0,
            actual_usd=actual_usd,
        )

    # -- SARC constraints -------------------------------------------------

    def constraints(self) -> List[Any]:
        """Return the Green SARC cost/carbon constraints as SARC ``Constraint``s."""
        sg = _require_sarc()
        return [
            sg.Constraint(
                id="green_sarc.budget_gate",
                klass=sg.ConstraintClass.HARD,
                verif=sg.EnforcementPoint.PAG,
                response=sg.Response.BLOCK,
                predicate=self._pag_predicate,
                description="Predictive cost/carbon Pre-Action Gate (Green SARC).",
            ),
            sg.Constraint(
                id="green_sarc.cost_carbon_audit",
                klass=sg.ConstraintClass.SOFT,
                verif=sg.EnforcementPoint.PAA,
                response=sg.Response.LOG,
                predicate=self._paa_predicate,
                description="Predicted-vs-actual cost/carbon auditor (Green SARC).",
            ),
        ]


def wrap_toolset(
    toolset: Any,
    governance: SarcCostCarbonGovernance,
    *,
    spec: Any = None,
    **governance_toolset_kwargs: Any,
) -> Any:
    """Wrap ``toolset`` in a SARC ``GovernanceToolset`` enforcing cost + carbon.

    If ``spec`` (an existing :class:`~sarc_governance.ConstraintSpec` of the
    caller's safety constraints) is given, Green SARC's cost/carbon constraints
    are appended to it — the two layers then share the four enforcement sites on
    one toolset.  Extra keyword arguments are forwarded to ``GovernanceToolset``.
    """
    sg = _require_sarc()
    constraints = list(spec.constraints) if spec is not None else []
    constraints += governance.constraints()
    return sg.GovernanceToolset(
        toolset, spec=sg.ConstraintSpec(constraints), **governance_toolset_kwargs
    )
