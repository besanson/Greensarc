"""Green SARC — predictive cost + carbon governance for agentic AI systems.

Green SARC wraps an agent's execution loop and decides, in real time, whether a
proposed action fits the remaining token budget and carbon ceiling — forecasting
that cost *before* the action fires rather than reconciling it after.

It applies the SARC governance architecture's four enforcement sites to the
FinOps / GreenOps domain (cost and carbon only):

- **Pre-Action Gate** (:class:`PreActionGate`)         — admit on a learned forecast.
- **Action-Time Monitor** (:class:`ActionTimeMonitor`) — circuit-break runaway loops.
- **Post-Action Auditor** (:class:`PostActionAuditor`) — log predicted-vs-actual.
- **Escalation Router** (:class:`EscalationRouter`)    — route on exhaustion.

The core is framework-agnostic and runs standalone.  Orchestrators such as KAOS
integrate through adapters in :mod:`green_sarc.adapters`; the dependency runs one
way (KAOS → Green SARC) and the core never imports an orchestrator.

See the working paper: Besanson, G. (2026), "Green SARC: Predictive FinOps as
Governance-by-Architecture for Agentic AI Systems."  The four-enforcement-site
architecture is borrowed from the SARC framework (arXiv:2605.07728,
github.com/besanson/sarc-governance).
"""

from __future__ import annotations

from green_sarc.auditor import AuditRecord, PostActionAuditor
from green_sarc.escalation import (
    DeterministicFallbackHandler,
    EscalationEvent,
    EscalationReason,
    EscalationRouter,
    log_only_handler,
)
from green_sarc.estimator import ColdStartEstimator, Estimator, LearnedEstimator
from green_sarc.forecast import Forecast, GateDecision, Verdict
from green_sarc.gate import PreActionGate
from green_sarc.governor import (
    ActionOutcome,
    GateRejected,
    GovernedResult,
    GreenGovernor,
)
from green_sarc.monitor import ActionTimeMonitor, CircuitTripped
from green_sarc.pricing import (
    CarbonModel,
    CostModel,
    IntensityProvider,
    ModelProfile,
    TableCarbonModel,
    TableCostModel,
    carbon_for_tokens,
    default_carbon_model,
    default_cost_model,
)
from green_sarc.state import Action, Budget, GovernanceContext
from green_sarc.stores import AuditStore, JSONLAuditStore, MemoryAuditStore
from green_sarc.trajectory import (
    NotImplementedTrajectoryEstimator,
    Plan,
    TrajectoryEstimator,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # state
    "Action",
    "Budget",
    "GovernanceContext",
    # forecast
    "Forecast",
    "GateDecision",
    "Verdict",
    # pricing / carbon (model-agnostic table)
    "ModelProfile",
    "CostModel",
    "CarbonModel",
    "IntensityProvider",
    "TableCostModel",
    "TableCarbonModel",
    "carbon_for_tokens",
    "default_cost_model",
    "default_carbon_model",
    # estimator (Phase 1)
    "Estimator",
    "ColdStartEstimator",
    "LearnedEstimator",
    # Phase 2 stub
    "Plan",
    "TrajectoryEstimator",
    "NotImplementedTrajectoryEstimator",
    # four enforcement sites
    "PreActionGate",
    "ActionTimeMonitor",
    "CircuitTripped",
    "PostActionAuditor",
    "AuditRecord",
    "EscalationRouter",
    "EscalationEvent",
    "EscalationReason",
    "DeterministicFallbackHandler",
    "log_only_handler",
    # governor
    "GreenGovernor",
    "ActionOutcome",
    "GovernedResult",
    "GateRejected",
    # stores
    "AuditStore",
    "MemoryAuditStore",
    "JSONLAuditStore",
]
