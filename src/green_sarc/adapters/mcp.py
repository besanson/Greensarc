"""KAOS adapter — Green SARC as an MCP server.

KAOS is MCP-native: an agent references tools by listing an ``MCPServer`` in its
``spec.mcpServers``; the KAOS operator injects the server's endpoint into the
agent pod and PAIS discovers its tools over MCP Streamable HTTP.  Green SARC
therefore exposes its two agent-facing sites as MCP tools:

- ``pre_action_gate`` — forecast a proposed action and return an admit/reject
  verdict plus the predicted cost and carbon.
- ``post_action_auditor`` — report the action's *actual* token usage so the
  predicted-vs-actual log is written and the estimator retrains.

This honours the one-way dependency (KAOS → Green SARC): deploying it requires
**no** change to KAOS or PAIS — just an ``MCPServer`` custom resource pointing at
this server.

:class:`GreenSarcMCPService` holds the pure, framework-free logic and is fully
testable without the ``mcp`` package installed.  :func:`build_mcp_server` wires
it into a FastMCP server and imports ``mcp`` lazily.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

from green_sarc.auditor import PostActionAuditor
from green_sarc.estimator import Estimator
from green_sarc.forecast import GateDecision
from green_sarc.gate import PreActionGate
from green_sarc.pricing import CarbonModel, CostModel, carbon_for_tokens
from green_sarc.state import Action, Budget, GovernanceContext
from green_sarc.stores.base import AuditStore
from green_sarc.stores.memory import MemoryAuditStore

__all__ = [
    "GreenSarcMCPService",
    "build_mcp_server",
]


class GreenSarcMCPService:
    """Stateful service backing the two Green SARC MCP tools.

    Because MCP tool calls are independent, the gate stashes each forecast under
    a generated ``action_id`` that the auditor call passes back, correlating the
    prediction with its actual.
    """

    def __init__(
        self,
        budget: Budget,
        estimator: Estimator,
        cost_model: CostModel,
        carbon_model: CarbonModel,
        store: Optional[AuditStore] = None,
    ) -> None:
        self.budget = budget
        self.cost_model = cost_model
        self.carbon_model = carbon_model
        self.gate = PreActionGate(estimator)
        self.auditor = PostActionAuditor(store or MemoryAuditStore(), estimator)
        self._pending: Dict[str, Tuple[Action, GateDecision]] = {}

    def pre_action_gate(
        self,
        *,
        kind: str,
        model: str = "",
        region: str = "",
        prompt_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """MCP tool: forecast and gate a proposed action."""
        action = Action(
            kind=kind,
            model=model,
            region=region,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )
        ctx = GovernanceContext(budget=self.budget)
        decision = self.gate.evaluate(action, ctx)
        action_id = uuid.uuid4().hex
        self._pending[action_id] = (action, decision)
        out = decision.to_dict()
        out["action_id"] = action_id
        out["budget_remaining_tokens"] = self.budget.remaining_tokens()
        out["carbon_remaining"] = self.budget.remaining_carbon()
        return out

    def post_action_auditor(
        self,
        *,
        action_id: str,
        actual_tokens: float,
        actual_carbon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """MCP tool: report actuals, write the audit record, retrain."""
        pending = self._pending.pop(action_id, None)
        if pending is None:
            return {"ok": False, "error": f"unknown action_id {action_id!r}"}
        action, decision = pending
        intensity = self.carbon_model.carbon_intensity(action.region)
        carbon = (
            float(actual_carbon)
            if actual_carbon is not None
            else carbon_for_tokens(
                self.cost_model,
                self.carbon_model,
                action.model,
                float(actual_tokens),
                action.region,
            )
        )
        self.budget.spend(float(actual_tokens), carbon)
        rec = self.auditor.record(
            action_id=action_id,
            action_kind=action.kind,
            model=action.model,
            region=action.region,
            forecast=decision.forecast,
            decision=decision,
            actual_cost=float(actual_tokens),
            actual_carbon=carbon,
            budget_remaining_tokens=self.budget.remaining_tokens(),
            carbon_remaining=self.budget.remaining_carbon(),
            carbon_intensity=intensity,
        )
        return {
            "ok": True,
            "cost_error": rec.cost_error,
            "carbon_error": rec.carbon_error,
            "budget_remaining_tokens": self.budget.remaining_tokens(),
            "carbon_remaining": self.budget.remaining_carbon(),
        }


def build_mcp_server(service: GreenSarcMCPService, name: str = "green-sarc") -> Any:
    """Wire a :class:`GreenSarcMCPService` into a FastMCP server.

    Imports ``mcp`` lazily; install the optional extra with
    ``pip install 'green-sarc[mcp]'``.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the MCP adapter requires the 'mcp' package; install green-sarc[mcp]"
        ) from exc

    server = FastMCP(name)

    @server.tool()
    def pre_action_gate(
        kind: str,
        model: str = "",
        region: str = "",
        prompt_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Forecast a proposed action's cost/carbon and return an admit verdict."""
        return service.pre_action_gate(
            kind=kind,
            model=model,
            region=region,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
        )

    @server.tool()
    def post_action_auditor(
        action_id: str,
        actual_tokens: float,
        actual_carbon: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Report an action's actual token usage to close the audit loop."""
        return service.post_action_auditor(
            action_id=action_id,
            actual_tokens=actual_tokens,
            actual_carbon=actual_carbon,
        )

    return server
