"""KAOS adapter demo: Green SARC as an MCP server.

Run it::

    python examples/kaos_mcp_adapter/run_demo.py

KAOS is the caller and it is MCP-native, so Green SARC exposes its Pre-Action
Gate and Post-Action Auditor as MCP tools.  This demo drives the *same* service
object that the MCP server wraps, simulating what a KAOS agent does inside its
reasoning loop:

    1. before an expensive step, call ``pre_action_gate`` and check the verdict;
    2. if admitted, run the step;
    3. afterwards, call ``post_action_auditor`` with the real token usage.

The dependency runs one way (KAOS -> Green SARC): nothing here imports KAOS.  To
serve this for real over MCP, see ``kaos_manifests.yaml`` in this directory and
the ``build_mcp_server`` call at the bottom (requires ``green-sarc[mcp]``).
"""

from __future__ import annotations

from green_sarc import Budget, LearnedEstimator, TableCarbonModel, TableCostModel
from green_sarc.adapters.mcp import GreenSarcMCPService

# Mock token usage a KAOS agent would observe per step (from the model response).
AGENT_STEPS = [
    ("chat.completion", 240.0),
    ("tool.search", 90.0),
    ("chat.completion", 5_000.0),  # this one will be rejected by the gate
    ("chat.completion", 260.0),
]


def main() -> None:
    cost_model = TableCostModel()
    carbon_model = TableCarbonModel(intensities={"eu-west": 230.0})
    budget = Budget(token_budget=2_000.0, carbon_ceiling=5.0, delta=0.05)

    service = GreenSarcMCPService(
        budget=budget,
        estimator=LearnedEstimator(cost_model, carbon_model, min_samples=2),
        cost_model=cost_model,
        carbon_model=carbon_model,
    )

    print("Simulating a KAOS agent invoking the Green SARC MCP tools:\n")
    for kind, actual_tokens in AGENT_STEPS:
        # --- MCP tool call #1: pre_action_gate -------------------------------
        gate = service.pre_action_gate(
            kind=kind,
            model="gpt-x",
            region="eu-west",
            prompt_tokens=120,
            max_tokens=180,
        )
        print(
            f"-> pre_action_gate({kind}): verdict={gate['verdict']} "
            f"predicted_cost={gate['cost_hat']:.0f} "
            f"budget_left={gate['budget_remaining_tokens']:.0f}"
        )
        if not gate["admitted"]:
            print(f"   agent backs off: {gate['reason']}\n")
            continue

        # --- the agent actually runs the step here ---------------------------

        # --- MCP tool call #2: post_action_auditor ---------------------------
        audit = service.post_action_auditor(
            action_id=gate["action_id"], actual_tokens=actual_tokens
        )
        print(
            f"<- post_action_auditor: actual={actual_tokens:.0f} "
            f"cost_error={audit['cost_error']:+.0f} "
            f"budget_left={audit['budget_remaining_tokens']:.0f}\n"
        )

    print("Audit log (predicted vs actual):")
    for rec in service.auditor.store.list():
        print(
            f"  {rec.action_kind:<16} pred={rec.predicted_cost:7.1f} "
            f"actual={rec.actual_cost:7.1f} err={rec.cost_error:+7.1f}"
        )

    # To serve this over MCP for a real KAOS agent (requires green-sarc[mcp]):
    #
    #     from green_sarc.adapters.mcp import build_mcp_server
    #     server = build_mcp_server(service, name="green-sarc")
    #     server.run(transport="streamable-http")  # then register it as an MCPServer CR
    print("\n(See kaos_manifests.yaml for the MCPServer + Agent custom resources.)")


if __name__ == "__main__":
    main()
