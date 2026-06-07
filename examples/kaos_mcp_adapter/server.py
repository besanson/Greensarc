"""Runnable Green SARC MCP server for deployment under KAOS.

Builds a :class:`~green_sarc.adapters.mcp.GreenSarcMCPService` from environment
variables and serves it over MCP Streamable HTTP. This is the entry point the
reference container (``deploy/Dockerfile``) runs, and the server an ``MCPServer``
custom resource points at (see ``kaos_manifests.yaml``).

Requires the MCP extra::

    pip install 'green-sarc[mcp]'

Configuration (all optional, with conservative defaults):

==============================  ==========================================
Environment variable            Meaning
==============================  ==========================================
GREEN_SARC_TOKEN_BUDGET         remaining token budget b_tok
GREEN_SARC_CARBON_CEILING       carbon ceiling B_co2 (gCO2e)
GREEN_SARC_DELTA                gate confidence parameter (admit at 1 - delta)
GREEN_SARC_CARBON_INTENSITY     default grid intensity kappa (gCO2e/kWh)
GREEN_SARC_NAME                 MCP server name (default: green-sarc)
==============================  ==========================================

The dependency runs one way: KAOS -> Green SARC. Nothing here imports KAOS.
"""

from __future__ import annotations

import os
from typing import Any

from green_sarc import Budget, LearnedEstimator, TableCarbonModel, TableCostModel
from green_sarc.adapters.mcp import GreenSarcMCPService, build_mcp_server


def build_server() -> Any:
    """Construct the configured Green SARC MCP server from the environment."""
    budget = Budget(
        token_budget=float(os.environ.get("GREEN_SARC_TOKEN_BUDGET", "1000000")),
        carbon_ceiling=float(os.environ.get("GREEN_SARC_CARBON_CEILING", "100000")),
        delta=float(os.environ.get("GREEN_SARC_DELTA", "0.05")),
    )
    cost_model = TableCostModel()
    carbon_model = TableCarbonModel(
        default_intensity=float(os.environ.get("GREEN_SARC_CARBON_INTENSITY", "400"))
    )
    service = GreenSarcMCPService(
        budget=budget,
        estimator=LearnedEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
    )
    return build_mcp_server(service, name=os.environ.get("GREEN_SARC_NAME", "green-sarc"))


if __name__ == "__main__":
    # FastMCP serves MCP Streamable HTTP; KAOS connects via MCP_SERVER_<NAME>_URL.
    build_server().run(transport="streamable-http")
