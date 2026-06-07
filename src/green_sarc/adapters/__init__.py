"""Orchestrator-facing adapters.

These are thin integrations layered on top of the framework-agnostic core.  The
dependency runs one way — the orchestrator (KAOS) calls Green SARC — and nothing
here is imported by the core.  Importing an adapter never requires its optional
third-party dependency until you actually build the server/consumer.

- :mod:`green_sarc.adapters.mcp`  — Green SARC as an MCP server exposing the
  Pre-Action Gate and Post-Action Auditor as MCP tools (KAOS is MCP-native).
- :mod:`green_sarc.adapters.otel` — consume an OpenTelemetry actuals stream to
  feed the Post-Action Auditor's predicted-vs-actual log.
"""

from __future__ import annotations

__all__: list[str] = []
