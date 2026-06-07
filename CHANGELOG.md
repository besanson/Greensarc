# Changelog

All notable changes to this project are documented here.

## [0.1.0] — 2026-06-07

Phase 1: per-action predictive cost + carbon governance.

### Added
- Four enforcement sites borrowed from the SARC architecture, specialised for FinOps/GreenOps:
  - **Pre-Action Gate** (`gate.py`) — admits an action only if the forecast token cost fits the
    remaining budget at confidence `(1 - delta)` and the forecast carbon fits the remaining ceiling.
  - **Action-Time Monitor** (`monitor.py`) — circuit breaker on loop count and marginal/total cost.
  - **Post-Action Auditor** (`auditor.py`) — logs predicted-vs-actual cost and carbon per action;
    the log is both the ESG/audit record and the estimator's training data.
  - **Escalation Router** (`escalation.py`) — routes to human review or a deterministic fallback on
    budget/carbon exhaustion; best-effort, never breaks the agent loop.
- Predictive per-action `Estimator` protocol with a zero-information `ColdStartEstimator` fallback and
  a `LearnedEstimator` that retrains on logged `(predicted, actual)` pairs (predict → act → log → retrain).
- Model-agnostic pricing/carbon table (`pricing.py`): `CostModel` / `CarbonModel` protocols with
  table-driven defaults; `kappa(rho, t)` carbon intensity.
- `GreenGovernor` (`governor.py`) wiring all four sites around an arbitrary async executor.
- Audit log persistence backends (`stores/`): in-memory and JSON Lines.
- Phase-2 `TrajectoryEstimator` interface stub (`trajectory.py`) — raises `NotImplementedError`.
- KAOS adapter: Green SARC as an **MCP server** exposing the gate and auditor as MCP tools
  (`adapters/mcp.py`), plus an OpenTelemetry actuals-consumer stub (`adapters/otel.py`).
- Runnable examples: a standalone four-site agent loop and a KAOS MCP-adapter demo.
