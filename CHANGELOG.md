# Changelog

All notable changes to this project are documented here.

## [Unreleased]

Senior-review hardening (audit items P0/P1).

### Fixed (P0 — correctness / safety)
- **P0-1** Atomic `Budget` reserve/commit closes a concurrent over-spend race; the
  governor and all three adapters reserve the forecast at the gate and commit
  actuals at the auditor.
- **P0-2** `TTLMap` bounds the adapter correlation maps by age and size (was an
  unbounded-dict memory leak); eviction releases any held reservation.
- **P0-3** Time-varying carbon intensity `kappa(rho, t)` (interpolated series +
  pluggable `IntensityProvider`); the `t` argument is no longer ignored.
- **P0-4** `LearnedEstimator` regresses completion tokens on prompt length and
  exposes the residual std to the gate.
- **P0-5** `EscalationRouter.route` returns a `RouteOutcome` instead of silently
  swallowing handler errors.
- **P0-6** Sidecar streams responses through (SSE-safe), parsing usage on the fly
  and auditing after the stream ends; non-streaming bodies use a bounded buffer.

### Added (P1 — production readiness)
- **P1-2** Optional **USD budget** enforced at the gate alongside tokens and carbon;
  `Budget.usd_budget`, predicted/actual USD on the audit record, USD in `inspect`.
- **P1-1** Estimator `save`/`load` and `bootstrap_from_jsonl`; `green-sarc bootstrap`.
- **P1-4** Optional `tiktoken` prompt-token counter (`tiktoken` extra).
- **P1-6** `SQLiteAuditStore` for durable, queryable runs.
- **P1-8** `strict` mode for the pricing/carbon tables (raise vs. warn-once).
- **P1-9** `AuditRecord` plan/session/parent ids; `JSONLTrajectoryStore` groups the
  log into per-plan trajectories (the Phase-2 data engine).
- **P1-10** Sidecar path matched by regex (`GREEN_SARC_PATH_REGEX`).
- Docs: `SECURITY.md`, `CONTRIBUTING.md`.

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
- KAOS adapters (one-way dependency, KAOS → Green SARC):
  - **MCP server** (`adapters/mcp.py`) exposing the gate and auditor as MCP tools (advisory).
  - **PAIS sidecar** (`adapters/pais_sidecar.py`) — dependency-free ASGI middleware that
    hard-gates `/v1/chat/completions`, returning HTTP 429 on rejection and auditing actuals
    from the response. `GreenSarcASGIMiddleware` wraps the PAIS app; `SidecarGate` is the
    testable core.
  - **OpenTelemetry** actuals-consumer (`adapters/otel.py`): span→actuals mapping plus a
    working in-process `GreenSarcSpanProcessor` (duck-typed against OpenTelemetry's
    `SpanProcessor`, so dependency-free and testable) that feeds ended spans to the auditor;
    cross-process OTLP collector ingestion remains a documented stub.
  - **SARC composition** (`adapters/sarc.py`): expresses the predictive gate and auditor as
    SARC `Constraint`s (HARD at `PAG`, SOFT at `PAA`) plugged into a SARC `GovernanceToolset`,
    so one toolset enforces safety and cost/carbon at the shared sites. Optional `sarc` extra;
    the core never imports SARC.
- Runnable examples: a standalone four-site agent loop, a KAOS MCP-adapter demo, a
  PAIS sidecar demo gating a mock `/v1/chat/completions` endpoint, and a SARC composition demo.
- Documentation (`docs/`): architecture, the relationship to the SARC framework, and a full
  KAOS integration guide — all cross-referenced to the upstream
  [SARC](https://github.com/besanson/sarc-governance),
  [KAOS](https://github.com/axsaucedo/kaos), and
  [PAIS](https://github.com/axsaucedo/pydantic-ai-server) repositories.
- Deployment reference: `deploy/Dockerfile` and an env-configured MCP server entry point
  (`examples/kaos_mcp_adapter/server.py`) registered via the `MCPServer` + `Agent` manifests.
