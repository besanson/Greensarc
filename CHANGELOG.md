# Changelog

All notable changes to this project are documented here.

## [Unreleased]

Third-pass audit follow-ups.

### Changed
- Real-grid data source upgraded from UK ESO to ElectricityMaps IT + US-CAISO;
  §11.5 now uses the two zones originally targeted in the brief. Fetched via
  `paper/scripts/fetch_grid.py` (key from `ELECTRICITYMAPS_API_KEY`); cached
  hourly CSVs committed under `paper/data/grid/` so §11.5 reproduces without API
  access.

### Added
- `make verify` / `python -m benchmarks.reproduce --verify REF` — reproduces the
  20-seed ablation and gates the headline numbers against
  `benchmarks/reference_summary.json` (2% relative tolerance per condition × metric,
  `+full` token reduction within 1.5 pp; override via `GREEN_SARC_VERIFY_TOL`). CI
  runs it on every push (3.12), and `release.yml` runs the full test matrix
  (3.11 + 3.12) before publishing.
- Paper upgraded to an arXiv preprint: split-conformal gate-safety theorem,
  paired-bootstrap ablation CIs, cold-start/calibration/sensitivity figures, and
  a soft-penalty-vs-gate comparison — all from reproducible data assets
  (`paper/scripts/gen_data.py`, `paper/scripts/build_figures.py`, `make paper`);
  `examples/learning_curve --emit-json` emits the cold-start curve.
- Paper §11 real-arrival ablation on the BurstGPT Azure GPT-3.5/GPT-4 trace
  (`paper/scripts/run_real_arrival.py`, `make paper-real-arrival`): the four-condition
  ablation reproduces the synthetic savings ordering on real arrivals (token/USD/carbon
  −55.7% / −55.0% / −67.4%), with a binding-budget companion to §9.

### Changed
- Benchmark now runs an **ablation** (`baseline → +scope → +scope+route → +full`) so
  each governance lever's contribution is isolated, with a paired-bootstrap 95% CI on
  the token reduction; a reference run is checked in at `benchmarks/reference_summary.json`
  (F-5, F-6).
- Reference pricing normalises real/dated model ids (`gpt-4o-2024-08-06 → gpt-4o`,
  `claude-3-5-sonnet-* → claude-sonnet`) via `canonical_model_id`, wired through an
  optional `TableCostModel.alias` hook (F-1); source citations added to `data.py` (F-2).
- `AdapterNode`: canonical `bound(tokens)`; new `scope_messages(...)` to truncate a real
  messages array; `scope` kept as an alias (F-3, F-4).
- `GreenGovernor.with_defaults` gains `estimator` and `bootstrap_jsonl` kwargs (F-10).
- `LearnedEstimator._stats` guarded by a `threading.Lock` (N-1); OpenAI example prints a spend summary
  (F-12); `release.yml` gates publish on the test job (F-13); `benchmarks` is now
  type-checked (F-8); runaway-SKU parameter documented as a stress scenario (F-7).

## [0.3.0] — 2026-06-09

### Added
- **Conformal calibration in the runtime gate.** New `green_sarc.calibrator`
  module (`Calibrator` protocol, `SplitConformal`, `ACIConformal`). `PreActionGate`
  gains an optional `calibrator=...` argument: supplying it replaces the
  Normal-σ token bound with a distribution-free conformal bound (working paper
  Theorem 2); omitting it preserves the existing behaviour exactly. Default
  behaviour unchanged; `make verify` holds. 9 new tests (split coverage on
  Gaussian/Pareto/mixture residuals, ACI coverage restoration under shift, and a
  gate/ShareGPT-style integration test). Re-exported from the package root.
- Paper companion: `paper/green-sarc.tex` at this tag is the exact source for all
  cited numbers; `python paper/scripts/check_stats.py` is the cross-check linter.

## [0.2.0] — 2026-06-08

Senior-review hardening (audit items P0/P1), the §8 benchmark, and out-of-the-box
usability (reference data, one-call constructor, real-LLM example).

### Added (usability)
- Reference pricing + carbon data (`data.py`): approximate list prices for common
  models and grid intensity for common regions, so outputs mean something out of the box.
- `GreenGovernor.with_defaults(...)` — a one-call constructor (budgets in, governor out).
- `examples/openai_governed/` — govern a real OpenAI-compatible agent loop; `docs/usage.md`.
- `release` workflow for building and PyPI publishing on tag (OIDC trusted publishing).

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

### Added (working paper §8)
- `AdapterNode` (`scoping.py`) — bounded state scoping that caps the per-hop prompt
  growth `p`, collapsing the State-Snowball `Θ(depth²)` cost toward linear.
- `benchmarks/` — reproducible synthetic **IBP** benchmark (baseline State-Snowball vs.
  Green SARC over N seeds) exercising the real gate/estimator/budget/breaker; reports
  token/USD/carbon reductions (≈ −47% / −68% / −67%), breaker trips, and forecast
  MAE/WAPE. `make reproduce` · `make benchmark-smoke`.

### Hardening (second-pass audit follow-ups)
- `LearnedEstimator` guards `_stats` with a `threading.Lock` (safe under a threaded
  server, not just a single event loop).
- SQLite store: single held connection (`check_same_thread=False`) with WAL, a write
  lock, indexes on `timestamp` and `(plan_id, timestamp)`, and a `meta.schema_version`.
- Sidecar fails closed when a worst-case reservation would exceed the remaining budget
  (explicit reject reason; never an unguarded spend).
- `PostActionAuditor.record` warns once when `prompt_tokens=0` accompanies a large
  `actual_cost` (the regression would otherwise degenerate at x=0).
- SARC adapter documents its `action_factory` / `usage_extractor` override hooks and the
  predicate `ctx` contract.
- Docs: Mermaid sequence diagrams for the three adapter paths; README headline result
  (MAE ≈ 645 → 12); a `paper` workflow renders the working paper to a PDF artifact.

### Deferred (intentional — see roadmap)
Infrastructure/release items deferred until there is a deployment or release that
needs them, rather than shipping unused machinery: Prometheus `/metrics` endpoints
(P1-3), a standalone OTLP receiver (P1-5; the in-process `GreenSarcSpanProcessor`
covers the common case), transport-level MCP auth (P1-7; threat documented in
`SECURITY.md`), a PyPI trusted-publishing workflow (P2-3), and `mypy --strict`
(P2-6).

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
