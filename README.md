# Green SARC

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20692197.svg)](https://doi.org/10.5281/zenodo.20692197)

**Predictive cost + carbon governance for agentic AI systems.**

Green SARC wraps an agent's execution loop and decides, in real time, whether a
proposed action fits the **remaining token budget** and **carbon ceiling** — and
it *forecasts* that cost **before** the action fires rather than reconciling it
after. It is an application of the SARC governance architecture's four
enforcement sites to the FinOps / GreenOps domain.

> Status: alpha (Phase 1). The core is framework-agnostic and runs standalone.
> KAOS integrates as the caller through an adapter; the dependency runs one way.
> As of v0.3.0 the Pre-Action Gate can admit on a distribution-free
> split-conformal / adaptive-conformal bound (opt-in via `calibrator=...`); the
> default Normal-σ behaviour is unchanged.

### Releases

- v0.3.0: runtime conformal calibration (`PreActionGate(calibrator=...)`), real-arrival ablation on BurstGPT, real-grid sensitivity (ElectricityMaps IT + US-CAISO), joint sensitivity grid, multi-step trajectory ablation on SWE-rebench OpenHands, adversarial threat model. Paper companion at tag `v0.3.0`.

### Headline result

The predictive gate is a *learned forecaster*, not a static rule. On the runnable
[`examples/learning_curve`](examples/learning_curve/run_demo.py) workload, the
token-cost forecast error collapses as the estimator learns — while a USD budget
and carbon ceiling are enforced in the same loop:

| Forecast stage | MAE (token cost) |
|---|---|
| Cold start (zero-information) | **≈ 645** |
| After ~30 actions (learned) | **≈ 12** |

A ~50× reduction — this is the `predict → act → log → retrain` loop working.

**Overhead:** the gate adds **p99 ≈ 3.7 µs** per decision on the default
Normal-σ path (~0.5 M decisions/sec; `benchmarks/gate_overhead.py`) — negligible
beside any model call.

And on the reproducible §8 **IBP benchmark** (`make reproduce`, 20 seeds, 400 SKUs,
baseline State-Snowball vs. Green SARC), enforcement placement — Adapter-Node
state scoping + energy-aware routing + the circuit breaker — yields:

| Metric | Reduction vs. baseline |
|---|---|
| Total tokens | **−47%** |
| Total USD | **−68%** |
| Carbon (fixed & time-varying κ) | **−67%** |

…with **0 gate rejections** (the savings come from *where* enforcement sits, not
from chance rejection) and forecast WAPE ≈ 4.5% (95% CI on token reduction
[46.1%, 48.6%], paired bootstrap). An **ablation** isolates each lever: state
scoping alone is −43% tokens, energy-aware routing drives the USD/carbon cut, and
the circuit breaker adds the rest. Run `make reproduce`; a reference run is
checked in at [`benchmarks/reference_summary.json`](benchmarks/reference_summary.json).

**Reproducibility.** Run `make verify` to reproduce the 20-seed ablation and check
the headline numbers against `benchmarks/reference_summary.json` (2% tolerance per
condition × metric, plus the `+full` token reduction within 1.5 pp). CI runs this
on every push, so any change that drifts the numbers fails.

## Documentation

Full docs live in [`docs/`](docs/):

- [Architecture](docs/architecture.md) — the four enforcement sites and the `predict → act → log → retrain` loop.
- [Relationship to SARC](docs/relationship-to-sarc.md) — what is borrowed from the **SARC** framework ([`besanson/sarc-governance`](https://github.com/besanson/sarc-governance)) and what is not.
- [KAOS integration](docs/kaos-integration.md) — how **KAOS** ([`axsaucedo/kaos`](https://github.com/axsaucedo/kaos)) calls Green SARC, across all three surfaces, with deployment.
- [**Use it**](docs/usage.md) — govern your own agent in 5 minutes (3-line setup).
- [Quickstart](docs/quickstart.md) — install, run, govern your own loop.
- [Metrics](docs/metrics.md) — Prometheus counters/gauges/latency histograms (optional `prometheus` extra) + a ready-to-import Grafana dashboard.
- [Live feeds](docs/feeds.md) — load real LiteLLM prices and live ElectricityMaps carbon intensity (stdlib-only, no extra required).

**Related repositories** (Green SARC depends on neither at runtime):
[SARC framework](https://github.com/besanson/sarc-governance) ·
[KAOS orchestrator](https://github.com/axsaucedo/kaos) ·
[PAIS runtime](https://github.com/axsaucedo/pydantic-ai-server)

---

## What it governs (and what it does not)

- ✅ **Cost and carbon only.** Token cost (`b_tok`), an optional **USD** budget, and carbon (`gCO2e`).
- ❌ **Not correctness, safety, or output quality.** Those are out of scope by
  design — Green SARC never tracks accuracy or quality as a governed quantity.
- ✅ **Standalone.** No dependency on SARC or any safety framework. It may
  *compose* with one (shared enforcement sites) but never requires it.
- ✅ **Predictive, not rule-based.** The gate admits actions on a *learned
  forecast*. The static-threshold rule is the degenerate zero-information case,
  implemented as the cold-start fallback — not the primary mechanism.

## The four enforcement sites

Green SARC borrows its structural backbone — four enforcement sites — from the
SARC framework and specialises each for cost/carbon:

| Site | Module | Role |
|---|---|---|
| **Pre-Action Gate** (PAG) | `gate.py` | Runs the predictive estimator on a proposed action; admits it only if forecast cost fits the remaining budget at confidence `1 - delta` **and** forecast carbon fits the remaining ceiling. Otherwise reject / down-route / escalate. |
| **Action-Time Monitor** (ATM) | `monitor.py` | Circuit breaker. Tracks loop count and marginal/total cost during execution; kills runaway retry / re-plan loops once a threshold is crossed. |
| **Post-Action Auditor** (PAA) | `auditor.py` | Logs **predicted vs actual** cost and carbon per action. This log is both the ESG/audit record **and** the estimator's training data. |
| **Escalation Router** (ER) | `escalation.py` | When budget or carbon is exhausted, routes to human review or a deterministic fallback instead of allowing silent overspend. Best-effort: a broken handler never breaks the agent loop. |

`GreenGovernor` (`governor.py`) wires all four around an arbitrary async
executor:

```python
import asyncio
from green_sarc import Action, ActionOutcome, GateRejected, GreenGovernor

# One line: wires the four sites + reference pricing/carbon tables. Budgets are yours.
gov = GreenGovernor.with_defaults(token_budget=10_000, usd_budget=0.50)

async def call_model(action: Action) -> ActionOutcome:
    # ... run the real model / tool, then report its actual usage ...
    return ActionOutcome(result="...", actual_tokens=240)

async def main():
    action = Action(kind="chat.completion", model="gpt-4o", region="us-east-1",
                    prompt_tokens=120, max_tokens=180)
    try:
        result = await gov.run_action(action, call_model)
        print(result.actual_cost, result.audit.cost_error, result.audit.actual_usd)
    except GateRejected as exc:
        print("blocked:", exc.decision.reason)  # too expensive: down-route / cheaper model

asyncio.run(main())
```

See [docs/usage.md](docs/usage.md) for the full 5-minute guide and a runnable
OpenAI-compatible example ([`examples/openai_governed`](examples/openai_governed/run_demo.py)).

## The learning loop: predict → act → log → retrain

This loop is the point of the system:

1. **predict** — the Pre-Action Gate asks the estimator for `(cost_hat,
   carbon_hat, confidence)` and admits the action only if it fits the budget.
2. **act** — the admitted action runs; the executor reports its real token usage.
3. **log** — the Post-Action Auditor writes an `AuditRecord` of predicted vs
   actual (the ESG record).
4. **retrain** — the same record is fed back into `estimator.update(...)`, so the
   next forecast is better.

With **no history**, the estimator cold-starts to the **zero-information gate**: a
conservative worst-case static threshold (`ColdStartEstimator`). As actuals
accumulate, `LearnedEstimator` takes over per `(action kind, model)` key.

## Phase 1 / Phase 2

- **Phase 1 (this release): per-action estimation.** Predict the cost of the
  *next action* only. `estimator.predict(action, context) -> Forecast`.
- **Phase 2 (interface stub only): trajectory estimation.** Predict the cost of
  an *entire plan* before the agent starts, enabling rejection of expensive
  *plans*, not just expensive *steps*. It is trainable only on Phase 1's logged
  trajectories, so it cannot be built until Phase 1 has produced data. The
  interface is fixed in `trajectory.py` and raises `NotImplementedError`.

The single-process `Budget` (thread-safe via `threading.Lock`) is authoritative
for one replica. For multi-replica deployments an **experimental** distributed
backend ships in Phase 1: `green_sarc.backends.RedisBudget` (optional `redis`
extra) — one atomic Lua script per reserve/commit/release, with TTL reclamation
of crashed-client reservations; atomic against a single Redis, no cross-region
reconciliation or fair-share yet (a Postgres durable ledger + fair-share are
Phase 2).

## The estimator is model-agnostic

The estimator predicts against **any** LLM: the caller supplies a pricing +
carbon table (`CostModel` / `CarbonModel` in `pricing.py`), and carbon is
computed as `energy_kwh(model, tokens) * kappa(region, t)` where `kappa` is the
region's carbon intensity (gCO2e/kWh). Ship the defaults, or override per model
and per region. Nothing is wired to a single provider.

## How KAOS integrates

[KAOS](https://github.com/axsaucedo/kaos) is a Kubernetes-native agent
*orchestration* framework. It is the **caller**: KAOS orchestrates agents and
Green SARC governs each agent action. **The dependency runs one way — KAOS →
Green SARC — and the core never imports KAOS.** The integration is an adapter on
top of the framework-agnostic core (`green_sarc/adapters/`).

KAOS is MCP-native (agents consume tools via the Model Context Protocol), so the
**primary adapter is an MCP server** (`adapters/mcp.py`) exposing two tools:

- `pre_action_gate(...)` — forecast a proposed action and return an admit/reject
  verdict plus predicted cost and carbon;
- `post_action_auditor(...)` — report the action's actual token usage to close
  the audit loop and retrain the estimator.

Deploying it requires **no change to KAOS or PAIS** — Green SARC registers as an
ordinary `MCPServer` custom resource and an agent lists it under
`spec.mcpServers` (see `examples/kaos_mcp_adapter/kaos_manifests.yaml`).

Two complementary surfaces:

- **PAIS sidecar / middleware** (`adapters/pais_sidecar.py`) — for
  **non-bypassable** gating. The MCP tool surface is agent-invoked (advisory); an
  agent could simply not call it. The sidecar is *hard*: it sits as ASGI
  middleware in front of PAIS's `/v1/chat/completions` endpoint so **every** model
  call is gated, returning HTTP `429` on rejection so the call never reaches the
  model, and reading actual token usage from the response to audit. Pure ASGI —
  no web-framework dependency. (`GreenSarcASGIMiddleware` wraps the PAIS `app`;
  `SidecarGate` is the testable core.)
- **OTel actuals feed** (`adapters/otel.py`) — KAOS/PAIS emit per-request
  OpenTelemetry spans carrying real token usage; the Post-Action Auditor can be
  fed from that stream. The span→actuals mapping is implemented and testable; the
  live OTLP receiver is a documented stub.

> Choosing a surface: use the **MCP server** for the lightest-touch, MCP-native
> integration (zero infra change, advisory gating); use the **sidecar** when you
> need hard, unbypassable enforcement at the model-call boundary. Both wrap the
> same framework-agnostic core. Note PAIS currently reports `usage` as zero, so
> the sidecar falls back to a length-based estimate and the OTel span is the best
> source of real actuals.

```text
        ┌────────── KAOS (orchestrator, K8s) ──────────┐
        │  Agent CR  ──spec.mcpServers: [green-sarc]──┐ │
        └─────────────────────────────────────────────┼─┘
                                                       │  MCP (one way)
                                     ┌─────────────────▼─────────────────┐
                                     │  Green SARC MCP server (adapter)  │
                                     │  pre_action_gate / post_action_…  │
                                     └─────────────────┬─────────────────┘
                                                       │
                          ┌────────────────────────────▼───────────────────────────┐
                          │  Green SARC core (framework-agnostic, standalone)       │
                          │  PAG · ATM · PAA · ER · Estimator · Budget · AuditStore │
                          └─────────────────────────────────────────────────────────┘
```

## Install

```bash
pip install -e ".[dev]"               # core + test/lint tooling
pip install -e ".[dev,mcp,otel,sarc]" # plus the KAOS adapters and the SARC composition adapter
```

Core has **no runtime dependencies**; `mcp`, `opentelemetry-sdk`, and
`sarc-governance` are optional extras pulled in only by their adapters.

## Run the examples

```bash
python examples/standalone_agent_loop/run_demo.py   # four sites: reject, breaker trip, audit log
python examples/kaos_mcp_adapter/run_demo.py        # KAOS agent driving the MCP gate + auditor (advisory)
python examples/pais_sidecar/run_demo.py            # sidecar hard-gating /v1/chat/completions (429)
python examples/sarc_composition/run_demo.py        # Green SARC as SARC constraints on one GovernanceToolset
python examples/learning_curve/run_demo.py          # forecast MAE drops as the estimator learns; USD budget enforced
```

## CLI

```bash
green-sarc inspect path/to/audit.jsonl   # predicted-vs-actual accuracy of a logged run
```

## Develop

```bash
make quality   # ruff (lint + format) · mypy · pytest
make test
```

## Layout

```
src/green_sarc/
  state.py        # Budget (b_tok, B_co2, delta_lat), Action, GovernanceContext
  forecast.py     # Forecast, Verdict, GateDecision
  pricing.py      # CostModel / CarbonModel protocols + table defaults; kappa(rho,t)
  estimator.py    # Estimator protocol; ColdStartEstimator, LearnedEstimator
  trajectory.py   # Phase-2 trajectory estimator stub (NotImplementedError)
  gate.py         # SITE 1  Pre-Action Gate
  monitor.py      # SITE 2  Action-Time Monitor (circuit breaker)
  auditor.py      # SITE 3  Post-Action Auditor + AuditRecord schema
  escalation.py   # SITE 4  Escalation Router + handlers
  governor.py     # GreenGovernor: wires the four sites around an async executor
  stores/         # AuditStore protocol + memory / JSONL backends
  adapters/       # KAOS-facing: mcp.py (MCP server), pais_sidecar.py (hard gate), otel.py
  cli.py          # `green-sarc inspect`
examples/         # standalone_agent_loop/, kaos_mcp_adapter/
tests/            # one test_*.py per module
```

## Relationship to SARC

Green SARC borrows the **four-enforcement-site architecture** from the SARC
framework and nothing more. SARC is the framework this layer takes those four
sites from (arXiv:2605.07728,
[github.com/besanson/sarc-governance](https://github.com/besanson/sarc-governance)).
Green SARC's core has no dependency on SARC and governs only cost and carbon. It
can, optionally, **compose** with SARC on a single governed toolset via
[`green_sarc/adapters/sarc.py`](src/green_sarc/adapters/sarc.py): the predictive
gate and auditor are expressed as SARC constraints at the shared `PAG`/`PAA`
sites, so one `GovernanceToolset` enforces both safety and cost/carbon (requires
`pip install 'green-sarc[sarc]'`). See
[docs/relationship-to-sarc.md](docs/relationship-to-sarc.md).

## Reference

Besanson, G. (2026). *Green SARC: Predictive FinOps as Governance-by-Architecture
for Agentic AI Systems.* Working paper. The paper source is in
[`paper/green-sarc.tex`](paper/green-sarc.tex) (see [`paper/`](paper/) for how it
maps to the code).

## License

MIT — see [LICENSE](LICENSE).
