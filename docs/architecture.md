# Architecture

Green SARC governs an agent's actions through **four enforcement sites** — the
backbone it borrows from the SARC framework (see
[relationship-to-sarc.md](relationship-to-sarc.md)). The core is
framework-agnostic; orchestrators integrate as adapters
([kaos-integration.md](kaos-integration.md)).

## The augmented state

Green SARC reasons over the agent's live cost/carbon state
([`state.py`](../src/green_sarc/state.py)):

- `b_tok` — remaining token budget (live, decrementing).
- `B_co2` / `carbon_spent` — carbon ceiling and running spend (gCO2e).
- `kappa(rho, t)` — carbon intensity (gCO2e/kWh) for region `rho` at time `t`
  ([`pricing.py`](../src/green_sarc/pricing.py)).
- `delta_lat` — latency / SLA headroom.
- `delta` — gate confidence parameter; the gate admits at confidence `1 − delta`.

## The four enforcement sites

```
   proposed action
        │
        ▼
┌───────────────────┐   reject / escalate
│ SITE 1  Pre-Action│──────────────────────────────► (HTTP 429 / down-route / human)
│ Gate (PAG)        │   forecast ≤ budget at 1−delta?
└─────────┬─────────┘   carbon ≤ ceiling?
          │ admit
          ▼
┌───────────────────┐   loop/cost limit crossed
│ SITE 2  Action-   │──────────────────────────────► CircuitTripped → escalate
│ Time Monitor (ATM)│   circuit breaker
└─────────┬─────────┘
          │ execute (caller's coroutine runs the model/tool)
          ▼
┌───────────────────┐
│ SITE 3  Post-     │   write AuditRecord(predicted, actual); estimator.update(...)
│ Action Auditor    │
│ (PAA)             │
└─────────┬─────────┘
          │ budget/carbon exhausted?
          ▼
┌───────────────────┐
│ SITE 4  Escalation│   route to human review / deterministic fallback (best-effort)
│ Router (ER)       │
└───────────────────┘
```

[`GreenGovernor.run_action`](../src/green_sarc/governor.py) is the single async
path an action travels through all four sites. The caller supplies an `execute`
coroutine that actually runs the action and reports its real token usage — this
is the seam the KAOS adapters plug into. Nothing in the governor imports an
orchestrator.

### SITE 1 — Pre-Action Gate ([`gate.py`](../src/green_sarc/gate.py))
Runs the estimator on the proposed action and admits it iff:

- `P[cost_hat ≤ b_tok] ≥ 1 − delta` — a one-sided upper bound on the forecast
  token cost (the `1 − delta` normal quantile when the estimator supplies a
  standard deviation; the point estimate treated as worst case otherwise), and
- `carbon_hat ≤ B_co2 − carbon_spent`.

Otherwise it rejects, down-routes, or — when budget/carbon is already exhausted —
escalates.

### SITE 2 — Action-Time Monitor ([`monitor.py`](../src/green_sarc/monitor.py))
A circuit breaker checked twice per action: `before()` (loop-count limit) and
`after(marginal_cost)` (marginal- and cumulative-cost limits). It kills runaway
retry / re-plan loops by raising `CircuitTripped`.

### SITE 3 — Post-Action Auditor ([`auditor.py`](../src/green_sarc/auditor.py))
Writes an [`AuditRecord`](../src/green_sarc/auditor.py) of predicted vs actual
cost and carbon and feeds it back into `estimator.update(...)`. This record is
**both** the ESG/audit log and the estimator's training data.

### SITE 4 — Escalation Router ([`escalation.py`](../src/green_sarc/escalation.py))
Dispatches escalation events (token/carbon exhaustion, gate reject, circuit trip)
to a pluggable async handler — log-only by default, or a deterministic fallback.
Best-effort: a handler that raises is logged and suppressed so it can never break
the agent loop.

## The learning loop: predict → act → log → retrain

This loop is the point of the system:

1. **predict** — the gate asks the estimator for `(cost_hat, carbon_hat,
   confidence)` and admits only if it fits the budget.
2. **act** — the admitted action runs; the executor reports real token usage.
3. **log** — the auditor records predicted vs actual.
4. **retrain** — that record updates the estimator, improving the next forecast.

With **no history**, the estimator cold-starts to the **zero-information gate**:
a conservative worst-case static threshold
([`ColdStartEstimator`](../src/green_sarc/estimator.py)). This static rule is the
degenerate, zero-information case — the fallback, not the primary mechanism. As
actuals accumulate, [`LearnedEstimator`](../src/green_sarc/estimator.py) takes
over per `(action kind, model)` key.

## Model-agnostic by design

The estimator predicts against **any** LLM. The caller supplies a pricing +
carbon table ([`CostModel` / `CarbonModel`](../src/green_sarc/pricing.py)) and
carbon is computed as `energy_kwh(model, tokens) × kappa(region, t)`. Nothing is
wired to a single provider.

## Phase 1 / Phase 2

- **Phase 1 (implemented): per-action estimation.**
  `estimator.predict(action, context) → Forecast`.
- **Phase 2 (interface stub only): trajectory estimation.** Predict the cost of
  an entire *plan* before the agent starts, to reject expensive plans rather than
  just expensive steps. It trains only on Phase 1's logged trajectories, so it
  cannot be built until Phase 1 has produced data. The interface is fixed in
  [`trajectory.py`](../src/green_sarc/trajectory.py) and raises
  `NotImplementedError`.
