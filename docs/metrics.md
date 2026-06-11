# Operational metrics

Green SARC's OpenTelemetry adapter exports **spans** (one per action). This page
covers the **metrics** path: counters, gauges, and latency histograms an
operator scrapes to watch gate decisions, budget burn-down, forecast error, and
circuit-breaker activity in real time.

The core stays dependency-free. `MetricsSink` is a stdlib-only `Protocol`, and
the default `NullSink` no-ops — the hot path is untouched unless you opt in.

## Wiring a sink

```python
from green_sarc import GreenGovernor, PrometheusSink

gov = GreenGovernor.with_defaults(token_budget=200_000, usd_budget=5.0)
gov.metrics = PrometheusSink()                       # emit to a fresh registry
# ...or expose on the global registry that start_http_server() scrapes:
import prometheus_client
gov.metrics = PrometheusSink(registry=prometheus_client.REGISTRY)
prometheus_client.start_http_server(9000)            # GET :9000/metrics
```

`PrometheusSink` requires the optional extra:

```bash
pip install "green-sarc[prometheus]"
```

By default each `PrometheusSink` owns a fresh `CollectorRegistry` so multiple
governors (and the test suite) coexist without `Duplicated timeseries` errors.
Pass `registry=prometheus_client.REGISTRY` for the single process-wide exporter.

## Metrics emitted

The `GreenGovernor` emits at the decision points it already observes in
`run_action`; supplying a sink never changes control flow.

| Series (`green_sarc_` prefix)          | Kind      | Labels   | Meaning |
|----------------------------------------|-----------|----------|---------|
| `gate_admitted_total`                  | counter   | —        | Actions the Pre-Action Gate admitted (and that won the budget reservation). |
| `gate_rejected_total`                  | counter   | `reason` | Actions blocked by the gate. `reason` ∈ `tokens` / `carbon` / `usd` / `exhausted` / `contention`. |
| `breaker_trips_total`                  | counter   | —        | Action-Time Monitor circuit-breaker trips (pre- or post-execution). |
| `escalations_total`                    | counter   | `reason` | Escalation Router dispatches; `reason` is the `EscalationReason` value. |
| `budget_tokens_remaining`              | gauge     | —        | Remaining token budget after the latest committed action. |
| `budget_usd_remaining`                 | gauge     | —        | Remaining USD budget (omitted when no USD budget is set). |
| `carbon_remaining_g`                   | gauge     | —        | Remaining carbon ceiling, gCO2e. |
| `forecast_abs_error_tokens`            | histogram | —        | Per-action `\|actual − predicted\|` token cost. |
| `gate_decision_seconds`                | histogram | —        | Pre-Action Gate decision latency. |

`contention` rejections are admitted-by-gate actions that lost a concurrent
budget reservation race — counted as rejections because the action did not run.

## Grafana

`deploy/grafana/green-sarc-dashboard.json` is a ready-to-import dashboard:
admit/reject rate, budget burn-down, carbon remaining, forecast mean absolute
error, breaker trips & escalations, and gate decision latency (p50/p99).

### A note on WAPE

The dashboard's "forecast error" panel reports the rolling **mean absolute
error** (`rate(..._sum) / rate(..._count)`), not WAPE. Proper WAPE
(`Σ|err| / Σactual`) needs a cumulative actual-tokens counter, which the current
metric set does not emit; mean absolute error is the faithful quantity from the
shipped instruments. Add an `actual_cost_tokens_total` counter if you need true
WAPE on the dashboard.

## Custom sinks

Implement the three primitives to forward metrics anywhere (StatsD, OTLP, logs):

```python
class LoggingSink:
    def incr(self, name, value=1.0, **labels): print("incr", name, value, labels)
    def gauge(self, name, value, **labels):    print("gauge", name, value, labels)
    def observe(self, name, value, **labels):  print("observe", name, value, labels)
```

Metric-name constants (`green_sarc.metrics.GATE_ADMITTED`, …) are exported so a
custom sink can switch on stable identifiers rather than string literals.
