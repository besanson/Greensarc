"""Operational metrics for the four enforcement sites (cross-site observability).

The OTel adapter exports *spans*; this module exports *metrics* — the counters,
gauges, and latency histograms an operator scrapes to watch gate admit/reject
rates, budget burn-down, forecast error, and circuit-breaker activity in real
time.

The core stays dependency-free.  :class:`MetricsSink` is a stdlib-only
:class:`~typing.Protocol`; the default :class:`NullSink` no-ops, so the hot path
is untouched unless an operator opts in.  :class:`PrometheusSink` (optional
``prometheus`` extra) maps the sink calls onto ``prometheus_client``
instruments.

The :class:`~green_sarc.governor.GreenGovernor` emits at the decision points it
already observes in ``run_action`` (admit/reject/escalate, breaker trip, the
post-action audit's predicted-vs-actual); supplying a sink never changes the
control flow.

Metric names (Prometheus exposition, with the ``green_sarc`` namespace prefix):

==============================  ======  ==============================  ==========================
logical name                    kind    Prometheus series               labels
==============================  ======  ==============================  ==========================
``gate_admitted_total``         counter ``green_sarc_gate_admitted_total``       --
``gate_rejected_total``         counter ``green_sarc_gate_rejected_total``       ``reason``
``breaker_trips_total``         counter ``green_sarc_breaker_trips_total``       --
``escalations_total``           counter ``green_sarc_escalations_total``         ``reason``
``budget_tokens_remaining``     gauge   ``green_sarc_budget_tokens_remaining``   --
``budget_usd_remaining``        gauge   ``green_sarc_budget_usd_remaining``      --
``carbon_remaining_g``          gauge   ``green_sarc_carbon_remaining_g``        --
``forecast_abs_error_tokens``   hist    ``green_sarc_forecast_abs_error_tokens`` --
``gate_decision_seconds``       hist    ``green_sarc_gate_decision_seconds``     --
==============================  ======  ==============================  ==========================

``reason`` on ``gate_rejected_total`` is one of ``tokens`` / ``carbon`` /
``usd`` / ``exhausted`` / ``contention``; on ``escalations_total`` it is the
:class:`~green_sarc.escalation.EscalationReason` value.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

__all__ = [
    "MetricsSink",
    "NullSink",
    "PrometheusSink",
    # logical metric names (stable identifiers used by the governor + dashboard)
    "GATE_ADMITTED",
    "GATE_REJECTED",
    "BREAKER_TRIPS",
    "ESCALATIONS",
    "BUDGET_TOKENS_REMAINING",
    "BUDGET_USD_REMAINING",
    "CARBON_REMAINING_G",
    "FORECAST_ABS_ERROR_TOKENS",
    "GATE_DECISION_SECONDS",
]

GATE_ADMITTED = "gate_admitted_total"
GATE_REJECTED = "gate_rejected_total"
BREAKER_TRIPS = "breaker_trips_total"
ESCALATIONS = "escalations_total"
BUDGET_TOKENS_REMAINING = "budget_tokens_remaining"
BUDGET_USD_REMAINING = "budget_usd_remaining"
CARBON_REMAINING_G = "carbon_remaining_g"
FORECAST_ABS_ERROR_TOKENS = "forecast_abs_error_tokens"
GATE_DECISION_SECONDS = "gate_decision_seconds"


@runtime_checkable
class MetricsSink(Protocol):
    """Where the governor emits operational metrics.

    Three primitives cover the metric kinds: ``incr`` (counters), ``gauge``
    (point-in-time values), and ``observe`` (histogram/summary samples).  Label
    values are passed as keyword arguments; an implementation that does not
    support a given metric or label simply ignores it.
    """

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Increment counter ``name`` by ``value`` (default 1)."""
        ...

    def gauge(self, name: str, value: float, **labels: str) -> None:
        """Set gauge ``name`` to ``value``."""
        ...

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record one ``value`` sample for histogram/summary ``name``."""
        ...


class NullSink:
    """Default sink: every call is a no-op (zero overhead, no dependencies)."""

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:  # noqa: D102
        pass

    def gauge(self, name: str, value: float, **labels: str) -> None:  # noqa: D102
        pass

    def observe(self, name: str, value: float, **labels: str) -> None:  # noqa: D102
        pass


class PrometheusSink:
    """Maps the sink primitives onto ``prometheus_client`` instruments.

    Requires the optional ``prometheus`` extra (``pip install
    green-sarc[prometheus]``).  By default the sink owns a fresh
    :class:`~prometheus_client.CollectorRegistry` so multiple governors (and the
    test suite) can coexist without ``Duplicated timeseries`` collisions; pass
    ``registry=prometheus_client.REGISTRY`` to expose on the global default
    registry that ``start_http_server`` / ``make_asgi_app`` scrape.
    """

    def __init__(self, namespace: str = "green_sarc", registry: Optional[Any] = None) -> None:
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

        self.registry = registry if registry is not None else CollectorRegistry()

        def _c(name: str, doc: str, labels: Optional[list] = None) -> Any:
            return Counter(f"{namespace}_{name}", doc, labels or [], registry=self.registry)

        def _g(name: str, doc: str) -> Any:
            return Gauge(f"{namespace}_{name}", doc, registry=self.registry)

        # Counter object names omit the `_total` suffix; prometheus_client adds it
        # to the exposed series automatically.
        self._counters: Dict[str, Any] = {
            GATE_ADMITTED: _c("gate_admitted", "Actions admitted by the Pre-Action Gate."),
            GATE_REJECTED: _c(
                "gate_rejected", "Actions rejected by the gate, by reason.", ["reason"]
            ),
            BREAKER_TRIPS: _c("breaker_trips", "Action-Time Monitor circuit-breaker trips."),
            ESCALATIONS: _c(
                "escalations", "Escalation Router dispatches, by reason.", ["reason"]
            ),
        }
        self._gauges: Dict[str, Any] = {
            BUDGET_TOKENS_REMAINING: _g("budget_tokens_remaining", "Remaining token budget."),
            BUDGET_USD_REMAINING: _g("budget_usd_remaining", "Remaining USD budget."),
            CARBON_REMAINING_G: _g("carbon_remaining_g", "Remaining carbon ceiling (gCO2e)."),
        }
        self._histograms: Dict[str, Any] = {
            FORECAST_ABS_ERROR_TOKENS: Histogram(
                f"{namespace}_forecast_abs_error_tokens",
                "Per-action |actual - predicted| token cost.",
                buckets=(1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, float("inf")),
                registry=self.registry,
            ),
            GATE_DECISION_SECONDS: Histogram(
                f"{namespace}_gate_decision_seconds",
                "Pre-Action Gate decision latency (seconds).",
                buckets=(1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, float("inf")),
                registry=self.registry,
            ),
        }

    def incr(self, name: str, value: float = 1.0, **labels: str) -> None:  # noqa: D102
        c = self._counters.get(name)
        if c is None:
            return
        (c.labels(**labels) if labels else c).inc(value)

    def gauge(self, name: str, value: float, **labels: str) -> None:  # noqa: D102
        g = self._gauges.get(name)
        if g is not None:
            g.set(value)

    def observe(self, name: str, value: float, **labels: str) -> None:  # noqa: D102
        h = self._histograms.get(name)
        if h is not None:
            h.observe(value)
