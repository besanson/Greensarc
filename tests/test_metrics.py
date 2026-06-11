"""Metrics sink: NullSink no-ops, PrometheusSink counts through the governor."""

from __future__ import annotations

import pytest

from green_sarc.estimator import ColdStartEstimator
from green_sarc.governor import ActionOutcome, GateRejected, GreenGovernor
from green_sarc.metrics import (
    GATE_ADMITTED,
    GATE_DECISION_SECONDS,
    GATE_REJECTED,
    NullSink,
)
from green_sarc.monitor import ActionTimeMonitor
from green_sarc.pricing import ModelProfile, TableCostModel
from green_sarc.state import Budget

from .conftest import make_action


def _cost_model() -> TableCostModel:
    return TableCostModel(
        default_profile=ModelProfile(
            energy_per_token_kwh=1.0e-6,
            usd_per_prompt_token=1.0e-5,
            usd_per_completion_token=2.0e-5,
        )
    )


def test_nullsink_is_a_noop():
    sink = NullSink()
    # None of these raise and none of them return anything meaningful.
    sink.incr(GATE_ADMITTED)
    sink.incr(GATE_REJECTED, reason="tokens")
    sink.gauge("budget_tokens_remaining", 123.0)
    sink.observe(GATE_DECISION_SECONDS, 0.0001)


def _governor(budget: Budget, sink, carbon_model) -> GreenGovernor:
    cost_model = _cost_model()
    return GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        monitor=ActionTimeMonitor(max_loops=100),
        metrics=sink,
    )


async def test_prometheus_sink_counts_admit(carbon_model):
    prom = pytest.importorskip("prometheus_client")  # noqa: F841
    from green_sarc.metrics import PrometheusSink

    sink = PrometheusSink()
    budget = Budget(token_budget=1_000_000.0, carbon_ceiling=1.0e9, usd_budget=100.0)
    gov = _governor(budget, sink, carbon_model)

    async def execute(action):
        return ActionOutcome(result="ok", actual_tokens=250.0)

    await gov.run_action(make_action(prompt=100, max_tokens=200), execute)

    reg = sink.registry
    assert reg.get_sample_value("green_sarc_gate_admitted_total") == 1.0
    # budget gauge reflects the post-action remaining tokens (started at 1e6).
    remaining = reg.get_sample_value("green_sarc_budget_tokens_remaining")
    assert remaining is not None and remaining < 1_000_000.0
    # one decision-latency observation was recorded.
    assert reg.get_sample_value("green_sarc_gate_decision_seconds_count") == 1.0
    # forecast error histogram saw one sample.
    assert reg.get_sample_value("green_sarc_forecast_abs_error_tokens_count") == 1.0


async def test_prometheus_sink_counts_reject(carbon_model):
    pytest.importorskip("prometheus_client")
    from green_sarc.metrics import PrometheusSink

    sink = PrometheusSink()
    # A USD ceiling far below the cold-start forecast forces a rejection.
    budget = Budget(token_budget=1_000_000.0, carbon_ceiling=1.0e9, usd_budget=1.0e-9)
    gov = _governor(budget, sink, carbon_model)

    async def execute(action):
        return ActionOutcome(result="ok", actual_tokens=250.0)

    with pytest.raises(GateRejected):
        await gov.run_action(make_action(prompt=100, max_tokens=200), execute)

    reg = sink.registry
    assert reg.get_sample_value("green_sarc_gate_rejected_total", {"reason": "usd"}) == 1.0
    assert reg.get_sample_value("green_sarc_gate_admitted_total") in (0.0, None)
