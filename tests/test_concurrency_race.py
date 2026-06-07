"""Concurrency tests for the reserve/commit budget protocol (audit P0-1).

Without reservation, two coroutines could each pass the gate (reading the same
remaining budget) and then both spend, over-admitting past the cap. The gate now
reserves the forecast upper bound atomically, so concurrent ``run_action`` calls
against a tight budget can never collectively over-spend.
"""

from __future__ import annotations

import asyncio

import pytest

from green_sarc.estimator import ColdStartEstimator
from green_sarc.governor import ActionOutcome, GateRejected, GreenGovernor
from green_sarc.monitor import ActionTimeMonitor
from green_sarc.state import Budget

from .conftest import make_action

pytestmark = pytest.mark.asyncio


def _slow_executor(tokens: float):
    async def execute(action):
        await asyncio.sleep(0)  # force interleaving between gate and commit
        return ActionOutcome(result="ok", actual_tokens=tokens)

    return execute


async def test_no_overspend_under_concurrency(cost_model, carbon_model):
    # Each action costs 250 actual; the cold-start worst-case forecast is 300.
    # Budget of 1000 tokens admits at most 3 reservations of 300 (=900 reserved).
    budget = Budget(token_budget=1000.0, carbon_ceiling=1_000.0)
    gov = GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        monitor=ActionTimeMonitor(max_loops=10_000),
    )

    async def one():
        try:
            await gov.run_action(make_action(prompt=100, max_tokens=200), _slow_executor(250.0))
            return True
        except GateRejected:
            return False

    results = await asyncio.gather(*[one() for _ in range(50)])

    admitted = sum(results)
    # Spend never exceeds the cap, and reservations are all released at the end.
    assert budget.token_budget >= 0.0
    assert budget.reserved_tokens == 0.0
    assert budget.carbon_spent <= budget.carbon_ceiling
    # Each admitted action actually spent 250 tokens; total spent == admitted*250.
    spent = 1000.0 - budget.token_budget
    assert spent == admitted * 250.0
    # With a 1000 cap and 300-token reservations, at most 3 could be admitted.
    assert 1 <= admitted <= 3


async def test_reservation_released_on_executor_error(cost_model, carbon_model):
    budget = Budget(token_budget=1000.0, carbon_ceiling=1_000.0)
    gov = GreenGovernor(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
    )

    async def boom(action):
        raise RuntimeError("model exploded")

    with pytest.raises(RuntimeError):
        await gov.run_action(make_action(), boom)

    # The reservation must be returned so the budget is not permanently leaked.
    assert budget.reserved_tokens == 0.0
    assert budget.remaining_tokens() == 1000.0
