"""Learning-curve demo: the predictive estimator gets better as it learns.

Run it::

    python examples/learning_curve/run_demo.py

This exercises the closed predict -> act -> log -> retrain loop on a realistic
agent whose completion length depends on its prompt length (plus noise). It
shows two things the audit asked for:

- **P0-4**: the regression estimator's forecast error (MAE) drops sharply once it
  has learned the prompt -> completion relationship — the cold-start worst case
  gives way to a calibrated forecast.
- **P1-2**: a **USD budget** is enforced alongside tokens and carbon.
"""

from __future__ import annotations

import asyncio
import random

from green_sarc import (
    Action,
    ActionOutcome,
    ActionTimeMonitor,
    Budget,
    GreenGovernor,
    LearnedEstimator,
    MemoryAuditStore,
    ModelProfile,
    TableCarbonModel,
    TableCostModel,
)


async def main() -> None:
    rng = random.Random(7)
    # Per-token energy and USD pricing so cost, carbon, and USD are all governed.
    cost_model = TableCostModel(
        profiles={
            "gpt-x": ModelProfile(
                energy_per_token_kwh=3.0e-7,
                usd_per_prompt_token=5.0e-7,
                usd_per_completion_token=1.5e-6,
            )
        }
    )
    carbon_model = TableCarbonModel(intensities={"eu-west": 230.0})
    budget = Budget(
        token_budget=10_000_000.0,
        carbon_ceiling=1.0e9,
        usd_budget=100.0,
        delta=0.1,
    )
    store = MemoryAuditStore()
    gov = GreenGovernor(
        budget=budget,
        estimator=LearnedEstimator(cost_model, carbon_model, min_samples=5),
        cost_model=cost_model,
        carbon_model=carbon_model,
        store=store,
        monitor=ActionTimeMonitor(max_loops=10_000),
    )

    async def run_one(prompt: int) -> None:
        # Ground truth the estimator must learn: completion ~ 40 + 0.6*prompt + noise.
        completion = max(1, int(40 + 0.6 * prompt + rng.gauss(0, 15)))

        async def execute(action: Action) -> ActionOutcome:
            return ActionOutcome(result="ok", actual_tokens=float(prompt + completion))

        action = Action(
            kind="chat.completion",
            model="gpt-x",
            region="eu-west",
            prompt_tokens=prompt,
            max_tokens=4000,
        )
        await gov.run_action(action, execute)

    n = 60
    for _ in range(n):
        await run_one(rng.randint(50, 600))

    records = [r for r in store.list() if r.actual_cost > 0]
    half = len(records) // 2

    def mae(rows) -> float:
        return sum(abs(r.cost_error) for r in rows) / len(rows)

    print(f"ran {len(records)} actions (predict -> act -> log -> retrain)\n")
    print(f"  forecast source, first action : {records[0].forecast_source}")
    print(f"  forecast source, last action  : {records[-1].forecast_source}")
    print(f"  MAE token cost, first half    : {mae(records[:half]):8.1f}")
    print(f"  MAE token cost, second half   : {mae(records[half:]):8.1f}   <- learned")
    print(f"\n  USD spent: ${budget.usd_spent:.4f} of ${budget.usd_budget:.2f} budget")
    print(f"  carbon:    {budget.carbon_spent:.2f} gCO2e")


if __name__ == "__main__":
    asyncio.run(main())
