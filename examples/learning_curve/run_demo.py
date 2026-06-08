"""Learning-curve demo: the predictive estimator gets better as it learns.

Run it::

    python examples/learning_curve/run_demo.py
    python examples/learning_curve/run_demo.py --emit-json paper/data/learning_curve.json

This exercises the closed predict -> act -> log -> retrain loop on a realistic
agent whose completion length depends on its prompt length (plus noise). It
shows two things the audit asked for:

- **P0-4**: the regression estimator's forecast error (MAE) drops sharply once it
  has learned the prompt -> completion relationship — the cold-start worst case
  gives way to a calibrated forecast.
- **P1-2**: a **USD budget** is enforced alongside tokens and carbon.

With ``--emit-json PATH`` the per-action history (forecast source, predicted vs.
actual token cost, gate admission) is written as JSON for the cold-start
learning-curve figure in the paper (``paper/scripts/build_figures.py``); the run
is deterministic given ``--seed``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path


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


async def main(emit_json: str | None = None, n: int = 60, seed: int = 7) -> None:
    rng = random.Random(seed)
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

    if emit_json is not None:
        # Per-action history for the cold-start learning-curve figure (§7/§8).
        # The store holds only admitted actions; we recover the gate rejection
        # rate from the number of actions attempted vs. recorded.
        history = [
            {
                "i": i,
                "source": r.forecast_source,
                "prompt_tokens": int(r.prompt_tokens),
                "predicted_cost": r.predicted_cost,
                "actual_cost": r.actual_cost,
                "abs_error": abs(r.cost_error),
            }
            for i, r in enumerate(records, start=1)
        ]
        payload = {
            "seed": seed,
            "n_attempted": n,
            "n_recorded": len(records),
            "rejection_rate": (n - len(records)) / n if n else 0.0,
            "ground_truth": {"intercept": 40.0, "slope": 0.6, "noise_std": 15.0},
            "mae_first_half": mae(records[:half]),
            "mae_second_half": mae(records[half:]),
            "wape_first_half": sum(abs(r.cost_error) for r in records[:half])
            / sum(r.actual_cost for r in records[:half]),
            "wape_second_half": sum(abs(r.cost_error) for r in records[half:])
            / sum(r.actual_cost for r in records[half:]),
            "history": history,
        }
        out = Path(emit_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\n  wrote {out}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Predictive-estimator learning-curve demo.")
    parser.add_argument(
        "--emit-json",
        default=None,
        help="write the per-action learning curve to this JSON path (for the paper figure).",
    )
    parser.add_argument("--n", type=int, default=60, help="number of actions to run.")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed (deterministic run).")
    args = parser.parse_args()
    asyncio.run(main(emit_json=args.emit_json, n=args.n, seed=args.seed))


if __name__ == "__main__":
    _cli()
