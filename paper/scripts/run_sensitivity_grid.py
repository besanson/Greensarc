"""Joint sensitivity grid over delta x scope_cap x route_fraction (§12.5).

Sweeps 4 deltas x 4 scope caps x 5 routing fractions = 80 cells x 20 seeds on the
IBP benchmark. Token/USD/carbon reductions come from the (non-binding) IBP harness
``run_condition`` exactly as gen_data uses it; over-budget incidence comes from a
binding-budget run of the same workload (delta only bites when the budget binds,
as §8/§12 show). Reports the Pareto frontier of (token saving, over-budget
incidence) and tags the headline operating point.

    python paper/scripts/run_sensitivity_grid.py --out paper/data/sensitivity_grid.json --seeds 20
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from green_sarc import (  # noqa: E402
    Action,
    ActionTimeMonitor,
    AdapterNode,
    Budget,
    CircuitTripped,
    GovernanceContext,
    LearnedEstimator,
    PreActionGate,
)
from green_sarc.auditor import AuditRecord  # noqa: E402

from benchmarks.ibp import (  # noqa: E402
    BIG_MODEL,
    REGION,
    SMALL_MODEL,
    IBPConfig,
    _completion,
    _models,
    run_condition,
)

DELTAS = [0.01, 0.05, 0.10, 0.20]
ROUTE_FRACTIONS = [0.0, 0.25, 0.50, 0.75, 1.0]
CAP_MULTIPLES = [0.5, 1.0, 2.0, 4.0]
# The §8 ablation's actual operating point: scope_cap=360 (~0.5x the median prompt
# of 740), routing 0.5, delta 0.1 (the IBP defaults).
HEADLINE = {"delta": 0.10, "cap_mult": 0.5, "route": 0.50}
# Over-budget only varies with delta when the forecast is uncertain; the IBP's
# native completion noise (25) is too small, so the over-budget axis is measured
# under the elevated noise (sigma~90) used by the §12 delta-sensitivity stress.
STRESS_NOISE = 90.0


def _paired_reduction(base: np.ndarray, treat: np.ndarray, rng, boot: int = 1000) -> float:
    if base.sum() == 0:
        return 0.0
    return 100.0 * (base.sum() - treat.sum()) / base.sum()


def _binding_over_budget(seed: int, cfg: IBPConfig, budget_tokens: float) -> float:
    """Over-budget incidence (admitted steps whose realized cost exceeded the
    remaining budget) for the full stack under a finite token budget."""
    import dataclasses
    import random

    cfg = dataclasses.replace(cfg, completion_noise=STRESS_NOISE)  # expose the delta tradeoff
    rng = random.Random(seed)
    cost_model, carbon_fixed, _ = _models()
    adapter = AdapterNode(cfg.scope_cap)
    est = LearnedEstimator(cost_model, carbon_fixed, min_samples=8)
    # warm the estimator so the gate is calibrated (steady-state, not cold start)
    for _ in range(800):
        pr = float(cfg.base_prompt + rng.randint(0, cfg.scope_cap))
        est.update(AuditRecord(
            action_id="w", action_kind="ibp.step", model=BIG_MODEL, region=REGION,
            predicted_cost=0.0, predicted_carbon=0.0, confidence=0.0,
            actual_cost=pr + _completion(pr, cfg, rng), actual_carbon=0.0,
            budget_remaining_tokens=0.0, carbon_remaining=0.0, carbon_intensity=350.0,
            admitted=True, verdict="admit", prompt_tokens=int(pr)))
    gate = PreActionGate(est)
    budget = Budget(token_budget=budget_tokens, carbon_ceiling=1e18, usd_budget=1e12,
                    delta=cfg.delta)
    admitted = over = 0
    for _sku in range(cfg.n_skus):
        simple = rng.random() < cfg.simple_fraction
        monitor = ActionTimeMonitor(max_loops=math.ceil(cfg.depth * cfg.max_loops_factor))
        for i in range(cfg.depth):
            accreted = adapter.bound(float(i * cfg.per_step_increment))
            prompt = float(cfg.base_prompt) + accreted
            model = SMALL_MODEL if simple else BIG_MODEL
            actual = prompt + _completion(prompt, cfg, rng)
            action = Action(kind="ibp.step", model=model, region=REGION,
                            prompt_tokens=int(prompt), max_tokens=cfg.max_tokens)
            dec = gate.evaluate(action, GovernanceContext(budget=budget, timestamp=0.0))
            if not dec.admitted:
                break
            try:
                monitor.before()
            except CircuitTripped:
                break
            rem = budget.remaining_tokens()
            if actual > rem:
                over += 1
            budget.spend(actual, 0.0, 0.0)
            est.update(AuditRecord(
                action_id="x", action_kind="ibp.step", model=model, region=REGION,
                predicted_cost=dec.forecast.cost_hat, predicted_carbon=0.0,
                confidence=dec.forecast.confidence, actual_cost=actual, actual_carbon=0.0,
                budget_remaining_tokens=budget.remaining_tokens(), carbon_remaining=0.0,
                carbon_intensity=350.0, admitted=True, verdict="admit",
                prompt_tokens=int(prompt)))
            admitted += 1
    return over / admitted if admitted else 0.0


def main(argv: Any = None) -> int:
    p = argparse.ArgumentParser(prog="run_sensitivity_grid", description=__doc__)
    p.add_argument("--out", default="paper/data/sensitivity_grid.json")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--skus", type=int, default=IBPConfig.n_skus)
    args = p.parse_args(argv)

    base_cfg = IBPConfig(n_skus=args.skus)
    median_prompt = base_cfg.base_prompt + (base_cfg.depth - 1) / 2 * base_cfg.per_step_increment
    caps = {m: int(m * median_prompt) for m in CAP_MULTIPLES}

    # Baseline (no governance) is independent of cap/route/delta: compute once.
    METRICS = ("tokens", "usd", "carbon_fixed_g")
    base = {k: np.array([run_condition(s, base_cfg, frozenset())[k] for s in range(args.seeds)])
            for k in METRICS}
    e_base = float(base["tokens"].mean())
    budget_tokens = 0.75 * e_base  # a binding budget so delta matters

    rng = np.random.default_rng(0)
    # Token/USD/carbon reduction depends on (cap, route) only (non-binding gate).
    full_feats = frozenset({"scope", "routing", "gate", "monitor"})
    red_cache: Dict[tuple, Dict[str, float]] = {}
    for m in CAP_MULTIPLES:
        for route in ROUTE_FRACTIONS:
            cfg = IBPConfig(n_skus=args.skus, scope_cap=caps[m], simple_fraction=route)
            full = {k: np.array([run_condition(s, cfg, full_feats)[k] for s in range(args.seeds)])
                    for k in METRICS}
            red_cache[(m, route)] = {
                "tokens": _paired_reduction(base["tokens"], full["tokens"], rng),
                "usd": _paired_reduction(base["usd"], full["usd"], rng),
                "carbon": _paired_reduction(base["carbon_fixed_g"], full["carbon_fixed_g"], rng),
            }

    cells: List[Dict[str, Any]] = []
    for d in DELTAS:
        for m in CAP_MULTIPLES:
            for route in ROUTE_FRACTIONS:
                cfg = IBPConfig(n_skus=args.skus, scope_cap=caps[m], simple_fraction=route, delta=d)
                ob = statistics.fmean(_binding_over_budget(s, cfg, budget_tokens)
                                      for s in range(args.seeds))
                red = red_cache[(m, route)]
                cells.append({
                    "delta": d, "cap_mult": m, "scope_cap": caps[m], "route_fraction": route,
                    "token_reduction": red["tokens"], "usd_reduction": red["usd"],
                    "carbon_reduction": red["carbon"], "over_budget_incidence": ob,
                    "is_headline": (d == HEADLINE["delta"] and m == HEADLINE["cap_mult"]
                                    and route == HEADLINE["route"]),
                })

    # Pareto frontier: maximise token reduction, minimise over-budget incidence.
    def dominated(c, others):
        return any(o["token_reduction"] >= c["token_reduction"]
                   and o["over_budget_incidence"] <= c["over_budget_incidence"]
                   and (o["token_reduction"] > c["token_reduction"]
                        or o["over_budget_incidence"] < c["over_budget_incidence"])
                   for o in others)
    for c in cells:
        c["on_frontier"] = not dominated(c, cells)

    headline = next(c for c in cells if c["is_headline"])
    frontier = [c for c in cells if c["on_frontier"]]
    data = {
        "seeds": args.seeds, "deltas": DELTAS, "cap_multiples": CAP_MULTIPLES,
        "route_fractions": ROUTE_FRACTIONS, "median_prompt": median_prompt,
        "caps": caps, "e_baseline_tokens": e_base, "binding_budget_tokens": budget_tokens,
        "n_cells": len(cells), "cells": cells,
        "headline": headline, "headline_on_frontier": headline["on_frontier"],
        "n_frontier": len(frontier),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out}  ({len(cells)} cells, {len(frontier)} on frontier)")
    print(f"  headline (δ={headline['delta']}, cap={headline['cap_mult']}×, "
          f"route={headline['route_fraction']}): token -{headline['token_reduction']:.1f}%, "
          f"over-budget {headline['over_budget_incidence']*100:.2f}%, "
          f"on frontier: {headline['on_frontier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
