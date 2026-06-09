"""Binding-budget gate evaluation (paper §9).

Runs the full Green SARC stack on the synthetic IBP workload under a *finite*
token budget, swept across ``B = frac x E[baseline tokens]`` for a grid of
fractions, 20 seeds each.  Per budget point we record gate admission rate,
empirical over-budget incidence, completed-trajectory rate, MAE on admitted
actions, and total tokens.  We also evaluate the §12 soft-penalty (Lagrangian)
baseline at *matched expected spend* so the two can be placed on a common
work-vs-overspend frontier.

This is paper-side analysis: it imports the real library primitives
(:class:`Budget`, :class:`PreActionGate`, :class:`LearnedEstimator`, ...) and the
IBP workload parameters from ``benchmarks.ibp`` (read-only); it does not modify
any library or benchmark code.  Deterministic per seed.

    python paper/scripts/run_binding_budget.py --out paper/data/binding_budget_sweep.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `benchmarks`

from green_sarc import (
    Action,
    ActionTimeMonitor,
    AdapterNode,
    Budget,
    CircuitTripped,
    GovernanceContext,
    LearnedEstimator,
    PreActionGate,
)
from green_sarc.auditor import AuditRecord

from benchmarks.ibp import (  # read-only reuse of the workload definition
    BIG_MODEL,
    REGION,
    SMALL_MODEL,
    IBPConfig,
    _completion,
    _models,
    run_condition,
)

FRACTIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
DELTA = 0.05
SEEDS = 20


def _warm_estimator(seed: int, cfg: IBPConfig, cost_model: Any, carbon: Any) -> LearnedEstimator:
    """Pre-train the estimator on an independent stream so the gate is calibrated.

    The cold-start transient (worst-case forecasts) is not what §9 measures; we
    isolate steady-state binding-budget behaviour by warming the per-key OLS on a
    held-out stream drawn from the same completion model.
    """
    est = LearnedEstimator(cost_model, carbon, min_samples=8)
    rng = random.Random(10_000 + seed)
    for _ in range(2_000):
        model = SMALL_MODEL if rng.random() < cfg.simple_fraction else BIG_MODEL
        prompt = float(cfg.base_prompt + rng.randint(0, cfg.scope_cap))
        completion = _completion(prompt, cfg, rng)
        est.update(
            AuditRecord(
                action_id="w", action_kind="ibp.step", model=model, region=REGION,
                predicted_cost=0.0, predicted_carbon=0.0, confidence=0.0,
                actual_cost=prompt + completion, actual_carbon=0.0,
                budget_remaining_tokens=0.0, carbon_remaining=0.0, carbon_intensity=350.0,
                admitted=True, verdict="admit", prompt_tokens=int(prompt),
            )
        )
    return est


def run_constrained(seed: int, cfg: IBPConfig, budget_tokens: float, delta: float) -> Dict[str, Any]:
    """Full Green SARC stack on the IBP workload under a finite token budget."""
    rng = random.Random(seed)
    cost_model, carbon_fixed, _ = _models()
    adapter = AdapterNode(cfg.scope_cap)
    est = _warm_estimator(seed, cfg, cost_model, carbon_fixed)
    gate = PreActionGate(est)
    budget = Budget(token_budget=budget_tokens, carbon_ceiling=1e18, usd_budget=1e12, delta=delta)

    attempted = admitted = over_budget = completed = 0
    abs_err = 0.0
    tokens = 0.0
    for _sku in range(cfg.n_skus):
        simple = rng.random() < cfg.simple_fraction
        steps_ok = 0
        monitor = ActionTimeMonitor(max_loops=math.ceil(cfg.depth * cfg.max_loops_factor))
        for i in range(cfg.depth):
            accreted = adapter.bound(float(i * cfg.per_step_increment))
            prompt = float(cfg.base_prompt) + accreted
            model = SMALL_MODEL if simple else BIG_MODEL
            completion = _completion(prompt, cfg, rng)
            actual = prompt + completion
            action = Action(kind="ibp.step", model=model, region=REGION,
                            prompt_tokens=int(prompt), max_tokens=cfg.max_tokens)
            decision = gate.evaluate(action, GovernanceContext(budget=budget, timestamp=0.0))
            attempted += 1
            if not decision.admitted:
                break  # budget can no longer fit this trajectory's next step
            try:
                monitor.before()
            except CircuitTripped:
                break
            remaining_before = budget.remaining_tokens()
            if actual > remaining_before:
                over_budget += 1
            budget.spend(actual, 0.0, 0.0)
            est.update(
                AuditRecord(
                    action_id=f"{seed}-{_sku}-{i}", action_kind="ibp.step", model=model,
                    region=REGION, predicted_cost=decision.forecast.cost_hat,
                    predicted_carbon=0.0, confidence=decision.forecast.confidence,
                    actual_cost=actual, actual_carbon=0.0,
                    budget_remaining_tokens=budget.remaining_tokens(), carbon_remaining=0.0,
                    carbon_intensity=350.0, admitted=True, verdict="admit",
                    prompt_tokens=int(prompt),
                )
            )
            admitted += 1
            abs_err += abs(actual - decision.forecast.cost_hat)
            tokens += actual
            steps_ok += 1
        if steps_ok == cfg.depth:
            completed += 1

    return {
        "admission_rate": admitted / attempted if attempted else 0.0,
        "over_budget_incidence": over_budget / admitted if admitted else 0.0,
        "completed_traj_rate": completed / cfg.n_skus,
        "mae_admitted": abs_err / admitted if admitted else 0.0,
        "tokens": tokens,
    }


def _soft_penalty_workload(seed: int, cfg: IBPConfig) -> List[float]:
    """Per-step token costs of the (scoped) workload for one seed, in order."""
    rng = random.Random(seed)
    costs: List[float] = []
    adapter = AdapterNode(cfg.scope_cap)
    for _sku in range(cfg.n_skus):
        for i in range(cfg.depth):
            accreted = adapter.bound(float(i * cfg.per_step_increment))
            prompt = float(cfg.base_prompt) + accreted
            costs.append(prompt + _completion(prompt, cfg, rng))
    return costs


def soft_penalty_at_budget(cfg: IBPConfig, budget_tokens: float) -> Dict[str, Any]:
    """Soft Lagrangian penalty tuned to spend ~B in expectation (budget-blind).

    Admit step iff value - lambda * cost > 0 (value = 1), i.e. cost < 1/lambda.
    Sweep lambda, pick the one whose mean realized spend is closest to B, and
    report its over-budget incidence (P[realized spend > B]) and completed rate.
    """
    series = [_soft_penalty_workload(s, cfg) for s in range(SEEDS)]
    best = None
    for lam in [10 ** e for e in [-3.4 + 0.1 * k for k in range(28)]]:
        thresh = 1.0 / lam
        spends = [sum(c for c in cs if c < thresh) for cs in series]
        gap = abs(statistics.fmean(spends) - budget_tokens)
        if best is None or gap < best[0]:
            best = (gap, lam, thresh, spends)
    _, lam, thresh, spends = best
    over = sum(1 for s in spends if s > budget_tokens) / SEEDS
    # Completed trajectory: all `depth` steps of a SKU pass the cost threshold.
    completed = []
    for cs in series:
        done = sum(
            1
            for k in range(cfg.n_skus)
            if all(cs[k * cfg.depth + i] < thresh for i in range(cfg.depth))
        )
        completed.append(done / cfg.n_skus)
    return {
        "lambda": lam,
        "completed_traj_rate": statistics.fmean(completed),
        "over_budget_incidence": over,
        "mean_spend": statistics.fmean(spends),
    }


def soft_penalty_frontier(cfg: IBPConfig, b_star: float) -> List[Dict[str, Any]]:
    """Trace the soft penalty's (completed, over-budget) frontier by sweeping lambda.

    Over-budget is measured against a single binding reference budget ``b_star``;
    as lambda falls the penalty admits more, completing more trajectories but
    eventually breaching ``b_star`` on every seed -- a curve the gate's frontier
    (over-budget ~ 0 at every completion level) dominates.
    """
    series = [_soft_penalty_workload(s, cfg) for s in range(SEEDS)]
    pts: List[Dict[str, Any]] = []
    for lam in [10 ** (-3.6 + 0.06 * k) for k in range(40)]:
        thresh = 1.0 / lam
        spends = [sum(c for c in cs if c < thresh) for cs in series]
        over = sum(1 for s in spends if s > b_star) / SEEDS
        completed = []
        for cs in series:
            done = sum(
                1
                for k in range(cfg.n_skus)
                if all(cs[k * cfg.depth + i] < thresh for i in range(cfg.depth))
            )
            completed.append(done / cfg.n_skus)
        pts.append(
            {
                "lambda": lam,
                "completed_traj_rate": statistics.fmean(completed),
                "over_budget_incidence": over,
                "mean_spend_frac": statistics.fmean(spends) / b_star,
            }
        )
    return pts


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="run_binding_budget", description=__doc__)
    parser.add_argument("--out", default="paper/data/binding_budget_sweep.json")
    parser.add_argument("--skus", type=int, default=IBPConfig.n_skus)
    parser.add_argument("--depth", type=int, default=IBPConfig.depth)
    args = parser.parse_args(argv)

    cfg = IBPConfig(n_skus=args.skus, depth=args.depth)

    # E[baseline tokens]: full State-Snowball, no governance (read-only reuse).
    base = [run_condition(s, cfg, frozenset())["tokens"] for s in range(SEEDS)]
    e_base = statistics.fmean(base)

    gate_points: List[Dict[str, Any]] = []
    penalty_points: List[Dict[str, Any]] = []
    for frac in FRACTIONS:
        B = frac * e_base
        runs = [run_constrained(s, cfg, B, DELTA) for s in range(SEEDS)]
        agg = {
            "fraction": frac,
            "budget_tokens": B,
            "admission_rate": statistics.fmean(r["admission_rate"] for r in runs),
            "over_budget_incidence": statistics.fmean(r["over_budget_incidence"] for r in runs),
            "completed_traj_rate": statistics.fmean(r["completed_traj_rate"] for r in runs),
            "mae_admitted": statistics.fmean(r["mae_admitted"] for r in runs),
            "tokens": statistics.fmean(r["tokens"] for r in runs),
        }
        gate_points.append(agg)
        pen = soft_penalty_at_budget(cfg, B)
        pen["fraction"] = frac
        pen["budget_tokens"] = B
        penalty_points.append(pen)

    b_star = 0.5 * e_base  # the binding reference budget for the Pareto frontier
    data = {
        "delta": DELTA,
        "seeds": SEEDS,
        "config": {"n_skus": cfg.n_skus, "depth": cfg.depth},
        "e_baseline_tokens": e_base,
        "fractions": FRACTIONS,
        "gate": gate_points,
        "soft_penalty": penalty_points,
        "pareto_reference_budget": b_star,
        "soft_penalty_frontier": soft_penalty_frontier(cfg, b_star),
        "max_over_budget_incidence": max(p["over_budget_incidence"] for p in gate_points),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  E[baseline tokens] = {e_base:,.0f}; delta = {DELTA}")
    for g in gate_points:
        print(
            f"  B={g['fraction']:>4}x  admit={g['admission_rate']:.2f}  "
            f"over-budget={g['over_budget_incidence']*100:5.2f}%  "
            f"completed={g['completed_traj_rate']:.2f}  MAE={g['mae_admitted']:.1f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
