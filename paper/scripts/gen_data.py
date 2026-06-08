"""Generate the paper's reproducible data assets from the IBP benchmark.

Writes ``paper/data/ibp_ablation.json``: per-seed metrics for every ablation
condition (``baseline`` -> ``+scope`` -> ``+scope+route`` -> ``+full``), the
paired-bootstrap 95% CIs on each lever's token/USD/carbon reduction, and a
sample of per-action forecast records (prompt, cost_hat, cost_std, actual) from
the full-governance condition for the calibration / conformal / reliability
figures.

Deterministic: every number is computed here from the synthetic benchmark, not
hand-entered.  Run via ``make paper-data`` or::

    python -m paper.scripts.gen_data --seeds 20 --out paper/data/ibp_ablation.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.ibp import CONDITIONS, FEATURES_FULL, IBPConfig, run_condition

_METRIC_KEYS = ["tokens", "usd", "carbon_fixed_g", "carbon_tv_g"]


def _paired_reduction_ci(
    baseline: List[float], treatment: List[float], iters: int = 5000
) -> Dict[str, float]:
    """Paired-bootstrap 95% CI on the mean % reduction (baseline -> treatment)."""
    rng = random.Random(0)
    n = len(baseline)
    reductions: List[float] = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(baseline[i] for i in idx)
        t = sum(treatment[i] for i in idx)
        reductions.append(0.0 if b == 0 else 100.0 * (b - t) / b)
    reductions.sort()
    point = (
        0.0
        if sum(baseline) == 0
        else 100.0 * (sum(baseline) - sum(treatment)) / sum(baseline)
    )
    return {
        "point": point,
        "lo": reductions[int(0.025 * iters)],
        "hi": reductions[int(0.975 * iters)],
    }


def generate(seeds: int, cfg: IBPConfig, sample_seeds: int = 4) -> Dict[str, Any]:
    per: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in CONDITIONS}
    calibration: List[Dict[str, Any]] = []
    for seed in range(seeds):
        for name, features in CONDITIONS:
            collect = calibration if (name == "+full" and seed < sample_seeds) else None
            per[name].append(run_condition(seed, cfg, features, collect=collect))

    # Per-seed arrays for every condition and metric (figures + CIs downstream).
    conditions: Dict[str, Any] = {}
    base = per["baseline"]
    for name, _ in CONDITIONS:
        rows = per[name]
        series = {k: [float(r[k]) for r in rows] for k in _METRIC_KEYS}
        entry: Dict[str, Any] = {"series": series}
        for k in ("admitted", "rejections", "breaker_trips"):
            if k in rows[0]:
                entry[k] = statistics.fmean(float(r[k]) for r in rows)
        for k in ("forecast_mae_tokens", "forecast_wape"):
            vals = [float(r[k]) for r in rows if k in r]
            if vals:
                entry[k] = statistics.fmean(vals)
        # Reduction vs. baseline with a paired-bootstrap CI (skip baseline itself).
        if name != "baseline":
            entry["reduction_ci"] = {
                k: _paired_reduction_ci(
                    [float(r[k]) for r in base], [float(r[k]) for r in rows]
                )
                for k in _METRIC_KEYS
            }
        conditions[name] = entry

    return {
        "seeds": seeds,
        "config": vars(cfg),
        "features_full": sorted(FEATURES_FULL),
        "conditions": conditions,
        "calibration_samples": calibration,
    }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="paper.scripts.gen_data", description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--skus", type=int, default=IBPConfig.n_skus)
    parser.add_argument("--depth", type=int, default=IBPConfig.depth)
    parser.add_argument("--out", default="paper/data/ibp_ablation.json")
    args = parser.parse_args(argv)

    cfg = IBPConfig(n_skus=args.skus, depth=args.depth)
    data = generate(args.seeds, cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    full = data["conditions"]["+full"]
    ci = full["reduction_ci"]["tokens"]
    print(f"wrote {out}")
    print(
        f"  +full token reduction: {ci['point']:.1f}%  "
        f"95% CI [{ci['lo']:.1f}%, {ci['hi']:.1f}%]"
    )
    print(f"  calibration samples collected: {len(data['calibration_samples'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
