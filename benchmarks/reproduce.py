"""Reproduce the IBP benchmark over multiple seeds (working paper §8).

Usage::

    python -m benchmarks.reproduce            # defaults (20 seeds)
    python -m benchmarks.reproduce --seeds 50 --skus 1000 --depth 12

Prints a baseline-vs-treatment table and writes ``artifacts/ibp_summary.json``.
The numbers are deterministic per seed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.ibp import IBPConfig, run_pair


def _mean_std(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
    }


def _aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    keys = set().union(*(r.keys() for r in rows))
    return {k: _mean_std([float(r[k]) for r in rows if k in r]) for k in keys}


def _reduction(baseline: float, treatment: float) -> float:
    return 0.0 if baseline == 0 else 100.0 * (baseline - treatment) / baseline


def run(seeds: int, cfg: IBPConfig) -> Dict[str, Any]:
    base_rows: List[Dict[str, Any]] = []
    treat_rows: List[Dict[str, Any]] = []
    for seed in range(seeds):
        base, treat = run_pair(seed, cfg)
        base_rows.append(base)
        treat_rows.append(treat)
    return {
        "config": vars(cfg),
        "seeds": seeds,
        "baseline": _aggregate(base_rows),
        "treatment": _aggregate(treat_rows),
    }


def _print_table(summary: Dict[str, Any]) -> None:
    b = summary["baseline"]
    t = summary["treatment"]
    print(
        f"\nIBP benchmark — {summary['seeds']} seeds, "
        f"{summary['config']['n_skus']} SKUs, depth {summary['config']['depth']}\n"
    )
    print(f"  {'metric':<22}{'baseline':>16}{'treatment':>16}{'reduction':>12}")
    print(f"  {'-' * 64}")
    for key, label in [
        ("tokens", "total tokens"),
        ("usd", "total USD"),
        ("carbon_fixed_g", "carbon fixed (g)"),
        ("carbon_tv_g", "carbon time-var (g)"),
    ]:
        bm, tm = b[key]["mean"], t[key]["mean"]
        print(f"  {label:<22}{bm:>16,.0f}{tm:>16,.0f}{_reduction(bm, tm):>11.1f}%")
    print(f"  {'-' * 64}")
    print(
        f"  treatment governance: "
        f"{t['admitted']['mean']:.0f} admitted, "
        f"{t.get('rejections', {}).get('mean', 0):.0f} gate rejections, "
        f"{t.get('breaker_trips', {}).get('mean', 0):.0f} breaker trips"
    )
    if "forecast_mae_tokens" in t:
        print(
            f"  treatment forecast:   MAE {t['forecast_mae_tokens']['mean']:.1f} tokens, "
            f"WAPE {t['forecast_wape']['mean'] * 100:.1f}%"
        )
    print()


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.reproduce", description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--skus", type=int, default=IBPConfig.n_skus)
    parser.add_argument("--depth", type=int, default=IBPConfig.depth)
    parser.add_argument("--out", default="artifacts/ibp_summary.json")
    args = parser.parse_args(argv)

    cfg = IBPConfig(n_skus=args.skus, depth=args.depth)
    summary = run(args.seeds, cfg)
    _print_table(summary)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
