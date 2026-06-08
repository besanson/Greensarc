"""Reproduce the IBP benchmark with an ablation over governance levers (§8).

Usage::

    python -m benchmarks.reproduce                 # 20 seeds, full ablation
    python -m benchmarks.reproduce --seeds 50 --skus 1000 --depth 12

Runs an **ablation** — ``baseline`` → ``+scope`` → ``+scope+route`` → ``+full`` —
so each lever's contribution is visible, not just "governance helps". Prints a
table with a paired-bootstrap 95% CI on the full-treatment token reduction and
writes ``artifacts/ibp_summary.json``. Deterministic per seed.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.ibp import CONDITIONS, IBPConfig, run_condition

_METRICS = [
    ("tokens", "total tokens"),
    ("usd", "total USD"),
    ("carbon_fixed_g", "carbon fixed (g)"),
    ("carbon_tv_g", "carbon time-var (g)"),
]


def _fmt(v: float) -> str:
    return f"{v / 1e6:.2f}M" if abs(v) >= 1e6 else f"{v:,.0f}"


def _mean(rows: List[Dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(r[key]) for r in rows)


def _bootstrap_ci(baseline: List[float], treatment: List[float], iters: int = 2000) -> tuple:
    """Paired-bootstrap 95% CI on the mean fractional reduction (baseline→treatment)."""
    rng = random.Random(0)
    n = len(baseline)
    reductions = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        b = sum(baseline[i] for i in idx)
        t = sum(treatment[i] for i in idx)
        reductions.append(0.0 if b == 0 else 100.0 * (b - t) / b)
    reductions.sort()
    lo = reductions[int(0.025 * iters)]
    hi = reductions[int(0.975 * iters)]
    return lo, hi


def run(seeds: int, cfg: IBPConfig) -> Dict[str, List[Dict[str, Any]]]:
    per: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in CONDITIONS}
    for seed in range(seeds):
        for name, features in CONDITIONS:
            per[name].append(run_condition(seed, cfg, features))
    return per


def _print_table(seeds: int, cfg: IBPConfig, per: Dict[str, List[Dict[str, Any]]]) -> None:
    names = [name for name, _ in CONDITIONS]
    base = per["baseline"]
    print(f"\nIBP ablation — {seeds} seeds, {cfg.n_skus} SKUs, depth {cfg.depth}\n")
    header = f"  {'metric':<20}" + "".join(f"{n:>15}" for n in names)
    print(header)
    print(f"  {'-' * (20 + 15 * len(names))}")
    for key, label in _METRICS:
        row = f"  {label:<20}" + "".join(f"{_fmt(_mean(per[n], key)):>15}" for n in names)
        print(row)
    # Reduction vs baseline (the ablation story: each lever adds saving).
    print("\n  token reduction vs baseline:")
    base_tok = _mean(base, "tokens")
    for name in names[1:]:
        red = 100.0 * (base_tok - _mean(per[name], "tokens")) / base_tok
        print(f"    {name:<16}{red:6.1f}%")
    lo, hi = _bootstrap_ci(
        [float(r["tokens"]) for r in base], [float(r["tokens"]) for r in per["+full"]]
    )
    print(f"  +full 95% CI on token reduction: [{lo:.1f}%, {hi:.1f}%] (paired bootstrap)")
    full = per["+full"]
    print(
        f"  +full governance: {_mean(full, 'admitted'):.0f} admitted, "
        f"{_mean(full, 'rejections'):.0f} gate rejections, "
        f"{_mean(full, 'breaker_trips'):.0f} breaker trips"
    )
    if "forecast_mae_tokens" in full[0]:
        print(
            f"  +full forecast:   MAE {_mean(full, 'forecast_mae_tokens'):.1f} tokens, "
            f"WAPE {_mean(full, 'forecast_wape') * 100:.1f}%\n"
        )


def _summary(seeds: int, cfg: IBPConfig, per: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    def agg(rows: List[Dict[str, Any]]) -> Dict[str, float]:
        keys = set().union(*(r.keys() for r in rows))
        return {k: _mean([r for r in rows if k in r], k) for k in keys}

    return {
        "seeds": seeds,
        "config": vars(cfg),
        "conditions": {name: agg(rows) for name, rows in per.items()},
    }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.reproduce", description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--skus", type=int, default=IBPConfig.n_skus)
    parser.add_argument("--depth", type=int, default=IBPConfig.depth)
    parser.add_argument("--out", default="artifacts/ibp_summary.json")
    args = parser.parse_args(argv)

    cfg = IBPConfig(n_skus=args.skus, depth=args.depth)
    per = run(args.seeds, cfg)
    _print_table(args.seeds, cfg, per)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_summary(args.seeds, cfg, per), indent=2), encoding="utf-8")
    print(f"  wrote {out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
