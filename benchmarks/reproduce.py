"""Reproduce the IBP benchmark with an ablation over governance levers (§8).

Usage::

    python -m benchmarks.reproduce                 # 20 seeds, full ablation
    python -m benchmarks.reproduce --seeds 50 --skus 1000 --depth 12
    python -m benchmarks.reproduce --verify benchmarks/reference_summary.json

Runs an **ablation** — ``baseline`` → ``+scope`` → ``+scope+route`` → ``+full`` —
so each lever's contribution is visible, not just "governance helps". Prints a
table with a paired-bootstrap 95% CI on the full-treatment token reduction and
writes the summary JSON to ``--out`` (default ``artifacts/ibp_summary.json``).
Deterministic per seed.

With ``--verify REF`` the fresh summary is compared against a committed reference
(``tokens``/``usd``/``carbon_fixed_g``/``carbon_tv_g`` for each of the four
conditions) within a 2% relative tolerance, plus the ``+full`` token-reduction
within 1.5 percentage points; it exits 2 on drift. Set ``GREEN_SARC_VERIFY_TOL``
(e.g. ``0.10``) to override the relative tolerance.
"""

from __future__ import annotations

import argparse
import json
import os
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
_VERIFY_METRICS = ["tokens", "usd", "carbon_fixed_g", "carbon_tv_g"]
_VERIFY_CONDITIONS = [name for name, _ in CONDITIONS]


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


def _token_reduction(conditions: Dict[str, Any]) -> float:
    base = float(conditions["baseline"]["tokens"])
    full = float(conditions["+full"]["tokens"])
    return 0.0 if base == 0 else 100.0 * (base - full) / base


def _verify(new_summary: Dict[str, Any], ref_path: str, tol: float) -> int:
    """Compare a fresh summary against a committed reference; 0 = OK, 2 = drift."""
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    ref_c = ref["conditions"]
    new_c = new_summary["conditions"]
    failures = []
    for condition in _VERIFY_CONDITIONS:
        for metric in _VERIFY_METRICS:
            r = float(ref_c[condition][metric])
            n = float(new_c[condition][metric])
            drift = abs(n - r) / max(r, 1.0)
            if drift > tol:
                failures.append((condition, metric, r, n, drift))

    ref_red = _token_reduction(ref_c)
    new_red = _token_reduction(new_c)
    reduction_drift = abs(new_red - ref_red)
    reduction_fail = reduction_drift > 1.5

    if failures or reduction_fail:
        print("verify: FAILED")
        print(f"  {'condition':<14}{'metric':<18}{'ref':>14}{'new':>14}{'drift':>9}")
        for condition, metric, r, n, drift in failures:
            print(f"  {condition:<14}{metric:<18}{r:>14.2f}{n:>14.2f}{drift * 100:>8.2f}%")
        if reduction_fail:
            print(
                f"  +full token-reduction: ref {ref_red:.1f}% vs new {new_red:.1f}% "
                f"(drift {reduction_drift:.1f}pp > 1.5pp)"
            )
        return 2

    print(f"verify: OK (4 conditions x 4 metrics within {tol * 100:.0f}% tolerance)")
    return 0


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks.reproduce", description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--skus", type=int, default=IBPConfig.n_skus)
    parser.add_argument("--depth", type=int, default=IBPConfig.depth)
    parser.add_argument("--out", default="artifacts/ibp_summary.json")
    parser.add_argument("--verify", default=None, help="reference summary JSON to check against")
    args = parser.parse_args(argv)

    cfg = IBPConfig(n_skus=args.skus, depth=args.depth)
    per = run(args.seeds, cfg)
    _print_table(args.seeds, cfg, per)

    summary = _summary(args.seeds, cfg, per)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {out}\n")

    if args.verify:
        tol = float(os.environ.get("GREEN_SARC_VERIFY_TOL", "0.02"))
        return _verify(summary, args.verify, tol)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
