"""Microbenchmark: what does the Pre-Action Gate cost per decision?

Measures :meth:`green_sarc.gate.PreActionGate.evaluate` latency over a large
number of warm decisions, for both admission-bound paths:

- **normal_sigma** — the default Normal-``sigma`` quantile bound; and
- **split_conformal** — the distribution-free bound (a fitted
  :class:`~green_sarc.calibrator.SplitConformal`).

Reports p50 / p99 in microseconds and decisions/second, single process, pinned
seed. A fixed forecast isolates the gate's own arithmetic (the estimator is a
constant stub) so the number is the governance overhead, not estimator-fit time.

    python benchmarks/gate_overhead.py --n 200000 --out paper/data/gate_overhead.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter_ns
from typing import Any, Dict

import numpy as np

from green_sarc.calibrator import SplitConformal
from green_sarc.forecast import Forecast
from green_sarc.gate import PreActionGate
from green_sarc.state import Action, Budget, GovernanceContext


class _FixedEstimator:
    """Returns a constant forecast with a positive std (exercises the Normal path)."""

    def __init__(self, cost_hat: float = 1000.0, cost_std: float = 150.0) -> None:
        self._f = Forecast(
            cost_hat=cost_hat,
            carbon_hat=cost_hat * 3.0e-7 * 500.0,
            confidence=0.95,
            cost_std=cost_std,
            usd_hat=cost_hat * 2.0e-6,
            source="benchmark",
        )

    def predict(self, action: Action, ctx: GovernanceContext) -> Forecast:
        return self._f

    def update(self, record: Any) -> None:  # pragma: no cover - unused here
        pass


def _percentiles(latencies_ns: np.ndarray) -> Dict[str, float]:
    p50 = float(np.percentile(latencies_ns, 50)) / 1000.0  # ns -> us
    p99 = float(np.percentile(latencies_ns, 99)) / 1000.0
    mean_ns = float(latencies_ns.mean())
    return {
        "p50_us": p50,
        "p99_us": p99,
        "mean_us": mean_ns / 1000.0,
        "decisions_per_sec": 1.0e9 / mean_ns if mean_ns else 0.0,
    }


def _time_path(gate: PreActionGate, n: int, warmup: int) -> Dict[str, float]:
    # A roomy budget so every decision takes the full admit path.
    ctx = GovernanceContext(budget=Budget(token_budget=1.0e18, carbon_ceiling=1.0e18, delta=0.05))
    action = Action(
        kind="chat.completion", model="bench", region="eu", prompt_tokens=500, max_tokens=1000
    )
    for _ in range(warmup):
        gate.evaluate(action, ctx)
    lat = np.empty(n, dtype=np.float64)
    for i in range(n):
        t0 = perf_counter_ns()
        gate.evaluate(action, ctx)
        lat[i] = perf_counter_ns() - t0
    return _percentiles(lat)


def run(n: int = 200_000, warmup: int = 5_000, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    est = _FixedEstimator()

    normal_gate = PreActionGate(est)  # no calibrator -> Normal-sigma path

    conformal = SplitConformal()
    conformal.fit(rng.normal(0.0, 150.0, size=2000))  # representative calibration residuals
    conformal_gate = PreActionGate(est, calibrator=conformal)

    return {
        "n_decisions": n,
        "warmup": warmup,
        "seed": seed,
        "normal_sigma": _time_path(normal_gate, n, warmup),
        "split_conformal": _time_path(conformal_gate, n, warmup),
    }


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(prog="gate_overhead", description=__doc__)
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--warmup", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="paper/data/gate_overhead.json")
    args = ap.parse_args(argv)

    result = run(args.n, args.warmup, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    for path in ("normal_sigma", "split_conformal"):
        r = result[path]
        print(
            f"  {path:<16} p50 {r['p50_us']:.2f} us  p99 {r['p99_us']:.2f} us  "
            f"{r['decisions_per_sec'] / 1e6:.2f} M decisions/s"
        )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
