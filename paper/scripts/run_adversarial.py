"""Toy adversarial robustness study against the Predictive Pre-Action Gate (§13).

Three attack classes (continuation inflation, scope-cap-aware padding,
model-substitution gaming) are constructed synthetically against the same gate
code path (PreActionGate + LearnedEstimator + Budget) that the IBP benchmark
exercises. Token counts only; no LLM is called. Reports, with paired-bootstrap
95% CIs over seeds, the gate's admission rate, realized over-budget rate,
realized/declared ratio, and a per-class failure mode.

    python paper/scripts/run_adversarial.py --out paper/data/adversarial.json \
        --seeds 20 --n-per-class 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from green_sarc import (  # noqa: E402
    Action,
    Budget,
    GovernanceContext,
    LearnedEstimator,
    PreActionGate,
)
from green_sarc.auditor import AuditRecord  # noqa: E402

from benchmarks.ibp import BIG_MODEL, REGION, SMALL_MODEL, _models  # noqa: E402

SCOPE_CAP = 360  # the IBP Adapter-Node cap the attacker knows
DELTA = 0.05


def _trained_gate(seed: int):
    """A gate whose estimator has learned the benign prompt->completion law."""
    cost_model, carbon_fixed, _ = _models()
    est = LearnedEstimator(cost_model, carbon_fixed, min_samples=8)
    rng = np.random.default_rng(seed)
    for _ in range(600):
        for model in (BIG_MODEL, SMALL_MODEL):
            prompt = float(rng.integers(80, SCOPE_CAP))
            completion = max(1.0, 40.0 + 0.6 * prompt + rng.normal(0, 20))  # benign law
            est.update(AuditRecord(
                action_id="w", action_kind="serve", model=model, region=REGION,
                predicted_cost=0.0, predicted_carbon=0.0, confidence=0.0,
                actual_cost=prompt + completion, actual_carbon=0.0,
                budget_remaining_tokens=0.0, carbon_remaining=0.0, carbon_intensity=350.0,
                admitted=True, verdict="admit", prompt_tokens=int(prompt)))
    return PreActionGate(est), cost_model


def _benign_completion(prompt: float, rng) -> float:
    return max(1.0, 40.0 + 0.6 * prompt + rng.normal(0, 20))


# Each attack returns (prompt_tokens, realized_completion, claimed_model, realized_model).
def attack_continuation(rng) -> Tuple[float, float, str, str]:
    prompt = float(rng.integers(80, SCOPE_CAP))
    realized = 8.0 * _benign_completion(prompt, rng)  # "continue indefinitely"
    return prompt, realized, BIG_MODEL, BIG_MODEL


def attack_padding(rng) -> Tuple[float, float, str, str]:
    prompt = float(SCOPE_CAP - 1)  # maximal admitted prompt, benign completion
    realized = _benign_completion(prompt, rng)
    return prompt, realized, BIG_MODEL, BIG_MODEL


def attack_substitution(rng) -> Tuple[float, float, str, str]:
    prompt = float(rng.integers(80, SCOPE_CAP))
    realized = _benign_completion(prompt, rng)
    return prompt, realized, SMALL_MODEL, BIG_MODEL  # claims efficient, runs on frontier


ATTACKS: Dict[str, Callable] = {
    "continuation_inflation": attack_continuation,
    "scope_cap_aware_padding": attack_padding,
    "model_substitution_gaming": attack_substitution,
}


def _run_class(name: str, attack: Callable, seeds: int, n: int) -> Dict[str, Any]:
    per_seed_admit, per_seed_over, per_seed_ratio = [], [], []
    for seed in range(seeds):
        gate, cost_model = _trained_gate(seed)
        rng = np.random.default_rng(10_000 + seed)
        admits = overs = 0
        ratios = []
        for _ in range(n):
            prompt, realized, claimed, real_model = attack(rng)
            action = Action(kind="serve", model=claimed, region=REGION,
                            prompt_tokens=int(prompt), max_tokens=8192)
            # Generous budget: we test the forecast's robustness, not budget binding.
            budget = Budget(token_budget=1e12, carbon_ceiling=1e18, usd_budget=1e12, delta=DELTA)
            dec = gate.evaluate(action, GovernanceContext(budget=budget, timestamp=0.0))
            bound = gate.cost_upper_bound(dec.forecast, DELTA)  # token safety bound
            if dec.admitted:
                admits += 1
            if name == "model_substitution_gaming":
                # Cost-side attack: realized USD at frontier vs forecast USD at efficient.
                fc_usd = cost_model.usd(claimed, prompt, realized)
                real_usd = cost_model.usd(real_model, prompt, realized)
                ratios.append(real_usd / fc_usd if fc_usd > 0 else 1.0)
                if real_usd > fc_usd * 1.01:
                    overs += 1
            else:
                realized_total = prompt + realized
                ratios.append(realized_total / bound if bound > 0 else 1.0)
                if realized_total > bound:
                    overs += 1
        per_seed_admit.append(admits / n)
        per_seed_over.append(overs / n)
        per_seed_ratio.append(statistics.fmean(ratios))
    return {
        "admission_rate": _ci(per_seed_admit),
        "over_budget_rate": _ci(per_seed_over),
        "realized_declared_ratio": _ci(per_seed_ratio),
        "failure_mode": _classify(name, per_seed_admit, per_seed_over),
    }


def _ci(xs: List[float], boot: int = 2000) -> Dict[str, float]:
    arr = np.array(xs)
    rng = np.random.default_rng(0)
    means = np.array([arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(boot)])
    means.sort()
    return {"point": float(arr.mean()), "lo": float(means[int(0.025 * boot)]),
            "hi": float(means[int(0.975 * boot)])}


def _classify(name: str, admit: List[float], over: List[float]) -> str:
    a, o = statistics.fmean(admit), statistics.fmean(over)
    if o >= 0.5 and a >= 0.5:
        return "under-estimates"   # admitted, then blew the bound
    if a >= 0.9 and o < 0.1:
        return "over-admits"       # stays within the contract, extracts max work
    return "correctly-rejects"


def main(argv: Any = None) -> int:
    p = argparse.ArgumentParser(prog="run_adversarial", description=__doc__)
    p.add_argument("--out", default="paper/data/adversarial.json")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--n-per-class", type=int, default=200)
    args = p.parse_args(argv)

    classes = {name: _run_class(name, atk, args.seeds, args.n_per_class)
               for name, atk in ATTACKS.items()}
    data = {"seeds": args.seeds, "n_per_class": args.n_per_class,
            "scope_cap": SCOPE_CAP, "delta": DELTA, "classes": classes}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for name, c in classes.items():
        print(f"  {name:<26} admit={c['admission_rate']['point']:.2f} "
              f"over-budget={c['over_budget_rate']['point']:.2f} "
              f"ratio={c['realized_declared_ratio']['point']:.2f} -> {c['failure_mode']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
