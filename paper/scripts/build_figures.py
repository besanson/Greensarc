"""Build the paper's figures (7 deterministic PDFs) and a stats summary.

Reads the committed data assets (``paper/data/ibp_ablation.json`` from
``paper.scripts.gen_data`` and ``paper/data/learning_curve.json`` from the
learning-curve demo) and computes the remaining experiments deterministically
from the benchmark model, then writes PDFs to ``paper/figures/`` and the headline
numbers to ``paper/data/figure_stats.json`` so every quoted value is reproducible.

Figures
-------
1. ``snowball_fit``      empirical quadratic cost vs. depth (Theorem 1) + scoped.
2. ``ablation_bars``     per-lever token/USD/carbon reduction with 95% CIs.
3. ``calibration``       predicted vs. actual token cost (learned forecasts).
4. ``reliability``       nominal vs. empirical coverage: Normal-σ gate vs. conformal.
5. ``delta_sensitivity`` δ knob: admission rate vs. overspend (overshoot) rate.
6. ``coldstart``         forecast MAE vs. actions seen (cold start -> learned).
7. ``penalty_vs_gate``   soft Lagrangian penalty cannot guarantee the budget; the gate can.

Run via ``make paper-figures`` or ``python -m paper.scripts.build_figures``.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.ibp import IBPConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "paper" / "data"
FIGS = ROOT / "paper" / "figures"

plt.rcParams.update(
    {
        "figure.figsize": (5.0, 3.4),
        "figure.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
    }
)
BLUE, ORANGE, GREEN, GREY = "#2E75B6", "#E07A2B", "#4C9A52", "#888888"


def _save(fig: plt.Figure, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    plt.close(fig)


# --------------------------------------------------------------------------
# 1. Snowball: empirical cost vs. depth (Theorem 1), baseline vs. scoped.
# --------------------------------------------------------------------------
def fig_snowball(cfg: IBPConfig, stats: Dict[str, Any]) -> None:
    depths = list(range(1, 41))
    s0, p, cap = float(cfg.base_prompt), float(cfg.per_step_increment), float(cfg.scope_cap)

    def cumulative(n: int, scoped: bool) -> float:
        total = 0.0
        for i in range(n):
            accreted = min(i * p, cap) if scoped else i * p
            total += s0 + accreted
        return total

    base = [cumulative(n, False) for n in depths]
    scoped = [cumulative(n, True) for n in depths]

    # Empirical quadratic fit of the baseline; the leading coefficient should
    # recover Theorem 1's p/2 (closed form T(n) = n*s0 + p*n(n-1)/2).
    coeffs = np.polyfit(depths, base, 2)  # [c2, c1, c0]
    c2_theory = p / 2.0
    stats["snowball"] = {
        "fit_c2": float(coeffs[0]),
        "fit_c1": float(coeffs[1]),
        "fit_c0": float(coeffs[2]),
        "theory_c2": c2_theory,
        "depth40_baseline": base[-1],
        "depth40_scoped": scoped[-1],
        "depth40_ratio": base[-1] / scoped[-1],
    }

    fig, ax = plt.subplots()
    ax.plot(depths, base, "o", color=BLUE, ms=3, label="baseline (State Snowball)")
    fit = np.poly1d(coeffs)
    ax.plot(depths, fit(depths), "-", color=BLUE, lw=1.2,
            label=rf"fit $\hat c_2 n^2$, $\hat c_2={coeffs[0]:.1f}$ ($p/2={c2_theory:.0f}$)")
    ax.plot(depths, scoped, "s-", color=GREEN, ms=3, lw=1.2,
            label="scoped (Adapter Node, linear)")
    ax.set_xlabel("loop depth $n$")
    ax.set_ylabel("cumulative prompt tokens")
    ax.legend(fontsize=7)
    ax.set_title("State Snowball: $\\Theta(n^2)$ vs. bounded scope")
    _save(fig, "snowball_fit")


# --------------------------------------------------------------------------
# 2. Ablation bars with 95% CIs.
# --------------------------------------------------------------------------
def fig_ablation(ab: Dict[str, Any], stats: Dict[str, Any]) -> None:
    conds = ["+scope", "+scope+route", "+full"]
    metrics = [("tokens", "tokens"), ("usd", "USD"), ("carbon_tv_g", "carbon")]
    x = np.arange(len(conds))
    width = 0.26
    fig, ax = plt.subplots()
    rec: Dict[str, Any] = {}
    for j, (key, label) in enumerate(metrics):
        pts, los, his = [], [], []
        for c in conds:
            ci = ab["conditions"][c]["reduction_ci"][key]
            pts.append(ci["point"])
            los.append(ci["point"] - ci["lo"])
            his.append(ci["hi"] - ci["point"])
            rec[f"{c}:{key}"] = ci
        ax.bar(x + (j - 1) * width, pts, width, yerr=[los, his], capsize=3,
               color=[BLUE, ORANGE, GREEN][j], label=label, error_kw={"lw": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylabel("% reduction vs. baseline")
    ax.set_title("Per-lever reduction (20 seeds, 95% CI)")
    ax.legend(fontsize=8)
    stats["ablation"] = rec
    _save(fig, "ablation_bars")


# --------------------------------------------------------------------------
# 3 & 4. Calibration scatter; reliability (Normal-σ gate vs. split conformal).
# --------------------------------------------------------------------------
def _learned_samples(ab: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        s for s in ab["calibration_samples"]
        if s["source"] == "learned" and s["cost_std"] and s["cost_std"] > 0.0
    ]


def fig_calibration(ab: Dict[str, Any], stats: Dict[str, Any]) -> None:
    s = _learned_samples(ab)
    pred = np.array([x["cost_hat"] for x in s])
    act = np.array([x["actual"] for x in s])
    fig, ax = plt.subplots()
    ax.scatter(pred, act, s=4, alpha=0.15, color=BLUE, edgecolors="none")
    lo, hi = float(min(pred.min(), act.min())), float(max(pred.max(), act.max()))
    ax.plot([lo, hi], [lo, hi], "-", color=GREY, lw=1.0, label="$y=x$")
    r2 = float(np.corrcoef(pred, act)[0, 1] ** 2)
    mae = float(np.mean(np.abs(act - pred)))
    wape = float(np.sum(np.abs(act - pred)) / np.sum(act))
    ax.set_xlabel("predicted token cost $\\hat c$")
    ax.set_ylabel("actual token cost")
    ax.set_title(f"Forecast calibration ($R^2={r2:.3f}$, WAPE={wape*100:.1f}%)")
    ax.legend(fontsize=8)
    stats["calibration"] = {"n": len(s), "r2": r2, "mae": mae, "wape": wape}
    _save(fig, "calibration")


def fig_reliability(ab: Dict[str, Any], stats: Dict[str, Any]) -> None:
    s = _learned_samples(ab)
    rng = random.Random(0)
    idx = list(range(len(s)))
    rng.shuffle(idx)
    half = len(idx) // 2
    cal = [s[i] for i in idx[:half]]
    test = [s[i] for i in idx[half:]]
    # One-sided conformity scores on the calibration split: r = actual - cost_hat.
    cal_scores = sorted(x["actual"] - x["cost_hat"] for x in cal)

    deltas = [0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
    nominal, cov_normal, cov_conf = [], [], []
    for d in deltas:
        z = NormalDist().inv_cdf(1.0 - d)
        # Split-conformal one-sided quantile at level 1-δ.
        k = min(len(cal_scores) - 1, math.ceil((len(cal_scores) + 1) * (1.0 - d)) - 1)
        q = cal_scores[max(0, k)]
        cn = statistics.fmean(
            1.0 if x["actual"] <= x["cost_hat"] + z * x["cost_std"] else 0.0 for x in test
        )
        cc = statistics.fmean(1.0 if x["actual"] <= x["cost_hat"] + q else 0.0 for x in test)
        nominal.append(1.0 - d)
        cov_normal.append(cn)
        cov_conf.append(cc)

    fig, ax = plt.subplots()
    ax.plot([0.5, 1.0], [0.5, 1.0], "--", color=GREY, lw=1.0, label="perfect")
    ax.plot(nominal, cov_normal, "o-", color=ORANGE, ms=3, lw=1.2, label="Normal-$\\sigma$ gate")
    ax.plot(nominal, cov_conf, "s-", color=GREEN, ms=3, lw=1.2, label="split conformal")
    ax.set_xlabel("nominal coverage $1-\\delta$")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Gate reliability (held-out test split)")
    ax.legend(fontsize=8, loc="lower right")
    stats["reliability"] = {
        "n_cal": len(cal), "n_test": len(test),
        "deltas": deltas, "nominal": nominal,
        "coverage_normal": cov_normal, "coverage_conformal": cov_conf,
    }
    _save(fig, "reliability")


# --------------------------------------------------------------------------
# 5. Delta sensitivity: admission vs. overshoot under a finite budget.
# --------------------------------------------------------------------------
def fig_delta(stats: Dict[str, Any]) -> None:
    # A finite-budget stream: pre-train the estimator, then gate a fresh stream.
    # Sweeping δ trades admission throughput against realized overspend.
    from green_sarc import (
        Action, Budget, GovernanceContext, LearnedEstimator,
        ModelProfile, PreActionGate, TableCarbonModel, TableCostModel,
    )
    from green_sarc.auditor import AuditRecord

    cost_model = TableCostModel(profiles={"m": ModelProfile(energy_per_token_kwh=3e-7,
                                                            usd_per_prompt_token=5e-7,
                                                            usd_per_completion_token=1.5e-6)})
    carbon = TableCarbonModel(intensities={"r": 300.0})

    # High forecast variance makes the (1-δ) safety margin z·σ material near the
    # budget boundary, so δ genuinely trades admission throughput for overspend.
    NOISE = 90.0

    def make_estimator() -> LearnedEstimator:
        est = LearnedEstimator(cost_model, carbon, min_samples=5)
        warm = random.Random(1)
        for _ in range(400):
            prompt = warm.randint(100, 500)
            completion = max(1, int(40 + 0.6 * prompt + warm.gauss(0, NOISE)))
            est.update(AuditRecord(
                action_id="w", action_kind="k", model="m", region="r",
                predicted_cost=0.0, predicted_carbon=0.0, confidence=0.0,
                actual_cost=float(prompt + completion), actual_carbon=0.0,
                budget_remaining_tokens=0.0, carbon_remaining=0.0,
                carbon_intensity=300.0, admitted=True, verdict="admit",
                prompt_tokens=prompt))
        return est

    deltas = [0.40, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]
    admit_rate, overshoot_rate = [], []
    for d in deltas:
        admits = overshoots = total = 0
        for seed in range(40):
            est = make_estimator()
            gate = PreActionGate(est)
            stream = random.Random(1000 + seed)
            # Budget sized so the gate fills it and must adjudicate the boundary.
            budget = Budget(token_budget=18_000.0, carbon_ceiling=1e12,
                            usd_budget=1e9, delta=d)
            for _ in range(120):
                prompt = stream.randint(100, 500)
                completion = max(1, int(40 + 0.6 * prompt + stream.gauss(0, NOISE)))
                actual = float(prompt + completion)
                action = Action(kind="k", model="m", region="r",
                                prompt_tokens=prompt, max_tokens=4000)
                dec = gate.evaluate(action, GovernanceContext(budget=budget, timestamp=0.0))
                total += 1
                if dec.admitted:
                    admits += 1
                    rem_before = budget.remaining_tokens()
                    budget.spend(actual, 0.0, 0.0)
                    if actual > rem_before:  # realized spend exceeded what was left
                        overshoots += 1
        admit_rate.append(admits / total)
        overshoot_rate.append(overshoots / max(admits, 1))

    fig, ax = plt.subplots()
    ax.plot(deltas, admit_rate, "o-", color=BLUE, ms=4, label="admission rate")
    ax.set_xscale("log")
    ax.set_xlabel("gate risk level $\\delta$ (log)")
    ax.set_ylabel("admission rate", color=BLUE)
    ax.tick_params(axis="y", labelcolor=BLUE)
    ax.set_ylim(0, 0.4)
    ax2 = ax.twinx()
    ax2.plot(deltas, [100 * r for r in overshoot_rate], "s-", color=ORANGE, ms=4,
             label="overspend rate (admitted)")
    ax2.set_ylabel("overspend rate (%)", color=ORANGE)
    ax2.tick_params(axis="y", labelcolor=ORANGE)
    ax2.grid(False)
    ax.set_title("Gate $\\delta$: throughput vs. overspend")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="center left")
    stats["delta"] = {"deltas": deltas, "admission_rate": admit_rate,
                      "overspend_rate": overshoot_rate}
    _save(fig, "delta_sensitivity")


# --------------------------------------------------------------------------
# 6. Cold start: forecast MAE vs. actions seen.
# --------------------------------------------------------------------------
def fig_coldstart(lc: Dict[str, Any], stats: Dict[str, Any]) -> None:
    hist = lc["history"]
    n = [h["i"] for h in hist]
    err = [h["abs_error"] for h in hist]
    # Rolling-window MAE (window 15) to show the cold-start -> learned transition.
    w = 15
    roll = [statistics.fmean(err[max(0, i - w + 1): i + 1]) for i in range(len(err))]
    fig, ax = plt.subplots()
    ax.plot(n, err, ".", color=GREY, ms=3, alpha=0.4, label="per-action |error|")
    ax.plot(n, roll, "-", color=BLUE, lw=1.5, label=f"rolling MAE (w={w})")
    ax.axhline(lc["ground_truth"]["noise_std"], color=GREEN, ls="--", lw=1.0,
               label=f"noise floor ($\\sigma$={lc['ground_truth']['noise_std']:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("actions observed")
    ax.set_ylabel("token-cost MAE (log)")
    ax.set_title("Cold start: forecast error decays as the loop learns")
    ax.legend(fontsize=8)
    stats["coldstart"] = {
        "mae_first_half": lc["mae_first_half"], "mae_second_half": lc["mae_second_half"],
        "wape_second_half": lc["wape_second_half"],
    }
    _save(fig, "coldstart")


# --------------------------------------------------------------------------
# 7. Soft-penalty Lagrangian baseline vs. the hard gate.
# --------------------------------------------------------------------------
def fig_penalty(stats: Dict[str, Any]) -> None:
    # N actions per run with stochastic cost; a hard budget B. A soft penalty
    # admits action i iff value - λ·cost > 0 (a per-action threshold, budget-
    # blind); the gate admits in arrival order while the forecast fits B.
    N, SEEDS = 200, 200
    value = 1.0
    B = 60_000.0

    def costs(seed: int) -> List[float]:
        r = random.Random(7000 + seed)
        out = []
        for _ in range(N):
            prompt = r.randint(100, 500)
            completion = max(1, int(40 + 0.6 * prompt + r.gauss(0, 20)))
            out.append(float(prompt + completion))
        return out

    lambdas = np.logspace(-4.2, -2.6, 22)
    mean_spend, breach_prob = [], []
    for lam in lambdas:
        thresh = value / lam  # admit iff cost < value/λ
        spends, breaches = [], 0
        for seed in range(SEEDS):
            cs = costs(seed)
            spend = sum(c for c in cs if c < thresh)
            spends.append(spend)
            if spend > B:
                breaches += 1
        mean_spend.append(statistics.fmean(spends) / B)
        breach_prob.append(breaches / SEEDS)

    # The gate: arrival-order admission while forecast (≈ actual here) fits B.
    gate_spends = []
    for seed in range(SEEDS):
        cs = costs(seed)
        spend = 0.0
        for c in cs:
            if spend + c <= B:
                spend += c
        gate_spends.append(spend / B)
    gate_breach = sum(1 for s in gate_spends if s > 1.0 + 1e-9) / SEEDS

    # λ that matches the budget in expectation, and its breach probability.
    j = int(np.argmin([abs(m - 1.0) for m in mean_spend]))
    stats["penalty"] = {
        "B": B, "lambda_star": float(lambdas[j]),
        "penalty_breach_prob_at_match": breach_prob[j],
        "gate_breach_prob": gate_breach,
        "gate_mean_spend_frac": statistics.fmean(gate_spends),
    }

    fig, ax = plt.subplots()
    ax.axhline(1.0, color=GREY, ls="--", lw=1.0, label="budget $B$")
    ax.plot(lambdas, mean_spend, "o-", color=ORANGE, ms=3, lw=1.2,
            label="soft penalty: mean spend / $B$")
    ax2 = ax.twinx()
    ax2.plot(lambdas, breach_prob, "s-", color=BLUE, ms=3, lw=1.2,
             label="soft penalty: P(breach)")
    ax2.set_ylabel("P(budget breach)", color=BLUE)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor=BLUE)
    ax2.grid(False)
    ax.axhline(statistics.fmean(gate_spends), color=GREEN, lw=1.4,
               label=f"gate: spend/$B$ (P(breach)={gate_breach:.0%})")
    ax.set_xscale("log")
    ax.set_xlabel("penalty weight $\\lambda$ (log)")
    ax.set_ylabel("mean spend / $B$", color=ORANGE)
    ax.tick_params(axis="y", labelcolor=ORANGE)
    ax.set_title("Soft penalty cannot guarantee the budget; the gate can")
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [ln.get_label() for ln in lines], fontsize=6.5, loc="center right")
    _save(fig, "penalty_vs_gate")


def main() -> int:
    ab = json.loads((DATA / "ibp_ablation.json").read_text())
    lc = json.loads((DATA / "learning_curve.json").read_text())
    cfg = IBPConfig(n_skus=ab["config"]["n_skus"], depth=ab["config"]["depth"])

    stats: Dict[str, Any] = {}
    fig_snowball(cfg, stats)
    fig_ablation(ab, stats)
    fig_calibration(ab, stats)
    fig_reliability(ab, stats)
    fig_delta(stats)
    fig_coldstart(lc, stats)
    fig_penalty(stats)

    (DATA / "figure_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"wrote 7 figures to {FIGS}")
    print(f"wrote {DATA / 'figure_stats.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
