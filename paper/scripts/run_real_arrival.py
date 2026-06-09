"""Real-arrival ablation on BurstGPT (paper §11).

Replays the public BurstGPT trace (real Azure GPT-3.5/GPT-4 serving traffic)
through the four-condition Green SARC ablation (baseline / +scope / +scope+route
/ +full), with paired-bootstrap 95% CIs, and a small binding-budget companion to
the §9 sweep.  This converts the headline ablation from synthetic IBP arrivals to
a real production trace.

Dataset: HPMLL/BurstGPT, ``data/BurstGPT_1.csv`` (CC-BY-4.0; Wang et al., KDD'25,
arXiv:2401.17644).  Schema: ``Timestamp, Model, Request tokens, Response tokens,
Total tokens, Log Type``.  We use token counts only -- no LLM is called.  The
download is cached as ``paper/data/burstgpt_subset.parquet`` (git-ignored); only
the derived ``paper/data/real_arrival.json`` is committed.

This is paper-side analysis: it reuses the real library primitives
(:class:`AdapterNode`, :class:`ActionTimeMonitor`, :class:`TableCostModel`, the
Pre-Action Gate / :class:`Budget` for the binding-budget part) and the model
profiles from ``benchmarks.ibp`` (read-only); it does not modify any library or
benchmark code.

    python paper/scripts/run_real_arrival.py --out paper/data/real_arrival.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `benchmarks`

from green_sarc import (  # noqa: E402
    Action,
    ActionTimeMonitor,
    AdapterNode,
    Budget,
    CircuitTripped,
    GovernanceContext,
    LearnedEstimator,
    PreActionGate,
    carbon_for_tokens,
)
from green_sarc.auditor import AuditRecord  # noqa: E402

from benchmarks.ibp import BIG_MODEL, REGION, SMALL_MODEL, _models  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper" / "data" / "burstgpt_subset.parquet"
URL = "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"
# GPT-4 -> frontier profile; ChatGPT (GPT-3.5) -> efficient profile.
MODEL_MAP = {"GPT-4": BIG_MODEL, "ChatGPT": SMALL_MODEL}
DELTA = 0.05
BUDGET_FRACTIONS = [0.5, 1.0, 1.5]


# --------------------------------------------------------------------------
# Data: download + cache, then temporal-cluster into trajectories.
# --------------------------------------------------------------------------
def _load(max_conversations: int):
    import pandas as pd

    if CACHE.exists():
        return pd.read_parquet(CACHE)
    print(f"  downloading BurstGPT from {URL} ...")
    raw = ROOT / "paper" / "data" / "_burstgpt_raw.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, raw)
    df = pd.read_csv(raw)
    df = df[df["Response tokens"] > 0].reset_index(drop=True)  # drop failed responses
    df = df.head(max_conversations).copy()
    df = df.rename(columns={"Request tokens": "prompt", "Response tokens": "completion",
                            "Log Type": "log_type"})
    df = df[["Timestamp", "Model", "prompt", "completion", "log_type"]]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    raw.unlink(missing_ok=True)
    return df


def _trajectories(df, window_s: int, max_depth: int) -> List[List[Dict[str, Any]]]:
    """Temporal clustering: same-model rows within ``window_s`` of the previous,
    capped at ``max_depth``.  API-log rows are single-step trajectories.

    BurstGPT v1.0 has no SessionID; this reconstructs burst-level trajectories.
    When v1.1 ships SessionID this becomes a one-line groupby.
    """
    trajs: List[List[Dict[str, Any]]] = []
    open_clusters: Dict[str, List[Dict[str, Any]]] = {}
    last_ts: Dict[str, float] = {}
    for row in df.itertuples(index=False):
        model = MODEL_MAP.get(row.Model, SMALL_MODEL)
        step = {"prompt": float(row.prompt), "completion": float(row.completion),
                "model": model}
        if "API" in str(row.log_type):  # single-step
            trajs.append([step])
            continue
        key = row.Model
        cl = open_clusters.get(key)
        if (cl is None or row.Timestamp - last_ts[key] > window_s or len(cl) >= max_depth):
            cl = []
            open_clusters[key] = cl
            trajs.append(cl)
        cl.append(step)
        last_ts[key] = row.Timestamp
    return [t for t in trajs if t]


# --------------------------------------------------------------------------
# Per-trajectory ablation metrics (exercises AdapterNode / ActionTimeMonitor /
# TableCostModel / carbon_for_tokens — the real library code).
# --------------------------------------------------------------------------
def _traj_metrics(traj, features, route_simple, cost_model, carbon, scope_cap, max_loops):
    adapter = AdapterNode(scope_cap)
    monitor = ActionTimeMonitor(max_loops=max_loops) if "monitor" in features else None
    tokens = usd = carbon_g = 0.0
    trips = 0
    for step in traj:
        prompt = adapter.bound(step["prompt"]) if "scope" in features else step["prompt"]
        model = SMALL_MODEL if ("routing" in features and route_simple) else step["model"]
        completion = step["completion"]
        if monitor is not None:
            try:
                monitor.before()
            except CircuitTripped:
                trips += 1
                break
        total = prompt + completion
        tokens += total
        usd += cost_model.usd(model, prompt, completion)
        carbon_g += carbon_for_tokens(cost_model, carbon, model, total, REGION, 0.0)
        if monitor is not None:
            try:
                monitor.after(total)
            except CircuitTripped:
                trips += 1
                break
    return tokens, usd, carbon_g, trips


CONDITIONS = [
    ("baseline", frozenset()),
    ("+scope", frozenset({"scope"})),
    ("+scope+route", frozenset({"scope", "routing"})),
    ("+full", frozenset({"scope", "routing", "gate", "monitor"})),
]


def _paired_ci(base: np.ndarray, treat: np.ndarray, boot: int, rng) -> Dict[str, float]:
    n = len(base)
    point = 100.0 * (base.sum() - treat.sum()) / base.sum() if base.sum() else 0.0
    reds = np.empty(boot)
    for i in range(boot):
        idx = rng.integers(0, n, n)
        b, t = base[idx].sum(), treat[idx].sum()
        reds[i] = 0.0 if b == 0 else 100.0 * (b - t) / b
    reds.sort()
    return {"point": point, "lo": float(reds[int(0.025 * boot)]),
            "hi": float(reds[int(0.975 * boot)])}


def ablation(trajs, cost_model, carbon, scope_cap, max_loops, boot: int) -> Dict[str, Any]:
    rng = np.random.default_rng(0)
    route_simple = rng.random(len(trajs)) < 0.5  # 50% routed to the efficient model
    per: Dict[str, Dict[str, np.ndarray]] = {}
    trips_total: Dict[str, int] = {}
    for name, feats in CONDITIONS:
        tok = np.empty(len(trajs))
        usd = np.empty(len(trajs))
        car = np.empty(len(trajs))
        trips = 0
        for j, tr in enumerate(trajs):
            tk, us, cr, tp = _traj_metrics(tr, feats, bool(route_simple[j]), cost_model,
                                           carbon, scope_cap, max_loops)
            tok[j], usd[j], car[j] = tk, us, cr
            trips += tp
        per[name] = {"tokens": tok, "usd": usd, "carbon": car}
        trips_total[name] = trips

    bootrng = np.random.default_rng(1)
    base = per["baseline"]
    out: Dict[str, Any] = {"conditions": {}}
    for name, _ in CONDITIONS:
        entry: Dict[str, Any] = {
            "tokens_total": float(per[name]["tokens"].sum()),
            "usd_total": float(per[name]["usd"].sum()),
            "carbon_total": float(per[name]["carbon"].sum()),
            "breaker_trips": trips_total[name],
        }
        if name != "baseline":
            entry["reduction_ci"] = {
                m: _paired_ci(base[m], per[name][m], boot, bootrng)
                for m in ("tokens", "usd", "carbon")
            }
        out["conditions"][name] = entry
    return out


# --------------------------------------------------------------------------
# Binding-budget companion on real arrivals (§11.3) — exercises the real gate.
# --------------------------------------------------------------------------
def _warm_estimator(trajs, cost_model, carbon) -> LearnedEstimator:
    est = LearnedEstimator(cost_model, carbon, min_samples=8)
    for tr in trajs[: min(len(trajs), 3000)]:
        for s in tr:
            est.update(AuditRecord(
                action_id="w", action_kind="serve", model=s["model"], region=REGION,
                predicted_cost=0.0, predicted_carbon=0.0, confidence=0.0,
                actual_cost=s["prompt"] + s["completion"], actual_carbon=0.0,
                budget_remaining_tokens=0.0, carbon_remaining=0.0, carbon_intensity=350.0,
                admitted=True, verdict="admit", prompt_tokens=int(s["prompt"])))
    return est


def binding_budget(trajs, cost_model, carbon, e_base: float, max_loops: int) -> Dict[str, Any]:
    points = []
    for phi in BUDGET_FRACTIONS:
        B = phi * e_base
        est = _warm_estimator(trajs, cost_model, carbon)
        gate = PreActionGate(est)
        budget = Budget(token_budget=B, carbon_ceiling=1e18, usd_budget=1e12, delta=DELTA)
        attempted = admitted = over = completed = 0
        for tr in trajs:
            ok = 0
            monitor = ActionTimeMonitor(max_loops=max_loops)
            for s in tr:
                action = Action(kind="serve", model=s["model"], region=REGION,
                                prompt_tokens=int(s["prompt"]), max_tokens=8192)
                dec = gate.evaluate(action, GovernanceContext(budget=budget, timestamp=0.0))
                attempted += 1
                if not dec.admitted:
                    break
                try:
                    monitor.before()
                except CircuitTripped:
                    break
                actual = s["prompt"] + s["completion"]
                if actual > budget.remaining_tokens():
                    over += 1
                budget.spend(actual, 0.0, 0.0)
                est.update(AuditRecord(
                    action_id="x", action_kind="serve", model=s["model"], region=REGION,
                    predicted_cost=dec.forecast.cost_hat, predicted_carbon=0.0,
                    confidence=dec.forecast.confidence, actual_cost=actual, actual_carbon=0.0,
                    budget_remaining_tokens=budget.remaining_tokens(), carbon_remaining=0.0,
                    carbon_intensity=350.0, admitted=True, verdict="admit",
                    prompt_tokens=int(s["prompt"])))
                admitted += 1
                ok += 1
            if ok == len(tr):
                completed += 1
        points.append({
            "fraction": phi, "budget_tokens": B,
            "admission_rate": admitted / attempted if attempted else 0.0,
            "over_budget_incidence": over / admitted if admitted else 0.0,
            "completed_traj_rate": completed / len(trajs),
        })

    # Soft-penalty frontier vs a binding reference B = 0.5 x e_base (mirrors §9).
    b_star = 0.5 * e_base
    step_costs = [s["prompt"] + s["completion"] for tr in trajs for s in tr]
    traj_costs = [[s["prompt"] + s["completion"] for s in tr] for tr in trajs]
    frontier = []
    lo = math.log10(1.0 / max(step_costs))
    hi = math.log10(1.0 / min(c for c in step_costs if c > 0))
    for lam in [10 ** (lo + (hi - lo) * k / 39) for k in range(40)]:
        thr = 1.0 / lam
        spend = sum(c for c in step_costs if c < thr)
        completed = statistics.fmean(
            1.0 if all(c < thr for c in tc) else 0.0 for tc in traj_costs)
        frontier.append({"lambda": lam, "completed_traj_rate": completed,
                         "over_budget_incidence": 1.0 if spend > b_star else 0.0,
                         "mean_spend_frac": spend / b_star})
    return {"delta": DELTA, "fractions": BUDGET_FRACTIONS, "points": points,
            "pareto_reference_budget": b_star, "soft_penalty_frontier": frontier}


def main(argv: Any = None) -> int:
    p = argparse.ArgumentParser(prog="run_real_arrival", description=__doc__)
    p.add_argument("--out", default="paper/data/real_arrival.json")
    p.add_argument("--max-conversations", type=int, default=50_000)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--session-window-seconds", type=int, default=60)
    p.add_argument("--max-depth", type=int, default=10)
    args = p.parse_args(argv)

    df = _load(args.max_conversations)
    trajs = _trajectories(df, args.session_window_seconds, args.max_depth)
    cost_model, carbon_fixed, _ = _models()
    scope_cap = int(2 * float(np.median(df["prompt"].to_numpy())))
    sizes = [len(t) for t in trajs]
    # The breaker is a safeguard against runaway loops: it triggers only above
    # 1.5x the trajectory depth cap, so on a real serving trace with no retry
    # storms it is dormant (an honest negative; cf. its role under the IBP
    # stress scenario). A median-based threshold would be degenerate here
    # because most BurstGPT trajectories are single-step.
    max_loops = math.ceil(1.5 * args.max_depth)

    abl = ablation(trajs, cost_model, carbon_fixed, scope_cap, max_loops, args.bootstrap)
    e_base_total = abl["conditions"]["baseline"]["tokens_total"]  # total baseline tokens
    bb = binding_budget(trajs, cost_model, carbon_fixed, e_base_total, max_loops)

    data = {
        "dataset": "HPMLL/BurstGPT BurstGPT_1.csv (CC-BY-4.0; Wang et al. 2025)",
        "n_requests": int(len(df)),
        "n_trajectories": len(trajs),
        "median_prompt_tokens": float(np.median(df["prompt"].to_numpy())),
        "median_trajectory_depth": float(np.median(sizes)),
        "max_trajectory_depth": int(max(sizes)),
        "scope_cap": scope_cap,
        "breaker_max_loops": max_loops,
        "session_window_seconds": args.session_window_seconds,
        "bootstrap": args.bootstrap,
        "model_mix": {k: int((df["Model"] == k).sum()) for k in df["Model"].unique()},
        "ablation": abl,
        "binding_budget": bb,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(f"wrote {out}")
    print(f"  {len(df)} requests -> {len(trajs)} trajectories "
          f"(median depth {np.median(sizes):.0f}, scope_cap {scope_cap})")
    for name, _ in CONDITIONS[1:]:
        r = abl["conditions"][name]["reduction_ci"]
        print(f"  {name:<14} tok {r['tokens']['point']:5.1f}% "
              f"usd {r['usd']['point']:5.1f}% carbon {r['carbon']['point']:5.1f}%")
    print(f"  breaker trips (+full): {abl['conditions']['+full']['breaker_trips']}")
    for pt in bb["points"]:
        print(f"  B={pt['fraction']}x admit={pt['admission_rate']:.2f} "
              f"over-budget={pt['over_budget_incidence']*100:.2f}% "
              f"completed={pt['completed_traj_rate']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
