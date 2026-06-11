"""Multi-step real-trace ablation on SWE-rebench OpenHands trajectories (§11.6).

Replays real multi-step agent plans (system/user/assistant/tool turns) through the
four-condition Green SARC stack, fits the State-Snowball quadratic on real
cumulative-prompt curves, and reports the breaker firing rate. Each assistant turn
is a governed step: its prompt is the cumulative token count of all preceding
turns (the real context accretion), its completion the assistant message. Token
counts via tiktoken; no LLM is called. The model is Qwen3-Coder-480B, mapped to
the benchmark's ``frontier`` profile.

Dataset: nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0), streamed (the full
file is ~2 GB); a compact (traj, turn, prompt_tokens, completion_tokens) table is
cached to paper/data/swe_rebench_subset.parquet (git-ignored).

    python paper/scripts/run_multistep_replay.py --out paper/data/multistep_replay.json \
        --max-trajectories 3000 --bootstrap 1000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root

from green_sarc import (  # noqa: E402
    ActionTimeMonitor,
    AdapterNode,
    CircuitTripped,
    carbon_for_tokens,
)

from benchmarks.ibp import BIG_MODEL, REGION, SMALL_MODEL, _models  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper" / "data" / "swe_rebench_subset.parquet"
DATASET = "nebius/SWE-rebench-openhands-trajectories"


def _tokenizer():
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return lambda s: len(enc.encode(s, disallowed_special=())) if isinstance(s, str) else 0


def _extract(max_trajectories: int) -> tuple[List[List[Dict[str, float]]], List[int]]:
    """Return per-trajectory governed assistant steps and a parallel resolved flag.

    The second list carries the dataset's real ``resolved`` field (0/1) per
    trajectory — the task-outcome signal used by the cost-utility frontier.
    """
    if CACHE.exists():
        import pandas as pd

        df = pd.read_parquet(CACHE)
        has_resolved = "resolved" in df.columns
        trajs: Dict[int, List[Dict[str, float]]] = {}
        resolved: Dict[int, int] = {}
        for r in df.itertuples(index=False):
            trajs.setdefault(int(r.traj), []).append(
                {"prompt": float(r.prompt_tokens), "completion": float(r.completion_tokens)}
            )
            if has_resolved:
                resolved[int(r.traj)] = int(r.resolved)
        keys = list(trajs.keys())
        return [trajs[k] for k in keys], [resolved.get(k, 0) for k in keys]

    from datasets import load_dataset

    tok = _tokenizer()
    ds = load_dataset(DATASET, split="train", streaming=True)
    out: List[List[Dict[str, float]]] = []
    out_resolved: List[int] = []
    rows = []
    for ti, ex in enumerate(ds):
        traj = ex.get("trajectory") or []
        is_resolved = int(ex.get("resolved") or 0)
        cum = 0
        depth = 0
        steps: List[Dict[str, float]] = []
        for msg in traj:
            if not isinstance(msg, dict):
                continue
            ntok = tok(msg.get("content", ""))
            if msg.get("role") == "assistant" and cum > 0:
                depth += 1
                steps.append({"prompt": float(cum), "completion": float(ntok)})
                rows.append((ti, depth, cum, ntok, is_resolved))
            cum += ntok
        if steps:
            out.append(steps)
            out_resolved.append(is_resolved)
        if len(out) >= max_trajectories:
            break
    try:
        import pandas as pd

        pd.DataFrame(
            rows, columns=["traj", "turn", "prompt_tokens", "completion_tokens", "resolved"]
        ).to_parquet(CACHE)
    except Exception as e:
        print(f"  (cache write skipped: {e})")
    return out, out_resolved


CONDITIONS = [
    ("baseline", frozenset()),
    ("+scope", frozenset({"scope"})),
    ("+scope+route", frozenset({"scope", "routing"})),
    ("+full", frozenset({"scope", "routing", "gate", "monitor"})),
]


def _traj_metrics(traj, features, route_simple, cost_model, carbon, scope_cap, max_loops):
    adapter = AdapterNode(scope_cap)
    monitor = ActionTimeMonitor(max_loops=max_loops) if "monitor" in features else None
    tokens = usd = carbon_g = 0.0
    tripped = False
    for s in traj:
        prompt = adapter.bound(s["prompt"]) if "scope" in features else s["prompt"]
        model = SMALL_MODEL if ("routing" in features and route_simple) else BIG_MODEL
        if monitor is not None:
            try:
                monitor.before()
            except CircuitTripped:
                tripped = True
                break
        total = prompt + s["completion"]
        tokens += total
        usd += cost_model.usd(model, prompt, s["completion"])
        carbon_g += carbon_for_tokens(cost_model, carbon, model, total, REGION, 0.0)
        if monitor is not None:
            try:
                monitor.after(total)
            except CircuitTripped:
                tripped = True
                break
    return tokens, usd, carbon_g, tripped


def _paired_ci(base: np.ndarray, treat: np.ndarray, boot: int, rng) -> Dict[str, float]:
    n = len(base)
    point = 100.0 * (base.sum() - treat.sum()) / base.sum() if base.sum() else 0.0
    reds = np.empty(boot)
    for i in range(boot):
        idx = rng.integers(0, n, n)
        b, t = base[idx].sum(), treat[idx].sum()
        reds[i] = 0.0 if b == 0 else 100.0 * (b - t) / b
    reds.sort()
    return {
        "point": point,
        "lo": float(reds[int(0.025 * boot)]),
        "hi": float(reds[int(0.975 * boot)]),
    }


def snowball_fit(trajs) -> Dict[str, Any]:
    """Fit cumulative prompt vs assistant-turn index per trajectory; compare c2 to p/2."""
    c2s, per_turn_growth = [], []
    for tr in trajs:
        if len(tr) < 4:
            continue
        n = np.arange(1, len(tr) + 1, dtype=float)
        cum = np.cumsum([s["prompt"] for s in tr])
        c2, c1 = np.polyfit(n, cum, 2)[:2]
        c2s.append(float(c2))
        prompts = [s["prompt"] for s in tr]
        per_turn_growth.extend(np.diff(prompts).tolist())
    p = float(np.median([g for g in per_turn_growth if g > 0])) if per_turn_growth else 0.0
    c2s_arr = np.array(c2s)
    return {
        "n_trajectories_fit": len(c2s),
        "median_c2": float(np.median(c2s_arr)),
        "frac_c2_positive": float(np.mean(c2s_arr > 0)),
        "median_per_turn_growth_p": p,
        "theoretical_c2_p_over_2": p / 2.0,
        "c2_sample": [float(x) for x in c2s_arr[:3000]],  # for the histogram figure
    }


def cost_utility_frontier(trajs, resolved, multipliers=(0.5, 1.0, 2.0, 4.0)) -> Dict[str, Any]:
    """Observational cost-utility frontier: token reduction vs resolution-rate harm.

    For each scope cap ``C = m * median(step prompt)`` we cap each governed step's
    prompt at ``C`` (mirroring ``AdapterNode.bound``) and report:

    - ``token_reduction_pct``: tokens saved vs the uncapped replay.
    - ``resolution_rate``: baseline mean(resolved) (cap-independent reference).
    - ``worst_case_resolution_rate``: resolved AND untruncated — i.e. every
      *resolved* trajectory whose actually-used context the cap would have cut is
      assumed to flip to unresolved. This is an **upper bound on harm**, not a
      causal estimate: truncation is simulated on logged trajectories, so the
      agent could not adapt to the smaller context.
    - ``frac_trajectories_truncated``: share of trajectories with >=1 cut step.

    All quantities are computed only from which trajectories' real context a cap
    would have touched; no outcome is re-simulated.
    """
    resolved_arr = np.array(resolved, dtype=float)
    n = len(trajs)
    base_resolution = float(resolved_arr.mean()) if n else 0.0
    all_prompts = [s["prompt"] for tr in trajs for s in tr]
    median_prompt = float(np.median(all_prompts)) if all_prompts else 0.0
    base_tokens = float(sum(s["prompt"] + s["completion"] for tr in trajs for s in tr))

    points = []
    for m in multipliers:
        cap = m * median_prompt
        capped_tokens = 0.0
        truncated = np.zeros(n, dtype=bool)
        for j, tr in enumerate(trajs):
            for s in tr:
                capped_tokens += min(s["prompt"], cap) + s["completion"]
                if s["prompt"] > cap:
                    truncated[j] = True
        token_reduction = (
            100.0 * (base_tokens - capped_tokens) / base_tokens if base_tokens else 0.0
        )
        # Worst case: a truncated-and-resolved trajectory is assumed to fail.
        survived = resolved_arr * (~truncated)
        points.append(
            {
                "cap_multiplier": m,
                "cap_tokens": cap,
                "token_reduction_pct": token_reduction,
                "resolution_rate": base_resolution,
                "worst_case_resolution_rate": float(survived.mean()) if n else 0.0,
                "frac_trajectories_truncated": float(truncated.mean()) if n else 0.0,
            }
        )
    return {
        "median_prompt_tokens": median_prompt,
        "baseline_resolution_rate": base_resolution,
        "n_resolved": int(resolved_arr.sum()),
        "points": points,
    }


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(prog="run_multistep_replay", description=__doc__)
    ap.add_argument("--out", default="paper/data/multistep_replay.json")
    ap.add_argument("--max-trajectories", type=int, default=3000)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args(argv)

    trajs, resolved = _extract(args.max_trajectories)
    cost_model, carbon_fixed, _ = _models()
    all_prompts = [s["prompt"] for tr in trajs for s in tr]
    scope_cap = int(2 * float(np.median(all_prompts)))
    depths = [len(tr) for tr in trajs]
    median_depth = float(np.median(depths))
    max_loops = math.ceil(1.5 * median_depth)  # breaker fires on the long-plan tail

    rng = np.random.default_rng(0)
    route_simple = rng.random(len(trajs)) < 0.5
    per: Dict[str, Dict[str, np.ndarray]] = {}
    trips: Dict[str, int] = {}
    for name, feats in CONDITIONS:
        tok = np.empty(len(trajs))
        usd = np.empty(len(trajs))
        car = np.empty(len(trajs))
        ntrip = 0
        for j, tr in enumerate(trajs):
            tk, us, cr, tp = _traj_metrics(
                tr, feats, bool(route_simple[j]), cost_model, carbon_fixed, scope_cap, max_loops
            )
            tok[j], usd[j], car[j] = tk, us, cr
            ntrip += int(tp)
        per[name] = {"tokens": tok, "usd": usd, "carbon": car}
        trips[name] = ntrip

    bootrng = np.random.default_rng(1)
    base = per["baseline"]
    conditions = {}
    for name, _ in CONDITIONS:
        entry: Dict[str, Any] = {
            "breaker_trips": trips[name],
            "breaker_trip_rate": trips[name] / len(trajs),
        }
        if name != "baseline":
            entry["reduction_ci"] = {
                m: _paired_ci(base[m], per[name][m], args.bootstrap, bootrng)
                for m in ("tokens", "usd", "carbon")
            }
        conditions[name] = entry

    data = {
        "dataset": "nebius/SWE-rebench-openhands-trajectories (CC-BY-4.0)",
        "n_trajectories": len(trajs),
        "median_depth": median_depth,
        "max_depth": int(max(depths)),
        "median_prompt_tokens": float(np.median(all_prompts)),
        "scope_cap": scope_cap,
        "breaker_max_loops": max_loops,
        "conditions": conditions,
        "snowball_fit": snowball_fit(trajs),
        "cost_utility": cost_utility_frontier(trajs, resolved),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print(
        f"wrote {out}: {len(trajs)} trajectories, median depth {median_depth:.0f}, "
        f"max {max(depths)}, scope_cap {scope_cap}, breaker max_loops {max_loops}"
    )
    for name, _ in CONDITIONS[1:]:
        r = conditions[name]["reduction_ci"]
        print(
            f"  {name:<14} tok {r['tokens']['point']:5.1f}% usd {r['usd']['point']:5.1f}% "
            f"carbon {r['carbon']['point']:5.1f}%"
        )
    print(
        f"  +full breaker trip rate: {conditions['+full']['breaker_trip_rate'] * 100:.1f}% "
        f"of trajectories"
    )
    sf = data["snowball_fit"]
    print(
        f"  snowball: median c2={sf['median_c2']:.1f} vs p/2={sf['theoretical_c2_p_over_2']:.1f}, "
        f"frac c2>0 = {sf['frac_c2_positive']:.2f}"
    )
    cu = data["cost_utility"]
    print(
        f"  cost-utility: baseline resolution {cu['baseline_resolution_rate'] * 100:.1f}% "
        f"({cu['n_resolved']} resolved)"
    )
    for pt in cu["points"]:
        print(
            f"    cap {pt['cap_multiplier']:>4}x: tok -{pt['token_reduction_pct']:.1f}%, "
            f"worst-case resolution {pt['worst_case_resolution_rate'] * 100:.1f}%, "
            f"truncated {pt['frac_trajectories_truncated'] * 100:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
