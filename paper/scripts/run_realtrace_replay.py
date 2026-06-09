"""Real-trace coverage validation on ShareGPT (paper §10).

Replays real multi-turn LLM conversations to validate the gate's calibration on
*real* (non-Gaussian) forecast residuals -- the experiment the prior draft
flagged as future work.  Dataset: ``anon8231489123/ShareGPT_Vicuna_unfiltered``
on the Hugging Face Hub (ungated; real ChatGPT/GPT-4 conversations).  We use only
token *counts* -- no LLM is called.  ShareGPT is used because LMSYS-Chat-1M is
gated and would not reproduce on a clean clone without an access token; ShareGPT
serves the same purpose (real, non-Gaussian residuals, real multi-turn depth) and
needs no credentials.

For each conversation we tokenize turns (tiktoken ``cl100k_base`` if available,
else a deterministic word-based fallback).  For each assistant ("gpt") turn we
form ``(prompt_tokens = cumulative preceding context, completion_tokens =
this turn)``.  We then:

  1. fit online OLS ``completion ~ a + b * prompt``; on a 50/50 cal/test split,
     compare Gaussian-sigma vs split-conformal empirical coverage over delta in
     [0.01, 0.4], and report residual non-Gaussianity (skew, kurtosis,
     Anderson-Darling);  -> realtrace_calibration.json
  2. fit the per-conversation cumulative prompt tokens vs turn depth to a
     quadratic, with a paired-bootstrap CI on the leading coefficient (the §4.3
     real-traffic State-Snowball check);
  3. (shift mode) train conformal on short conversations, deploy on long ones,
     and compare fixed-quantile vs adaptive conformal inference (ACI) rolling
     coverage;  -> realtrace_shift.json

The extracted compact table is cached to ``paper/data/sharegpt_subset.parquet``
(git-ignored, regenerated on first run); the two JSONs are committed provenance.

    python paper/scripts/run_realtrace_replay.py \
        --out paper/data/realtrace_calibration.json \
        --shift-out paper/data/realtrace_shift.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "paper" / "data" / "sharegpt_subset.parquet"
DATASET = "anon8231489123/ShareGPT_Vicuna_unfiltered"
DATA_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
MAX_CONVOS = 12_000
MAX_PAIRS = 50_000
DELTAS = [0.40, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]


# --------------------------------------------------------------------------
# Tokenization (counts only; deterministic fallback if tiktoken is absent).
# --------------------------------------------------------------------------
def _make_tokenizer() -> Tuple[Any, str]:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        # Treat literal control strings in the data as ordinary text.
        return (lambda s: len(enc.encode(s, disallowed_special=())), "tiktoken/cl100k_base")
    except Exception:
        # ~0.75 tokens/word is a standard rough English heuristic.
        return (lambda s: max(1, int(round(len(s.split()) / 0.75))), "wordcount/0.75")


def _extract() -> Tuple[List[Dict[str, Any]], str]:
    """Stream ShareGPT, tokenize turns, return per-(gpt-turn) records."""
    from datasets import load_dataset

    tok, tok_name = _make_tokenizer()
    ds = load_dataset(DATASET, data_files=DATA_FILE, split="train", streaming=True)
    rows: List[Dict[str, Any]] = []
    n_convo = 0
    for ex in ds:
        conv = ex.get("conversations") or []
        if not conv:
            continue
        turns = []
        for t in conv:
            if isinstance(t, str):  # streamed items are JSON strings
                try:
                    t = json.loads(t)
                except (ValueError, TypeError):
                    continue
            if isinstance(t, dict):
                turns.append(t)
        if not turns:
            continue
        n_convo += 1
        cum = 0
        depth = 0
        turn_tokens = [tok(t.get("value") or "") for t in turns]
        for t, ntok in zip(turns, turn_tokens):
            role = t.get("from", "")
            # Cap to a realistic deployment context window (8k prompt / 4k
            # completion); raw ShareGPT contains whole-document pastes that are
            # not representative of a governed step.
            if role == "gpt" and 0 < cum <= 8192 and 0 < ntok <= 4096:
                depth += 1
                rows.append(
                    {
                        "conv_id": n_convo,
                        "turn_index": depth,
                        "prompt_tokens": float(cum),
                        "completion_tokens": float(ntok),
                    }
                )
            cum += ntok
        if len(rows) >= MAX_PAIRS or n_convo >= MAX_CONVOS:
            break
    return rows, tok_name


def _load_or_extract() -> Tuple[List[Dict[str, Any]], str]:
    if CACHE.exists():
        import pandas as pd

        df = pd.read_parquet(CACHE)
        return df.to_dict("records"), str(df.attrs.get("tokenizer", "cached"))
    rows, tok_name = _extract()
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        df.attrs["tokenizer"] = tok_name
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(CACHE)
    except Exception as e:  # caching is best-effort
        print(f"  (cache write skipped: {e})")
    return rows, tok_name


# --------------------------------------------------------------------------
# §10.1-10.3: calibration + coverage + non-Gaussianity.
# --------------------------------------------------------------------------
def _ols(prompt: np.ndarray, completion: np.ndarray) -> Tuple[float, float, float]:
    """Return (alpha, beta, residual_std) for completion ~ alpha + beta*prompt."""
    n = len(prompt)
    sx, sy = prompt.sum(), completion.sum()
    sxx, sxy = float((prompt * prompt).sum()), float((prompt * completion).sum())
    denom = n * sxx - sx * sx
    beta = (n * sxy - sx * sy) / denom if denom > 0 else 0.0
    alpha = (sy - beta * sx) / n
    resid = completion - (alpha + beta * prompt)
    sigma = float(np.sqrt(np.sum(resid ** 2) / max(n - 2, 1)))
    return alpha, beta, sigma


def calibration(rows: List[Dict[str, Any]], tok_name: str) -> Dict[str, Any]:
    from scipy import stats

    prompt = np.array([r["prompt_tokens"] for r in rows])
    completion = np.array([r["completion_tokens"] for r in rows])
    idx = np.arange(len(rows))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    half = len(idx) // 2
    cal, test = idx[:half], idx[half:]

    alpha, beta, sigma = _ols(prompt[cal], completion[cal])
    pred_cal = alpha + beta * prompt[cal]
    pred_test = alpha + beta * prompt[test]
    r_cal = completion[cal] - pred_cal            # one-sided scores (over-shoot)
    r_test = completion[test] - pred_test

    nominal, cov_gauss, cov_conf = [], [], []
    for d in DELTAS:
        z = NormalDist().inv_cdf(1.0 - d)
        q = float(np.quantile(r_cal, 1.0 - d, method="higher"))
        cov_gauss.append(float(np.mean(r_test <= z * sigma)))
        cov_conf.append(float(np.mean(r_test <= q)))
        nominal.append(1.0 - d)

    # Non-Gaussianity diagnostics on calibration residuals.
    ad = stats.anderson(r_cal, dist="norm")
    _, normaltest_p = stats.normaltest(r_cal)
    crit_1pct = float(ad.critical_values[-1])  # 1% significance level

    # δ = 0.05 deviation (acceptance gate).
    j05 = DELTAS.index(0.05)
    return {
        "dataset": DATASET,
        "tokenizer": tok_name,
        "n_pairs": len(rows),
        "n_cal": int(half),
        "n_test": int(len(idx) - half),
        "ols": {"alpha": alpha, "beta": beta, "sigma": sigma},
        "deltas": DELTAS,
        "nominal": nominal,
        "coverage_gaussian": cov_gauss,
        "coverage_conformal": cov_conf,
        "gaussian_dev_at_005_pp": abs(cov_gauss[j05] - (1 - 0.05)) * 100.0,
        "conformal_dev_at_005_pp": abs(cov_conf[j05] - (1 - 0.05)) * 100.0,
        "max_conformal_dev_pp": max(abs(c - n) for c, n in zip(cov_conf, nominal)) * 100.0,
        "residuals": {
            "skew": float(stats.skew(r_cal)),
            "kurtosis_excess": float(stats.kurtosis(r_cal)),
            "anderson_darling_stat": float(ad.statistic),
            "anderson_darling_crit_1pct": crit_1pct,
            "rejects_normal_1pct": bool(ad.statistic > crit_1pct),
            "normaltest_p": float(normaltest_p),
        },
        "residual_sample": [float(x) for x in r_cal[:4000]],  # for the histogram/Q-Q figure
    }


def turn_depth_quadratic(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """§4.3: cumulative prompt tokens vs turn depth on real chat traffic."""
    # Per conversation: depth = #gpt turns; total billed prompt = sum of the
    # per-turn prompt contexts (the analog of T_prompt(n) in Theorem 1).
    by_conv: Dict[int, Tuple[int, float]] = {}
    for r in rows:
        cid = int(r["conv_id"])
        d0, p0 = by_conv.get(cid, (0, 0.0))
        by_conv[cid] = (max(d0, int(r["turn_index"])), p0 + float(r["prompt_tokens"]))
    depths = np.array([v[0] for v in by_conv.values()], dtype=float)
    totals = np.array([v[1] for v in by_conv.values()], dtype=float)
    mask = depths >= 2  # need >=2 turns to see curvature
    depths, totals = depths[mask], totals[mask]
    c2, c1, c0 = (float(x) for x in np.polyfit(depths, totals, 2))

    rng = np.random.default_rng(1)
    n = len(depths)
    boot = []
    for _ in range(2000):
        s = rng.integers(0, n, n)
        boot.append(float(np.polyfit(depths[s], totals[s], 2)[0]))
    boot.sort()
    return {
        "n_conversations": int(n),
        "max_depth": int(depths.max()),
        "fit_c2": c2,
        "fit_c1": c1,
        "fit_c0": c0,
        "c2_ci95": [boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot))]],
        "c2_positive": bool(boot[int(0.025 * len(boot))] > 0),
    }


# --------------------------------------------------------------------------
# §10.5 / A.6: distribution shift, fixed-quantile vs adaptive conformal (ACI).
# --------------------------------------------------------------------------
def shift_experiment(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    depth = np.array([r["turn_index"] for r in rows])
    med = float(np.median(depth))
    reg1 = [r for r in rows if r["turn_index"] <= med]   # short-context regime
    reg2 = [r for r in rows if r["turn_index"] > med]    # long-context regime (shifted)
    if len(reg2) < 500:  # ensure a usable deployment stream
        reg1, reg2 = rows[: len(rows) // 2], rows[len(rows) // 2 :]

    p1 = np.array([r["prompt_tokens"] for r in reg1])
    c1arr = np.array([r["completion_tokens"] for r in reg1])
    alpha, beta, _ = _ols(p1, c1arr)
    r1 = c1arr - (alpha + beta * p1)

    delta = 0.10
    q_fixed = float(np.quantile(r1, 1.0 - delta, method="higher"))

    rng = np.random.default_rng(2)
    order = np.arange(len(reg2))
    rng.shuffle(order)
    stream = [reg2[i] for i in order]

    gamma = 0.02  # ACI learning rate (Gibbs & Candes 2021)
    alpha_t = delta
    fixed_hits, aci_hits, q_aci_hist = [], [], []
    for r in stream:
        pred = alpha + beta * r["prompt_tokens"]
        resid = r["completion_tokens"] - pred
        fixed_hits.append(1.0 if resid <= q_fixed else 0.0)
        q_aci = float(np.quantile(r1, min(max(1.0 - alpha_t, 0.0), 1.0), method="higher"))
        q_aci_hist.append(q_aci)
        err = 1.0 if resid > q_aci else 0.0
        aci_hits.append(1.0 if resid <= q_aci else 0.0)
        alpha_t = float(min(max(alpha_t + gamma * (delta - err), 1e-3), 0.5))

    def rolling(hits: List[float], w: int = 500) -> List[float]:
        out, run = [], 0.0
        for i, h in enumerate(hits):
            run += h
            if i >= w:
                run -= hits[i - w]
            out.append(run / min(i + 1, w))
        return out

    target = 1 - delta
    fixed_cov = statistics.fmean(fixed_hits)
    aci_cov = statistics.fmean(aci_hits)
    return {
        "delta": delta,
        "target_coverage": target,
        "gamma": gamma,
        "regime1_n": len(reg1),
        "regime2_n": len(reg2),
        "fixed_quantile_coverage": fixed_cov,
        "aci_coverage": aci_cov,
        "fixed_undercoverage_pp": (target - fixed_cov) * 100.0,
        "aci_dev_pp": abs(target - aci_cov) * 100.0,
        "rolling_fixed": rolling(fixed_hits),
        "rolling_aci": rolling(aci_hits),
    }


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(prog="run_realtrace_replay", description=__doc__)
    parser.add_argument("--out", default="paper/data/realtrace_calibration.json")
    parser.add_argument("--shift-out", default="paper/data/realtrace_shift.json")
    args = parser.parse_args(argv)

    rows, tok_name = _load_or_extract()
    print(f"  extracted {len(rows)} (prompt, completion) pairs via {tok_name}")

    cal = calibration(rows, tok_name)
    cal["turn_depth_quadratic"] = turn_depth_quadratic(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cal, indent=2), encoding="utf-8")

    shift = shift_experiment(rows)
    Path(args.shift_out).write_text(json.dumps(shift, indent=2), encoding="utf-8")

    rd = cal["residuals"]
    print(f"wrote {args.out}")
    print(
        f"  residuals: skew={rd['skew']:.2f} kurtosis={rd['kurtosis_excess']:.2f} "
        f"A-D={rd['anderson_darling_stat']:.1f} (1% crit {rd['anderson_darling_crit_1pct']:.2f}) "
        f"-> rejects normal: {rd['rejects_normal_1pct']}"
    )
    print(
        f"  coverage @δ=0.05: gaussian dev {cal['gaussian_dev_at_005_pp']:.1f} pp, "
        f"conformal dev {cal['conformal_dev_at_005_pp']:.1f} pp "
        f"(max conformal dev {cal['max_conformal_dev_pp']:.1f} pp)"
    )
    q = cal["turn_depth_quadratic"]
    print(f"  §4.3 turn-depth c2={q['fit_c2']:.2f} CI{q['c2_ci95']} positive={q['c2_positive']}")
    print(f"wrote {args.shift_out}")
    print(
        f"  shift: fixed-quantile coverage {shift['fixed_quantile_coverage']*100:.1f}% "
        f"(target {shift['target_coverage']*100:.0f}%), ACI {shift['aci_coverage']*100:.1f}%"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
