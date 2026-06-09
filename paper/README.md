# Paper

The preprint this repository implements:

> Besanson, G. (2026). **Green SARC: Predictive Cost and Carbon Governance for
> Agentic AI Systems.** Preprint, Universidad Torcuato Di Tella.

Source: [`green-sarc.tex`](green-sarc.tex).

## Build

Everything is reproducible from the repository — no number in the paper is
hand-entered. From the repository root:

```bash
make paper-binding-budget  # data/binding_budget_sweep.json (§9, finite-budget sweep)
make paper-realtrace       # data/realtrace_calibration.json + realtrace_shift.json (§10, ShareGPT)
make paper-data            # all of the above + ablation (20 seeds) + learning curve
make paper-figures         # build the 11 figures (figures/*.pdf) + data/figure_stats.json
make paper                 # the above, then compile green-sarc.pdf (needs a local LaTeX toolchain)
python paper/scripts/check_stats.py   # verify every cited statistic resolves to figure_stats.json
```

§10 streams `anon8231489123/ShareGPT_Vicuna_unfiltered` from the Hugging Face
Hub (ungated; token counts only, no LLM calls) and caches the extracted table to
`data/sharegpt_subset.parquet` (git-ignored). LMSYS-Chat-1M is gated and would
not reproduce on a clean clone, so ShareGPT is used as the real-trace source.

If you have no LaTeX locally, push to the repo: the
[`paper`](../.github/workflows/paper.yml) workflow renders the PDF as a CI
artifact. The data assets in [`data/`](data/) and figures in
[`figures/`](figures/) are committed so the PDF builds without re-running the
benchmark.

- **Data:** [`scripts/gen_data.py`](scripts/gen_data.py) →
  [`data/ibp_ablation.json`](data/ibp_ablation.json) (per-seed metrics for every
  ablation condition, paired-bootstrap 95% CIs, and ~16k per-action calibration
  samples). The cold-start curve [`data/learning_curve.json`](data/learning_curve.json)
  comes from `examples/learning_curve/run_demo.py --emit-json`.
- **Figures + stats:** [`scripts/build_figures.py`](scripts/build_figures.py)
  builds `snowball_fit`, `ablation_bars`, `calibration`, `reliability`,
  `delta_sensitivity`, `coldstart`, `penalty_vs_gate`, and writes
  `data/figure_stats.json` — the single source of every quoted value.

## How the paper maps to the code

| Paper construct | Implemented in |
|---|---|
| Augmented state `S' = S ∪ {b_tok, κ(ρ,t), Δ_lat}` (§5.1, Table 1) | [`state.py`](../src/green_sarc/state.py), [`pricing.py`](../src/green_sarc/pricing.py) |
| Predictive estimator `f̂_θ(a,x) = (ĉ, ê)` (§5.2) | [`estimator.py`](../src/green_sarc/estimator.py) |
| Zero-information gate (degenerate case) | `ColdStartEstimator` |
| Predictive Pre-Action Gate, admit at `1 − δ` (§5, §6, Table 2) | [`gate.py`](../src/green_sarc/gate.py) |
| Closed loop `predict → act → log → retrain` (§5.3) | [`governor.py`](../src/green_sarc/governor.py) + `LearnedEstimator.update` |
| Split-conformal calibration & Theorem 2 (§7) | [`build_figures.py`](scripts/build_figures.py) (`fig_reliability`), `gate.py` upper bound |
| Loop circuit breaker — Action-Time Monitor (§6, Table 2) | [`monitor.py`](../src/green_sarc/monitor.py) |
| Per-trajectory cost/carbon log — Post-Action Auditor (§5.3, §6) | [`auditor.py`](../src/green_sarc/auditor.py) |
| Escalation to human / deterministic fallback (§6, Table 2) | [`escalation.py`](../src/green_sarc/escalation.py) |
| Phase 1 per-action; Phase 2 trajectory (§5.4) | Phase 1 implemented; Phase 2 stub in [`trajectory.py`](../src/green_sarc/trajectory.py) |
| State-Snowball `Θ(n²)` theorem & Adapter Nodes (§4, §6) | [`scoping.py`](../src/green_sarc/scoping.py) `AdapterNode` |
| Anytime-valid trajectory bound, Theorem 3 (§7) | [`build_figures.py`](scripts/build_figures.py) (`fig_realtrace_*`), `gate.py` |
| Ablation + CIs, cold start, calibration (§8) | [`benchmarks/`](../benchmarks/), [`scripts/gen_data.py`](scripts/gen_data.py) |
| Binding-budget gate sweep + Pareto (§9) | [`scripts/run_binding_budget.py`](scripts/run_binding_budget.py) |
| Real-trace coverage + distribution shift (§10) | [`scripts/run_realtrace_replay.py`](scripts/run_realtrace_replay.py) |
| Sensitivity to δ (§11); soft-penalty baseline (§12) | [`scripts/build_figures.py`](scripts/build_figures.py) |

## Scope notes (implementation vs. paper)

- This repository implements **Phase 1** (per-action estimation). The Phase-2
  full-trajectory estimator `F̂_θ(π,x)` is an interface stub
  ([`trajectory.py`](../src/green_sarc/trajectory.py)) — by design, since it
  trains only on Phase 1's logged trajectories.
- Consistent with §3, the implementation governs **cost and carbon only**; the
  quality floor `U_min` in the §5.7 optimization is the caller's concern (it
  blocks the degenerate "do nothing" solution) and is not tracked here.
- The latency-headroom field `Δ_lat` is declared in the augmented state (§5.1)
  but **not enforced** in Phase 1 — the paper records this divergence explicitly
  (§11), and so does [`state.py`](../src/green_sarc/state.py).
- Green SARC is **standalone** (the §3 reading): the core has no dependency on
  SARC and composes with it via shared enforcement sites rather than importing
  it. See [`docs/relationship-to-sarc.md`](../docs/relationship-to-sarc.md).
- The synthetic IBP evaluation (§8) is reproducible: see [`benchmarks/`](../benchmarks/)
  (`make reproduce`). The §4/§6 **Adapter Nodes** (state scoping that bounds the
  per-hop increment `p`) are implemented in [`scoping.py`](../src/green_sarc/scoping.py)
  and exercised by the benchmark's treatment condition.
