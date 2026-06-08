# Paper

The working paper this repository implements:

> Besanson, G. (2026). **Green SARC: Predictive FinOps as Governance-by-Architecture
> for Agentic AI Systems.** Working paper, Universidad Torcuato Di Tella.

Source: [`green-sarc.tex`](green-sarc.tex).

## Build

```bash
pdflatex green-sarc.tex
pdflatex green-sarc.tex   # second pass resolves cross-references
```

## How the paper maps to the code

| Paper construct | Implemented in |
|---|---|
| Augmented state `S' = S ∪ {b_tok, κ(ρ,t), Δ_lat}` (§5.1) | [`state.py`](../src/green_sarc/state.py), [`pricing.py`](../src/green_sarc/pricing.py) |
| Predictive estimator `f̂_θ(a,x) = (ĉ, ê)` (§5.2) | [`estimator.py`](../src/green_sarc/estimator.py) |
| Zero-information gate (degenerate case) | `ColdStartEstimator` |
| Predictive Pre-Action Gate, admit at `1 − δ` (§5, Table 1) | [`gate.py`](../src/green_sarc/gate.py) |
| Closed loop `predict → act → log → retrain` (§5.3) | [`governor.py`](../src/green_sarc/governor.py) + `LearnedEstimator.update` |
| Loop circuit breaker — Action-Time Monitor (§5.7, Table 1) | [`monitor.py`](../src/green_sarc/monitor.py) |
| Per-trajectory cost/carbon log — Post-Action Auditor (§5.3, Table 1) | [`auditor.py`](../src/green_sarc/auditor.py) |
| Escalation to human / deterministic fallback (Table 1) | [`escalation.py`](../src/green_sarc/escalation.py) |
| Phase 1 per-action ("Bike"); Phase 2 trajectory ("Car") (§5.4) | Phase 1 implemented; Phase 2 stub in [`trajectory.py`](../src/green_sarc/trajectory.py) |
| State-Snowball `Θ(n²)` theorem & Adapter Nodes (§4, §6) | [`scoping.py`](../src/green_sarc/scoping.py) `AdapterNode` |
| Synthetic IBP evaluation (§8) | [`benchmarks/ibp.py`](../benchmarks/ibp.py), [`benchmarks/reproduce.py`](../benchmarks/reproduce.py) |

## Scope notes (implementation vs. paper)

- This repository implements **Phase 1** (per-action estimation). The Phase-2
  full-trajectory estimator `F̂_θ(π,x)` is an interface stub
  ([`trajectory.py`](../src/green_sarc/trajectory.py)) — by design, since it
  trains only on Phase 1's logged trajectories.
- Consistent with §3, the implementation governs **cost and carbon only**; the
  quality floor `U_min` that appears in the §5.7 optimization is the caller's
  concern (it blocks the degenerate "do nothing" solution) and is not tracked as
  a governed quantity here.
- The paper (§7) frames Green SARC as *extending* the SARC reference
  implementation. This repository realizes the **standalone** reading of §3:
  the core has no dependency on SARC and composes with it via shared enforcement
  sites rather than importing it. See
  [`docs/relationship-to-sarc.md`](../docs/relationship-to-sarc.md).
- The synthetic IBP evaluation (§8) is reproducible: see [`benchmarks/`](../benchmarks/)
  (`make reproduce`). The §4/§6 **Adapter Nodes** (state scoping that bounds the
  per-hop increment `p`) are implemented in [`scoping.py`](../src/green_sarc/scoping.py)
  and exercised by the benchmark's treatment condition.
