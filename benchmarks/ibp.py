"""Synthetic Integrated Business Planning (IBP) benchmark (working paper §8).

A fan-out demand-forecasting pipeline over many SKUs, where each SKU is handled
by a small agent loop.  Two conditions are compared on the *same* synthetic
workload:

- **baseline** — the naive multi-agent pipeline: full **State Snowball** (each
  step re-submits the accreted context, so the per-step prompt grows by ``p``
  tokens per hop), a single high-capability model, no pre-flight estimation, no
  circuit breakers.  Its cumulative prompt cost is ``Theta(depth^2)`` (Theorem 1).
- **treatment** — the same pipeline under Green SARC: :class:`AdapterNode` state
  scoping (caps ``p``), the real predictive Pre-Action Gate + ``LearnedEstimator``
  + ``Budget``, the Action-Time Monitor circuit breaker (caps runaway SKUs), and
  energy-aware routing of "simple" SKUs to a smaller model.

No real LLM is called: token usage is simulated from a known
``completion ~ alpha + beta * prompt`` relationship plus noise, so the estimator
has something real to learn and the numbers are deterministic per seed.  The
treatment path exercises the *real* governance stack end to end.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from green_sarc import (
    Action,
    ActionTimeMonitor,
    AdapterNode,
    Budget,
    CircuitTripped,
    GateDecision,
    GovernanceContext,
    LearnedEstimator,
    ModelProfile,
    PostActionAuditor,
    PreActionGate,
    TableCarbonModel,
    TableCostModel,
    carbon_for_tokens,
)
from green_sarc.stores.memory import MemoryAuditStore

REGION = "eu-west"
BIG_MODEL = "frontier"
SMALL_MODEL = "efficient"


@dataclass
class IBPConfig:
    """Workload parameters (defaults sized for a fast, expressive run)."""

    n_skus: int = 400
    depth: int = 10
    base_prompt: int = 200
    per_step_increment: int = 120  # p — the State-Snowball per-hop growth
    scope_cap: int = 360  # AdapterNode.max_scope_tokens (treatment only)
    max_tokens: int = 4000
    completion_intercept: float = 60.0
    completion_slope: float = 0.45
    completion_noise: float = 25.0
    # Fraction of SKUs that try to loop 3x depth — a stress scenario for the
    # circuit breaker (retry/re-plan storms), not a parameter from the paper.
    runaway_fraction: float = 0.05
    max_loops_factor: float = 1.5  # breaker max_loops = ceil(depth * factor)
    simple_fraction: float = 0.5  # SKUs routed to the small model (treatment)
    delta: float = 0.1


def _models() -> Tuple[TableCostModel, TableCarbonModel, TableCarbonModel]:
    cost_model = TableCostModel(
        profiles={
            BIG_MODEL: ModelProfile(
                energy_per_token_kwh=6.0e-7,
                usd_per_prompt_token=1.0e-5,
                usd_per_completion_token=3.0e-5,
            ),
            SMALL_MODEL: ModelProfile(
                energy_per_token_kwh=1.5e-7,
                usd_per_prompt_token=2.0e-6,
                usd_per_completion_token=6.0e-6,
            ),
        }
    )
    carbon_fixed = TableCarbonModel(intensities={REGION: 350.0})
    # A daily grid-intensity curve (gCO2e/kWh): cleaner overnight, dirtier midday.
    curve = [
        (float(h * 3600), 250.0 + 180.0 * math.sin((h - 6) / 24.0 * 2 * math.pi))
        for h in range(24)
    ]
    carbon_tv = TableCarbonModel(time_series={REGION: curve})
    return cost_model, carbon_fixed, carbon_tv


def _completion(prompt: float, cfg: IBPConfig, rng: random.Random) -> int:
    return max(
        1,
        int(
            cfg.completion_intercept
            + cfg.completion_slope * prompt
            + rng.gauss(0, cfg.completion_noise)
        ),
    )


# Which governance levers are active. Baseline = none; full treatment = all four.
# The ablation isolates each lever's contribution to the saving (audit F-5).
FEATURES_FULL = frozenset({"scope", "routing", "gate", "monitor"})
CONDITIONS = [
    ("baseline", frozenset()),
    ("+scope", frozenset({"scope"})),
    ("+scope+route", frozenset({"scope", "routing"})),
    ("+full", FEATURES_FULL),
]


def run_condition(
    seed: int,
    cfg: IBPConfig,
    features: "frozenset[str]" = FEATURES_FULL,
    collect: Optional[list] = None,
) -> Dict[str, Any]:
    """Run one condition over the workload; return aggregate metrics.

    ``features`` selects which governance levers are active — any subset of
    ``{"scope", "routing", "gate", "monitor"}`` — so an ablation can attribute the
    saving to each lever rather than to "governance" as a black box.

    If ``collect`` is a list, one dict per gated action is appended to it with the
    raw forecast fields ``{prompt, cost_hat, cost_std, actual, source, admitted}``
    — the per-action calibration data the paper's conformal/reliability figures
    consume (the ``AuditRecord`` does not retain ``cost_std``).
    """
    use_scope = "scope" in features
    use_routing = "routing" in features
    use_gate = "gate" in features
    use_monitor = "monitor" in features

    rng = random.Random(seed)
    cost_model, carbon_fixed, carbon_tv = _models()
    adapter = AdapterNode(cfg.scope_cap)

    gate: Optional[PreActionGate] = None
    auditor: Optional[PostActionAuditor] = None
    budget: Optional[Budget] = None
    store: Optional[MemoryAuditStore] = None
    if use_gate:
        estimator = LearnedEstimator(cost_model, carbon_fixed, min_samples=8)
        gate = PreActionGate(estimator)
        store = MemoryAuditStore()
        auditor = PostActionAuditor(store, estimator)
        budget = Budget(token_budget=1.0e15, carbon_ceiling=1.0e18, delta=cfg.delta)

    tokens = usd = carbon_fx = carbon_tvv = 0.0
    rejections = breaker_trips = admitted = 0
    gstep = 0

    for _sku in range(cfg.n_skus):
        runaway = rng.random() < cfg.runaway_fraction
        simple = rng.random() < cfg.simple_fraction
        steps = cfg.depth * 3 if runaway else cfg.depth
        monitor = (
            ActionTimeMonitor(max_loops=math.ceil(cfg.depth * cfg.max_loops_factor))
            if use_monitor
            else None
        )

        for i in range(steps):
            accreted = float(i * cfg.per_step_increment)
            if use_scope:
                accreted = adapter.bound(accreted)  # cap the snowball
            prompt = float(cfg.base_prompt) + accreted
            model = SMALL_MODEL if (use_routing and simple) else BIG_MODEL
            completion = _completion(prompt, cfg, rng)
            actual = prompt + completion
            t = float((gstep % 24) * 3600)
            gstep += 1

            decision: Optional[GateDecision] = None
            if use_gate:
                assert gate is not None and budget is not None
                action = Action(
                    kind="ibp.step",
                    model=model,
                    region=REGION,
                    prompt_tokens=int(prompt),
                    max_tokens=cfg.max_tokens,
                )
                decision = gate.evaluate(action, GovernanceContext(budget=budget, timestamp=t))
                if collect is not None:
                    collect.append(
                        {
                            "prompt": float(prompt),
                            "cost_hat": float(decision.forecast.cost_hat),
                            "cost_std": (
                                None
                                if decision.forecast.cost_std is None
                                else float(decision.forecast.cost_std)
                            ),
                            "actual": float(actual),
                            "source": decision.forecast.source,
                            "admitted": bool(decision.admitted),
                        }
                    )
                if not decision.admitted:
                    rejections += 1
                    continue
            if monitor is not None:
                try:
                    monitor.before()
                except CircuitTripped:
                    breaker_trips += 1
                    break  # the breaker caps a runaway SKU here

            usd_step = cost_model.usd(model, prompt, completion)
            carbon_step = carbon_for_tokens(cost_model, carbon_fixed, model, actual, REGION, t)
            tokens += actual
            usd += usd_step
            carbon_fx += carbon_step
            carbon_tvv += carbon_for_tokens(cost_model, carbon_tv, model, actual, REGION, t)

            if use_gate:
                assert budget is not None and auditor is not None and decision is not None
                budget.spend(actual, carbon_step, usd_step)
                auditor.record(
                    action_id=f"{seed}-{gstep}",
                    action_kind="ibp.step",
                    model=model,
                    region=REGION,
                    forecast=decision.forecast,
                    decision=decision,
                    actual_cost=actual,
                    actual_carbon=carbon_step,
                    budget_remaining_tokens=budget.remaining_tokens(),
                    carbon_remaining=budget.remaining_carbon(),
                    carbon_intensity=carbon_fixed.carbon_intensity(REGION, t),
                    prompt_tokens=int(prompt),
                    actual_usd=usd_step,
                )
                admitted += 1
            if monitor is not None:
                try:
                    monitor.after(actual)
                except CircuitTripped:
                    breaker_trips += 1
                    break

    metrics: Dict[str, Any] = {
        "tokens": tokens,
        "usd": usd,
        "carbon_fixed_g": carbon_fx,
        "carbon_tv_g": carbon_tvv,
        "rejections": rejections,
        "breaker_trips": breaker_trips,
        "admitted": admitted,
    }
    if use_gate and store is not None:
        ran = [r for r in store.list() if r.actual_cost > 0]
        if ran:
            abs_err = sum(abs(r.cost_error) for r in ran)
            metrics["forecast_mae_tokens"] = abs_err / len(ran)
            metrics["forecast_wape"] = abs_err / sum(r.actual_cost for r in ran)
    return metrics


def run_pair(seed: int, cfg: IBPConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(baseline_metrics, full_treatment_metrics)`` for one seed."""
    return run_condition(seed, cfg, frozenset()), run_condition(seed, cfg, FEATURES_FULL)
