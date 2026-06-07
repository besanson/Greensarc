"""Pre-Action Gate (SITE 1).

Runs the predictive estimator on a proposed action and admits it only if:

- the forecast token cost fits the remaining budget at confidence ``1 - delta``:
  ``P[cost_hat <= b_tok] >= 1 - delta``; and
- the forecast carbon fits the remaining ceiling:
  ``carbon_hat <= B_co2 - carbon_spent``.

Otherwise the gate rejects, down-routes, or — when the budget/carbon is already
exhausted — escalates.

The confidence test is implemented as a one-sided upper bound: when the
estimator supplies a standard deviation the gate forms the ``1 - delta``
quantile of a normal forecast; when it does not (the cold-start case) the point
estimate is treated as the worst case, which is conservative by construction.
"""

from __future__ import annotations

from statistics import NormalDist

from green_sarc.estimator import Estimator
from green_sarc.forecast import Forecast, GateDecision, Verdict
from green_sarc.state import Action, GovernanceContext

__all__ = [
    "PreActionGate",
]


class PreActionGate:
    """Admits or rejects a proposed action on a learned cost/carbon forecast."""

    def __init__(self, estimator: Estimator) -> None:
        self.estimator = estimator

    def cost_upper_bound(self, forecast: Forecast, delta: float) -> float:
        """Upper bound on token cost at confidence ``1 - delta``.

        With a standard deviation this is the ``1 - delta`` quantile of a normal
        forecast; without one the point estimate is the worst case.
        """
        if forecast.cost_std and forecast.cost_std > 0.0:
            z = NormalDist().inv_cdf(1.0 - delta)
            return forecast.cost_hat + z * forecast.cost_std
        return forecast.cost_hat

    def evaluate(self, action: Action, ctx: GovernanceContext) -> GateDecision:
        """Forecast ``action`` and return the admission :class:`GateDecision`."""
        budget = ctx.budget

        # Already out of budget / carbon / USD: nothing to gate — escalate.
        if (
            budget.is_token_exhausted()
            or budget.is_carbon_exhausted()
            or budget.is_usd_exhausted()
        ):
            forecast = self.estimator.predict(action, ctx)
            return GateDecision(
                verdict=Verdict.ESCALATE,
                forecast=forecast,
                reason="budget, carbon, or USD ceiling already exhausted",
            )

        forecast = self.estimator.predict(action, ctx)
        cost_bound = self.cost_upper_bound(forecast, budget.delta)
        token_ok = cost_bound <= budget.remaining_tokens()
        carbon_ok = forecast.carbon_hat <= budget.remaining_carbon()
        usd_ok = forecast.usd_hat <= budget.remaining_usd()

        if token_ok and carbon_ok and usd_ok:
            return GateDecision(verdict=Verdict.ADMIT, forecast=forecast, reason="within budget")

        reasons = []
        if not token_ok:
            reasons.append(
                f"forecast cost {cost_bound:.1f} exceeds remaining "
                f"{budget.remaining_tokens():.1f} tokens at {1 - budget.delta:.0%} confidence"
            )
        if not carbon_ok:
            reasons.append(
                f"forecast carbon {forecast.carbon_hat:.3f} exceeds remaining "
                f"{budget.remaining_carbon():.3f} gCO2e"
            )
        if not usd_ok:
            reasons.append(
                f"forecast cost ${forecast.usd_hat:.4f} exceeds remaining "
                f"${budget.remaining_usd():.4f}"
            )
        return GateDecision(verdict=Verdict.REJECT, forecast=forecast, reason="; ".join(reasons))
