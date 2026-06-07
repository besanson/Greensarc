"""Action-Time Monitor (SITE 2) — the circuit breaker.

Tracks the loop count and marginal cost while the agent executes and trips once
a threshold is crossed, killing runaway retry / re-plan loops before they drain
the budget.

The breaker is checked twice per action by the governor: ``before`` an action
(loop-count limit) and ``after`` it (marginal- and cumulative-cost limits).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = [
    "CircuitTripped",
    "ActionTimeMonitor",
]


class CircuitTripped(Exception):
    """Raised when the Action-Time Monitor kills a runaway loop."""

    def __init__(self, reason: str, *, loop_count: int) -> None:
        self.reason = reason
        self.loop_count = loop_count
        super().__init__(f"circuit breaker tripped at loop {loop_count}: {reason}")


@dataclass
class ActionTimeMonitor:
    """Circuit breaker over loop count and marginal / cumulative cost.

    Parameters
    ----------
    max_loops:
        Maximum number of actions in one governed run before tripping.
    max_marginal_cost:
        Optional ceiling on the cost of any single action.
    max_total_cost:
        Optional ceiling on cumulative observed cost across the run.
    """

    max_loops: int = 10
    max_marginal_cost: Optional[float] = None
    max_total_cost: Optional[float] = None
    loop_count: int = 0
    total_cost: float = 0.0
    tripped: bool = False
    reason: str = ""

    def before(self) -> None:
        """Register the start of an action; trip on the loop-count limit."""
        self.loop_count += 1
        if self.loop_count > self.max_loops:
            self.tripped = True
            self.reason = f"loop count {self.loop_count} exceeds max {self.max_loops}"
            raise CircuitTripped(self.reason, loop_count=self.loop_count)

    def after(self, marginal_cost: float) -> None:
        """Register an action's observed cost; trip on a cost limit."""
        self.total_cost += marginal_cost
        if self.max_marginal_cost is not None and marginal_cost > self.max_marginal_cost:
            self.tripped = True
            self.reason = (
                f"marginal cost {marginal_cost:.1f} exceeds max {self.max_marginal_cost:.1f}"
            )
            raise CircuitTripped(self.reason, loop_count=self.loop_count)
        if self.max_total_cost is not None and self.total_cost > self.max_total_cost:
            self.tripped = True
            self.reason = f"total cost {self.total_cost:.1f} exceeds max {self.max_total_cost:.1f}"
            raise CircuitTripped(self.reason, loop_count=self.loop_count)

    def reset(self) -> None:
        """Reset the breaker for a fresh run."""
        self.loop_count = 0
        self.total_cost = 0.0
        self.tripped = False
        self.reason = ""
