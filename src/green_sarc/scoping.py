"""Adapter Node — bounded state scoping (working paper §4 / §6).

The **State Snowball** is the failure mode where a multi-agent loop re-submits
the full accreted context at every step, so the per-step prompt grows by ``p``
tokens per hop and the cumulative prompt cost is ``Theta(n^2)`` in loop depth
(Theorem 1).

An :class:`AdapterNode` projects an agent's upstream context down to a minimal
downstream scope, **capping the per-hop increment ``p``** and collapsing the
quadratic term toward linear.  It is deliberately tiny and framework-agnostic:
it operates on a token count (or any additive context measure), so it composes
with the four enforcement sites without knowing anything about message formats.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AdapterNode"]


@dataclass
class AdapterNode:
    """Caps the context an agent carries into its next step.

    ``max_scope_tokens`` is the largest accreted-context size (in tokens) the
    node will pass downstream.  Without it, snowballed context at step ``i`` is
    ``base + i * p``; with it, it is ``base + min(i * p, max_scope_tokens)`` —
    bounding ``p``'s contribution and so the prompt cost.
    """

    max_scope_tokens: int

    def scope(self, upstream_context_tokens: float) -> float:
        """Return the bounded downstream context size."""
        return min(float(upstream_context_tokens), float(self.max_scope_tokens))
