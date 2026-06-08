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
from typing import Any, Callable, Dict, List, Optional

__all__ = ["AdapterNode"]


def _default_message_counter(messages: List[Dict[str, Any]]) -> int:
    """Cheap chars/4 token estimate over message text (override with a tokenizer)."""
    total = 0
    for message in messages:
        content = message.get("content", "")
        text = content if isinstance(content, str) else str(content)
        total += max(1, len(text) // 4) + 4
    return total


@dataclass
class AdapterNode:
    """Caps the context an agent carries into its next step.

    ``max_scope_tokens`` is the largest accreted-context size (in tokens) the node
    will pass downstream. Without it, snowballed context at step ``i`` is
    ``base + i * p``; with it, it is ``base + min(i * p, max_scope_tokens)`` —
    bounding ``p``'s contribution and so the prompt cost (working paper §4/§6).

    Use :meth:`bound` on a token count, or :meth:`scope_messages` to truncate a
    real OpenAI/Anthropic-style ``messages`` array.
    """

    max_scope_tokens: int

    def bound(self, upstream_context_tokens: float) -> float:
        """Return the bounded downstream context size (in tokens)."""
        return min(float(upstream_context_tokens), float(self.max_scope_tokens))

    # Backwards-compatible alias for :meth:`bound`.
    def scope(self, upstream_context_tokens: float) -> float:
        return self.bound(upstream_context_tokens)

    def scope_messages(
        self,
        messages: List[Dict[str, Any]],
        *,
        counter: Optional[Callable[[List[Dict[str, Any]]], int]] = None,
    ) -> List[Dict[str, Any]]:
        """Truncate a ``messages`` array to fit ``max_scope_tokens``.

        Keeps the first message (typically the system prompt) and drops the
        oldest of the remainder until the estimated token count is within the cap.
        ``counter`` defaults to a chars/4 estimate; pass a real tokenizer for
        exact counts.
        """
        count = counter or _default_message_counter
        kept = list(messages)
        while len(kept) > 1 and count(kept) > self.max_scope_tokens:
            del kept[1]  # drop the oldest non-system message
        return kept
