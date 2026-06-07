"""Tests for the Escalation Router (SITE 4)."""

from __future__ import annotations

from green_sarc.escalation import (
    DeterministicFallbackHandler,
    EscalationEvent,
    EscalationReason,
    EscalationRouter,
)


def _event(reason=EscalationReason.TOKEN_EXHAUSTED) -> EscalationEvent:
    return EscalationEvent(reason=reason, action_id="a1", action_kind="chat.completion")


async def test_router_dispatches_to_handler():
    seen = []

    async def handler(event: EscalationEvent) -> None:
        seen.append(event)

    router = EscalationRouter(handler)
    await router.route(_event())
    assert len(seen) == 1
    assert seen[0].reason is EscalationReason.TOKEN_EXHAUSTED


async def test_router_suppresses_handler_errors():
    async def boom(event: EscalationEvent) -> None:
        raise RuntimeError("handler failed")

    router = EscalationRouter(boom)
    # Must not propagate — a broken handler can never break the agent loop.
    await router.route(_event())


async def test_deterministic_fallback_handler():
    def fallback(event: EscalationEvent) -> str:
        return f"cached-answer-for-{event.action_id}"

    event = _event(EscalationReason.CARBON_EXHAUSTED)
    router = EscalationRouter(DeterministicFallbackHandler(fallback))
    await router.route(event)
    assert event.metadata["fallback_result"] == "cached-answer-for-a1"


async def test_default_handler_is_log_only():
    router = EscalationRouter()
    await router.route(_event())  # logs; must not raise


def test_event_to_dict():
    d = _event().to_dict()
    assert d["reason"] == "token_exhausted"
    assert d["action_kind"] == "chat.completion"
