"""Tests for the PAIS sidecar adapter (pure gate logic + ASGI middleware)."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from green_sarc.adapters.pais_sidecar import (
    GreenSarcASGIMiddleware,
    SidecarGate,
    count_message_tokens,
    extract_response_text,
    extract_usage_tokens,
    tiktoken_message_counter,
)
from green_sarc.estimator import ColdStartEstimator
from green_sarc.state import Budget


def _sidecar(budget, cost_model, carbon_model) -> SidecarGate:
    return SidecarGate(
        budget=budget,
        estimator=ColdStartEstimator(cost_model, carbon_model),
        cost_model=cost_model,
        carbon_model=carbon_model,
        region="eu-west",
    )


def _chat_request(content: str = "hello there", model: str = "test-model", max_tokens: int = 50):
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }


# -- helper token functions -------------------------------------------------


def test_count_message_tokens_and_usage_helpers():
    msgs = [{"role": "user", "content": "x" * 40}]
    assert count_message_tokens(msgs) == 10 + 4  # 40/4 + per-message overhead

    assert extract_usage_tokens({"usage": {"total_tokens": 123}}) == 123.0
    assert extract_usage_tokens({"usage": {"prompt_tokens": 10, "completion_tokens": 5}}) == 15.0
    assert extract_usage_tokens({"usage": {"total_tokens": 0}}) == 0.0  # PAIS case

    resp = {"choices": [{"message": {"role": "assistant", "content": "answer"}}]}
    assert extract_response_text(resp) == "answer"


# -- pure SidecarGate -------------------------------------------------------


def test_gate_request_admits_and_audits(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)

    decision, action_id = sc.gate_request(_chat_request())
    assert decision.admitted

    rec = sc.audit_response(action_id, {"usage": {"total_tokens": 250}})
    assert rec is not None
    assert rec.actual_cost == 250.0
    assert budget.remaining_tokens() == 10_000.0 - 250.0
    assert len(sc.store.list()) == 1


def test_gate_request_rejects_over_budget(cost_model, carbon_model):
    budget = Budget(token_budget=10.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    decision, _ = sc.gate_request(_chat_request(max_tokens=4000))
    assert not decision.admitted
    assert decision.verdict.value == "reject"


def test_audit_falls_back_to_text_estimate_when_usage_zero(cost_model, carbon_model):
    # PAIS reports usage=0; the sidecar estimates from the response text instead.
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    _, action_id = sc.gate_request(_chat_request(content="hi"))
    resp = {"usage": {"total_tokens": 0}, "choices": [{"message": {"content": "y" * 40}}]}
    rec = sc.audit_response(action_id, resp)
    assert rec is not None
    # prompt estimate (>=1) + 40/4 completion estimate
    assert rec.actual_cost >= 10.0


def test_audit_unknown_action_id_returns_none(cost_model, carbon_model):
    sc = _sidecar(Budget(1000.0, 100.0), cost_model, carbon_model)
    assert sc.audit_response("nope", {"usage": {"total_tokens": 1}}) is None


def test_tiktoken_counter_uses_extra_or_errors_clearly():
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="tiktoken"):
            tiktoken_message_counter("gpt-4")
        return
    counter = tiktoken_message_counter("gpt-4")
    assert counter([{"role": "user", "content": "hello world"}]) > 0


# -- ASGI middleware --------------------------------------------------------


def _make_mock_pais(usage_total: int = 250):
    """A minimal ASGI app standing in for PAIS /v1/chat/completions."""
    state = {"called": False}

    async def app(scope, receive, send):
        state["called"] = True
        # drain the request body
        more = True
        while more:
            msg = await receive()
            more = msg.get("more_body", False) if msg["type"] == "http.request" else False
        payload = json.dumps(
            {
                "id": "x",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"total_tokens": usage_total},
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": payload, "more_body": False})

    return app, state


async def _drive(
    app, body: bytes, *, path="/v1/chat/completions", method="POST", headers=None
) -> List[Dict[str, Any]]:
    scope = {"type": "http", "path": path, "method": method, "headers": headers or []}
    sent = {"done": False}

    async def receive():
        if not sent["done"]:
            sent["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    out: List[Dict[str, Any]] = []

    async def send(message):
        out.append(message)

    await app(scope, receive, send)
    return out


async def test_middleware_forwards_admitted_call_and_audits(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    pais, state = _make_mock_pais(usage_total=250)
    app = GreenSarcASGIMiddleware(pais, sc)

    messages = await _drive(app, json.dumps(_chat_request()).encode())

    assert state["called"] is True
    assert messages[0]["type"] == "http.response.start"
    assert messages[0]["status"] == 200
    # response body forwarded unchanged
    body = json.loads(messages[1]["body"])
    assert body["usage"]["total_tokens"] == 250
    # actuals audited and budget spent
    assert budget.remaining_tokens() == 10_000.0 - 250.0
    assert sc.store.list()[-1].actual_cost == 250.0


async def test_middleware_rejects_with_429_and_does_not_call_app(cost_model, carbon_model):
    budget = Budget(token_budget=10.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    pais, state = _make_mock_pais()
    app = GreenSarcASGIMiddleware(pais, sc)

    messages = await _drive(app, json.dumps(_chat_request(max_tokens=4000)).encode())

    assert state["called"] is False  # the model was never reached
    assert messages[0]["status"] == 429
    error = json.loads(messages[1]["body"])["error"]
    assert error["type"] == "green_sarc_budget_exceeded"
    assert budget.remaining_tokens() == 10.0  # nothing spent


async def test_middleware_passes_through_other_paths(cost_model, carbon_model):
    sc = _sidecar(Budget(10_000.0, 100.0), cost_model, carbon_model)
    pais, state = _make_mock_pais()
    app = GreenSarcASGIMiddleware(pais, sc)

    messages = await _drive(app, b"", path="/health", method="GET")
    assert state["called"] is True
    assert messages[0]["status"] == 200
    assert len(sc.store.list()) == 0  # not gated/audited


# -- streaming (SSE) --------------------------------------------------------


def _make_sse_pais(total_tokens: int = 250):
    """A mock PAIS that streams an OpenAI-style SSE chat completion with usage."""
    state = {"called": False}

    async def app(scope, receive, send):
        state["called"] = True
        more = True
        while more:
            msg = await receive()
            more = msg.get("more_body", False) if msg["type"] == "http.request" else False
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        chunks = [
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
            b'data: {"choices":[],"usage":{"total_tokens":%d}}\n\n' % total_tokens,
            b"data: [DONE]\n\n",
        ]
        for i, c in enumerate(chunks):
            await send({"type": "http.response.body", "body": c, "more_body": i < len(chunks) - 1})

    return app, state


async def test_sse_stream_passthrough_and_usage_audited(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    pais, state = _make_sse_pais(total_tokens=250)
    app = GreenSarcASGIMiddleware(pais, sc)

    messages = await _drive(app, json.dumps(_chat_request()).encode())

    assert state["called"] is True
    assert messages[0]["status"] == 200
    # Streaming preserved: each upstream chunk was forwarded (not buffered into one).
    body_chunks = [m for m in messages if m["type"] == "http.response.body"]
    assert len(body_chunks) >= 3
    # Usage parsed from the SSE stream and audited after it ended.
    rec = sc.store.list()[-1]
    assert rec.actual_cost == 250.0
    assert rec.extra.get("actuals_source") == "sse_usage"
    assert budget.remaining_tokens() == 10_000.0 - 250.0


async def test_non_streaming_overflow_falls_back_to_estimate(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    pais, _ = _make_mock_pais(usage_total=250)
    # A tiny buffer cap forces the non-streaming overflow path.
    app = GreenSarcASGIMiddleware(pais, sc, max_buffer_bytes=8)

    await _drive(app, json.dumps(_chat_request()).encode())
    rec = sc.store.list()[-1]
    assert rec.extra.get("actuals_source") == "estimate_overflow"


async def test_path_regex_matches_trailing_slash(cost_model, carbon_model):
    sc = _sidecar(Budget(10_000.0, 100.0), cost_model, carbon_model)
    pais, state = _make_mock_pais()
    app = GreenSarcASGIMiddleware(pais, sc)
    # Default regex tolerates a trailing slash.
    messages = await _drive(
        app, json.dumps(_chat_request()).encode(), path="/v1/chat/completions/"
    )
    assert state["called"] is True
    assert messages[0]["status"] == 200
    assert len(sc.store.list()) == 1


def test_orphan_gate_releases_reservation(cost_model, carbon_model):
    from green_sarc._ttl_map import TTLMap

    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    # Force the orphan path: a 1-entry pending map evicts the first reservation.
    sc._pending = TTLMap(max_size=1, on_evict=sc._release_pending)

    d1, _ = sc.gate_request(_chat_request())
    assert d1.admitted
    reserved = budget.reserved_tokens
    assert reserved > 0.0

    d2, _ = sc.gate_request(_chat_request())  # evicts the first, releasing its reservation
    assert d2.admitted
    assert budget.reserved_tokens == reserved  # one reservation outstanding, not two


# -- A5: health endpoints, Retry-After, optional bearer auth ----------------


async def test_healthz_bypasses_gating(cost_model, carbon_model):
    sc = _sidecar(Budget(10_000.0, 100.0), cost_model, carbon_model)
    pais, state = _make_mock_pais()
    app = GreenSarcASGIMiddleware(pais, sc)

    messages = await _drive(app, b"", path="/healthz", method="GET")

    assert state["called"] is False  # health handled by the middleware, not the app
    assert messages[0]["status"] == 200
    assert json.loads(messages[1]["body"])["status"] == "ok"


async def test_readyz_503_when_budget_exhausted(cost_model, carbon_model):
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    app = GreenSarcASGIMiddleware(_make_mock_pais()[0], sc)

    ok = await _drive(app, b"", path="/readyz", method="GET")
    assert ok[0]["status"] == 200

    budget.spend(10_000.0, 0.0)  # drain the token budget
    drained = await _drive(app, b"", path="/readyz", method="GET")
    assert drained[0]["status"] == 503


async def test_429_carries_retry_after_and_structured_body(cost_model, carbon_model):
    budget = Budget(token_budget=10.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    app = GreenSarcASGIMiddleware(_make_mock_pais()[0], sc, retry_after_s=7)

    messages = await _drive(app, json.dumps(_chat_request(max_tokens=4000)).encode())

    assert messages[0]["status"] == 429
    headers = {k.lower(): v for k, v in messages[0]["headers"]}
    assert headers[b"retry-after"] == b"7"
    body = json.loads(messages[1]["body"])
    assert set(("reason", "predicted_tokens", "budget_remaining")) <= set(body)
    assert body["budget_remaining"] == 10.0


async def test_auth_401_without_token(monkeypatch, cost_model, carbon_model):
    monkeypatch.setenv("GREEN_SARC_AUTH_TOKEN", "s3cret")
    sc = _sidecar(Budget(10_000.0, 100.0), cost_model, carbon_model)
    pais, state = _make_mock_pais()
    app = GreenSarcASGIMiddleware(pais, sc)

    # No Authorization header -> 401, app never reached, nothing gated.
    messages = await _drive(app, json.dumps(_chat_request()).encode())
    assert messages[0]["status"] == 401
    assert state["called"] is False

    # Wrong token -> 401 too.
    bad = await _drive(
        app,
        json.dumps(_chat_request()).encode(),
        headers=[(b"authorization", b"Bearer wrong")],
    )
    assert bad[0]["status"] == 401


async def test_auth_200_with_valid_bearer(monkeypatch, cost_model, carbon_model):
    monkeypatch.setenv("GREEN_SARC_AUTH_TOKEN", "s3cret")
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    pais, state = _make_mock_pais(usage_total=250)
    app = GreenSarcASGIMiddleware(pais, sc)

    messages = await _drive(
        app,
        json.dumps(_chat_request()).encode(),
        headers=[(b"authorization", b"Bearer s3cret")],
    )
    assert messages[0]["status"] == 200
    assert state["called"] is True
    assert budget.remaining_tokens() == 10_000.0 - 250.0


async def test_unauthenticated_when_token_unset(monkeypatch, cost_model, carbon_model):
    monkeypatch.delenv("GREEN_SARC_AUTH_TOKEN", raising=False)
    budget = Budget(token_budget=10_000.0, carbon_ceiling=100.0)
    sc = _sidecar(budget, cost_model, carbon_model)
    app = GreenSarcASGIMiddleware(_make_mock_pais()[0], sc)  # warns once, stays open

    messages = await _drive(app, json.dumps(_chat_request()).encode())
    assert messages[0]["status"] == 200  # unchanged behaviour
