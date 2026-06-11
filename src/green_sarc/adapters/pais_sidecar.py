"""KAOS adapter — Green SARC as a PAIS sidecar / middleware.

KAOS agents run on PAIS (Pydantic AI Server), a FastAPI app exposing an
OpenAI-compatible ``POST /v1/chat/completions`` endpoint.  Where the MCP adapter
is *advisory* (an agent chooses to call the gate tool), this adapter is **hard**:
it sits as ASGI middleware in front of that endpoint so **every** model call is
gated.  A rejected call is answered with HTTP ``429`` and never reaches the
model; an admitted call is forwarded and its actual token usage is read back from
the response to close the audit loop.

The dependency still runs one way (KAOS → Green SARC) and the core is untouched:
:class:`SidecarGate` is the pure, framework-free logic (testable with plain
dicts), and :class:`GreenSarcASGIMiddleware` is a dependency-free ASGI wrapper
around it — no FastAPI/Starlette/httpx import required.

Deployment: run this middleware in a container in the agent pod (a sidecar) and
point the agent's model traffic at it, or wrap the PAIS ``app`` directly with
``app.add_middleware``.

.. note::
   PAIS currently hardcodes ``usage`` to zero in its responses, so when the
   response carries no usable token count this adapter falls back to a
   length-based estimate of the response text.  Prefer wiring real usage from
   the model response or the OpenTelemetry span (see :mod:`green_sarc.adapters.otel`).
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from green_sarc._ttl_map import TTLMap
from green_sarc.auditor import AuditRecord, PostActionAuditor
from green_sarc.estimator import Estimator
from green_sarc.forecast import GateDecision, Verdict
from green_sarc.gate import PreActionGate
from green_sarc.pricing import CarbonModel, CostModel, carbon_for_tokens
from green_sarc.state import Action, Budget, GovernanceContext
from green_sarc.stores.base import AuditStore
from green_sarc.stores.memory import MemoryAuditStore

logger = logging.getLogger(__name__)

__all__ = [
    "estimate_text_tokens",
    "count_message_tokens",
    "tiktoken_message_counter",
    "extract_usage_tokens",
    "extract_response_text",
    "SidecarGate",
    "GreenSarcASGIMiddleware",
]

# Rough chars-per-token ratio for the heuristic token counter used when no
# tokenizer and no reported usage are available.
_CHARS_PER_TOKEN = 4


def estimate_text_tokens(text: str) -> int:
    """Cheap length-based token estimate (``~4`` chars per token)."""
    if not text:
        return 0
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    # Vision / multi-part content: concatenate any text parts.
    if isinstance(content, list):
        parts = [p.get("text", "") for p in content if isinstance(p, dict)]
        return " ".join(parts)
    return str(content)


def count_message_tokens(messages: List[Dict[str, Any]]) -> int:
    """Estimate prompt tokens for an OpenAI-style ``messages`` array."""
    total = 0
    for msg in messages:
        total += estimate_text_tokens(_message_text(msg)) + 4  # small per-message overhead
    return total


def tiktoken_message_counter(model: str) -> Callable[[List[Dict[str, Any]]], int]:
    """Return an accurate prompt-token counter backed by ``tiktoken``.

    Pass the result as ``SidecarGate(prompt_token_counter=...)`` for exact counts
    instead of the chars/4 heuristic.  Requires the optional extra
    (``pip install 'green-sarc[tiktoken]'``).
    """
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "tiktoken_message_counter requires the 'tiktoken' extra; install green-sarc[tiktoken]"
        ) from exc
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    def count(messages: List[Dict[str, Any]]) -> int:
        return sum(len(encoding.encode(_message_text(m))) + 4 for m in messages)

    return count


def extract_usage_tokens(response: Dict[str, Any]) -> float:
    """Read total tokens from an OpenAI-style ``usage`` object (0 if absent)."""
    usage = response.get("usage") or {}
    total = usage.get("total_tokens")
    if total:
        return float(total)
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    if prompt or completion:
        return float(prompt + completion)
    return 0.0


def extract_response_text(response: Dict[str, Any]) -> str:
    """Concatenate assistant message text from a chat-completion response."""
    texts: List[str] = []
    for choice in response.get("choices", []) or []:
        message = choice.get("message") or {}
        texts.append(_message_text(message))
    return " ".join(t for t in texts if t)


@dataclass
class SidecarGate:
    """Pure gate + auditor logic for the PAIS ``/v1/chat/completions`` path.

    Translates an OpenAI-style chat request into an :class:`Action`, gates it,
    and — on the way back — reads actual token usage from the response to write
    the audit record and decrement the budget.
    """

    budget: Budget
    estimator: Estimator
    cost_model: CostModel
    carbon_model: CarbonModel
    region: str = ""
    store: AuditStore = field(default_factory=MemoryAuditStore)
    prompt_token_counter: Callable[[List[Dict[str, Any]]], int] = count_message_tokens
    gate: PreActionGate = field(init=False)
    auditor: PostActionAuditor = field(init=False)
    _pending: TTLMap[str, Tuple[Action, GateDecision, float, float]] = field(
        default_factory=TTLMap
    )

    def __post_init__(self) -> None:
        self.gate = PreActionGate(self.estimator)
        self.auditor = PostActionAuditor(self.store, self.estimator)
        self._pending = TTLMap(on_evict=self._release_pending)

    def _release_pending(
        self, _key: str, value: Tuple[Action, GateDecision, float, float]
    ) -> None:
        """Return the budget reservation an orphaned (un-audited) gate held."""
        _action, _decision, reserve_tokens, reserve_carbon = value
        self.budget.release(reserve_tokens, reserve_carbon)

    def gate_request(self, body: Dict[str, Any]) -> Tuple[GateDecision, str]:
        """Gate an OpenAI-style chat-completion request body.

        Returns the :class:`GateDecision` and a correlation ``action_id`` to pass
        back to :meth:`audit_response`.
        """
        messages = body.get("messages", []) or []
        action = Action(
            kind="chat.completion",
            model=str(body.get("model", "")),
            region=self.region,
            prompt_tokens=self.prompt_token_counter(messages),
            max_tokens=body.get("max_tokens"),
        )
        decision = self.gate.evaluate(action, GovernanceContext(budget=self.budget))
        action_id = uuid.uuid4().hex
        if decision.admitted:
            reserve_tokens = self.gate.cost_upper_bound(decision.forecast, self.budget.delta)
            reserve_carbon = decision.forecast.carbon_hat
            if self.budget.reserve(reserve_tokens, reserve_carbon):
                self._pending.put(action_id, (action, decision, reserve_tokens, reserve_carbon))
            else:
                # The worst-case reservation would exceed the remaining budget
                # (or it was raced out): fail closed — the middleware 429s, and we
                # never fall through to an unguarded spend.
                decision = GateDecision(
                    verdict=Verdict.REJECT,
                    forecast=decision.forecast,
                    reason="reservation would exceed remaining budget",
                )
        return decision, action_id

    def audit_response(
        self,
        action_id: str,
        response: Dict[str, Any],
        *,
        actual_tokens: Optional[float] = None,
        actuals_source: str = "usage",
    ) -> Optional[AuditRecord]:
        """Record actuals for a previously gated request; spend the budget.

        ``actual_tokens`` overrides the value read from the response; when it is
        ``None`` the reported ``usage`` is used, falling back to a length-based
        estimate of the response text (PAIS reports zero usage today).
        """
        pending = self._pending.pop(action_id, None)
        if pending is None:
            return None
        action, decision, reserve_tokens, reserve_carbon = pending

        if actual_tokens is None:
            actual_tokens = extract_usage_tokens(response)
            if actual_tokens <= 0.0:
                # PAIS hardcodes usage=0: fall back to a text-length estimate.
                prompt = float(action.prompt_tokens or 0)
                actual_tokens = prompt + estimate_text_tokens(extract_response_text(response))

        carbon = carbon_for_tokens(
            self.cost_model, self.carbon_model, action.model, float(actual_tokens), action.region
        )
        prompt = float(action.prompt_tokens or 0)
        actual_usd = self.cost_model.usd(
            action.model, prompt, max(0.0, float(actual_tokens) - prompt)
        )
        self.budget.commit(
            reserve_tokens, reserve_carbon, float(actual_tokens), carbon, actual_usd
        )
        return self.auditor.record(
            action_id=action_id,
            action_kind=action.kind,
            model=action.model,
            region=action.region,
            forecast=decision.forecast,
            decision=decision,
            actual_cost=float(actual_tokens),
            actual_carbon=carbon,
            budget_remaining_tokens=self.budget.remaining_tokens(),
            carbon_remaining=self.budget.remaining_carbon(),
            carbon_intensity=self.carbon_model.carbon_intensity(action.region),
            prompt_tokens=action.prompt_tokens or 0,
            actual_usd=actual_usd,
            extra={"actuals_source": actuals_source},
        )


# -- ASGI types (kept local so the adapter needs no web framework) ----------
Scope = Dict[str, Any]
Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


def _scan_sse_usage(data: bytes) -> Tuple[float, bytes, bool]:
    """Scan complete SSE lines for the latest ``usage.total_tokens``.

    Returns ``(max_usage_seen, trailing_partial_line, saw_done)``.  OpenAI-style
    streaming with ``stream_options.include_usage`` emits a final ``data:`` event
    whose ``usage`` carries the cumulative total, so we take the maximum usage
    seen rather than summing.
    """
    usage = 0.0
    seen_done = False
    *lines, residual = data.split(b"\n")
    for raw_line in lines:
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:") :].strip()
        if payload == b"[DONE]":
            seen_done = True
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            usage = max(usage, extract_usage_tokens(obj))
    return usage, residual, seen_done


class GreenSarcASGIMiddleware:
    """Dependency-free ASGI middleware that hard-gates one HTTP path.

    Wrap a PAIS ASGI ``app`` so every ``POST`` to ``path`` is gated: rejected
    calls get ``429`` and never reach the app; admitted calls are forwarded and
    their response is read to write the audit record.

    Responses are streamed through unchanged (each chunk is forwarded
    immediately), so Server-Sent-Event (``text/event-stream``) chat completions
    are not broken or buffered: token usage is parsed from the SSE ``data:``
    events on the fly and the audit is written once the stream ends (audit P0-6).
    Non-streaming JSON bodies are mirrored into a bounded buffer
    (``max_buffer_bytes``) for usage parsing; past the cap the audit falls back to
    an estimate tagged ``actuals_source="estimate_overflow"``.
    """

    def __init__(
        self,
        app: ASGIApp,
        sidecar: SidecarGate,
        *,
        path: str = "/v1/chat/completions",
        path_regex: Optional[str] = None,
        max_buffer_bytes: int = 8 * 1024 * 1024,
        stream_passthrough: bool = True,
        health_path: str = "/healthz",
        ready_path: str = "/readyz",
        retry_after_s: int = 5,
        auth_token_env: str = "GREEN_SARC_AUTH_TOKEN",
    ) -> None:
        self.app = app
        self.sidecar = sidecar
        self.path = path
        # Match the gated path by regex so trailing slashes, query strings, and
        # other OpenAI-compatible servers can be guarded (env override available).
        pattern = (
            path_regex or os.environ.get("GREEN_SARC_PATH_REGEX") or rf"^{re.escape(path)}/?$"
        )
        self.path_re = re.compile(pattern)
        self.max_buffer_bytes = max_buffer_bytes
        self.stream_passthrough = stream_passthrough
        self.health_path = health_path
        self.ready_path = ready_path
        self.retry_after_s = retry_after_s
        # Optional shared-secret auth: when GREEN_SARC_AUTH_TOKEN is set, governed
        # paths require `Authorization: Bearer <token>`; when unset, the sidecar is
        # open (unchanged behaviour) and warns once so the gap is visible in logs.
        self.auth_token = os.environ.get(auth_token_env) or None
        if self.auth_token is None:
            logger.warning(
                "Green SARC sidecar is UNAUTHENTICATED: set %s to require a bearer "
                "token on governed paths (budget-drain / auditor-spoofing surface).",
                auth_token_env,
            )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") == "http":
            req_path = scope.get("path", "")
            # Health probes bypass gating and auth (liveness/readiness for k8s).
            if req_path == self.health_path:
                await self._health(send, ready=True)
                return
            if req_path == self.ready_path:
                exhausted = (
                    self.sidecar.budget.is_token_exhausted()
                    or self.sidecar.budget.is_carbon_exhausted()
                    or self.sidecar.budget.is_usd_exhausted()
                )
                await self._health(send, ready=not exhausted)
                return

        if (
            scope.get("type") != "http"
            or not self.path_re.match(scope.get("path", ""))
            or scope.get("method", "").upper() != "POST"
        ):
            await self.app(scope, receive, send)
            return

        # Optional shared-secret check on governed paths (fail closed with 401).
        if self.auth_token is not None and not self._authorized(scope):
            await self._unauthorized(send)
            return

        body = await self._read_body(receive)
        try:
            request = json.loads(body) if body else {}
        except json.JSONDecodeError:
            # Not JSON we understand — forward ungated rather than break the call.
            await self.app(scope, self._replay(body), send)
            return

        decision, action_id = self.sidecar.gate_request(request)
        if not decision.admitted:
            await self._reject(send, decision)
            return

        state: Dict[str, Any] = {
            "streaming": False,
            "buf": bytearray(),
            "overflow": False,
            "sse_usage": 0.0,
            "residual": b"",
            "done": False,
        }

        async def capture_send(message: Message) -> None:
            mtype = message["type"]
            if mtype == "http.response.start":
                headers = {k.lower(): v for k, v in message.get("headers", [])}
                content_type = headers.get(b"content-type", b"").decode().lower()
                transfer = headers.get(b"transfer-encoding", b"").decode().lower()
                state["streaming"] = self.stream_passthrough and (
                    content_type.startswith("text/event-stream") or "chunked" in transfer
                )
                await send(message)  # forward headers immediately
            elif mtype == "http.response.body":
                await send(message)  # passthrough: forward each chunk immediately
                chunk = message.get("body", b"")
                if state["streaming"]:
                    usage, state["residual"], _done = _scan_sse_usage(state["residual"] + chunk)
                    state["sse_usage"] = max(state["sse_usage"], usage)
                elif not state["overflow"]:
                    if len(state["buf"]) + len(chunk) <= self.max_buffer_bytes:
                        state["buf"].extend(chunk)
                    else:
                        state["overflow"] = True
                if not message.get("more_body", False) and not state["done"]:
                    state["done"] = True
                    self._audit(action_id, state)

        await self.app(scope, self._replay(body), capture_send)
        if not state["done"]:  # app sent no terminal body chunk; audit best-effort
            self._audit(action_id, state)

    def _audit(self, action_id: str, state: Dict[str, Any]) -> None:
        if state["streaming"]:
            if state["sse_usage"] > 0.0:
                self.sidecar.audit_response(
                    action_id, {}, actual_tokens=state["sse_usage"], actuals_source="sse_usage"
                )
            else:
                self.sidecar.audit_response(action_id, {}, actuals_source="estimate_stream")
            return
        if state["overflow"]:
            self.sidecar.audit_response(action_id, {}, actuals_source="estimate_overflow")
            return
        raw = bytes(state["buf"])
        try:
            response = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            response = {}
        source = "usage" if extract_usage_tokens(response) > 0.0 else "estimate_text"
        self.sidecar.audit_response(action_id, response, actuals_source=source)

    def _authorized(self, scope: Scope) -> bool:
        """True when the request carries the configured `Authorization: Bearer`."""
        for k, v in scope.get("headers", []):
            if k.lower() == b"authorization":
                value = v.decode("latin-1").strip()
                if value.lower().startswith("bearer "):
                    return value[7:].strip() == self.auth_token
                return False
        return False

    async def _send_json(self, send: Send, status: int, payload: dict, extra_headers=()) -> None:
        body = json.dumps(payload).encode()
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ]
        headers.extend(extra_headers)
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body, "more_body": False})

    async def _health(self, send: Send, *, ready: bool) -> None:
        await self._send_json(
            send, 200 if ready else 503, {"status": "ok" if ready else "budget_exhausted"}
        )

    async def _unauthorized(self, send: Send) -> None:
        await self._send_json(
            send,
            401,
            {"error": {"message": "Green SARC: missing or invalid bearer token",
                       "type": "green_sarc_unauthorized"}},
            extra_headers=[(b"www-authenticate", b"Bearer")],
        )

    async def _reject(self, send: Send, decision: GateDecision) -> None:
        await self._send_json(
            send,
            429,
            {
                "error": {
                    "message": f"Green SARC: {decision.reason}",
                    "type": "green_sarc_budget_exceeded",
                    "verdict": decision.verdict.value,
                },
                "reason": decision.reason,
                "predicted_tokens": decision.forecast.cost_hat,
                "budget_remaining": self.sidecar.budget.remaining_tokens(),
            },
            extra_headers=[(b"retry-after", str(self.retry_after_s).encode())],
        )

    @staticmethod
    async def _read_body(receive: Receive) -> bytes:
        chunks = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] == "http.request":
                chunks.extend(message.get("body", b""))
                more = message.get("more_body", False)
            elif message["type"] == "http.disconnect":
                break
        return bytes(chunks)

    @staticmethod
    def _replay(body: bytes) -> Receive:
        sent = False

        async def receive() -> Message:
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.disconnect"}

        return receive
