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

__all__ = [
    "estimate_text_tokens",
    "count_message_tokens",
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
                # Raced out by a concurrent reservation: reject (the middleware 429s).
                decision = GateDecision(
                    verdict=Verdict.REJECT,
                    forecast=decision.forecast,
                    reason="insufficient budget after concurrent reservations",
                )
        return decision, action_id

    def audit_response(
        self,
        action_id: str,
        response: Dict[str, Any],
        *,
        actual_tokens: Optional[float] = None,
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
        self.budget.commit(reserve_tokens, reserve_carbon, float(actual_tokens), carbon)
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
        )


# -- ASGI types (kept local so the adapter needs no web framework) ----------
Scope = Dict[str, Any]
Message = Dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class GreenSarcASGIMiddleware:
    """Dependency-free ASGI middleware that hard-gates one HTTP path.

    Wrap a PAIS ASGI ``app`` so every ``POST`` to ``path`` is gated: rejected
    calls get ``429`` and never reach the app; admitted calls are forwarded and
    their response is read to write the audit record.
    """

    def __init__(
        self,
        app: ASGIApp,
        sidecar: SidecarGate,
        *,
        path: str = "/v1/chat/completions",
    ) -> None:
        self.app = app
        self.sidecar = sidecar
        self.path = path

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("path") != self.path
            or scope.get("method", "").upper() != "POST"
        ):
            await self.app(scope, receive, send)
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

        captured: Dict[str, Any] = {"start": None, "body": bytearray()}

        async def capture_send(message: Message) -> None:
            mtype = message["type"]
            if mtype == "http.response.start":
                captured["start"] = message
            elif mtype == "http.response.body":
                captured["body"].extend(message.get("body", b""))
                if message.get("more_body"):
                    return
                await self._flush_and_audit(send, captured, action_id)

        await self.app(scope, self._replay(body), capture_send)

    async def _flush_and_audit(self, send: Send, captured: Dict[str, Any], action_id: str) -> None:
        raw = bytes(captured["body"])
        try:
            response = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            response = {}
        self.sidecar.audit_response(action_id, response)
        if captured["start"] is not None:
            await send(captured["start"])
        await send({"type": "http.response.body", "body": raw, "more_body": False})

    async def _reject(self, send: Send, decision: GateDecision) -> None:
        payload = json.dumps(
            {
                "error": {
                    "message": f"Green SARC: {decision.reason}",
                    "type": "green_sarc_budget_exceeded",
                    "verdict": decision.verdict.value,
                }
            }
        ).encode()
        await send(
            {
                "type": "http.response.start",
                "status": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": payload, "more_body": False})

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
