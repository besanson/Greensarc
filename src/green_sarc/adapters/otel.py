"""KAOS adapter — consume an OpenTelemetry actuals stream.

KAOS/PAIS emit per-request OpenTelemetry spans; the actual token usage of a
model call lands there (pydantic-ai's instrumentation records ``gen_ai.usage.*``
attributes).  The Post-Action Auditor's predicted-vs-actual log can therefore be
fed from the OTel stream instead of an explicit auditor tool call.

This module provides the framework-agnostic mapping from a span to actuals
(:class:`OTelActualsConsumer.ingest_span`) — testable with a plain dict and no
``opentelemetry`` dependency.  The live OTLP receiver that would push spans into
it is a documented stub (:meth:`OTelActualsConsumer.serve_otlp`); wiring it is
deferred until the integration surface is exercised end-to-end.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

__all__ = [
    "SpanActuals",
    "OTelActualsConsumer",
]

# Span attribute keys pydantic-ai / OpenAI-compatible instrumentation use.
_INPUT_TOKEN_KEYS = ("gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens")
_OUTPUT_TOKEN_KEYS = ("gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens")
_TOTAL_TOKEN_KEYS = ("gen_ai.usage.total_tokens",)


class SpanActuals:
    """Actual token usage extracted from a single span."""

    def __init__(self, action_id: str, actual_tokens: float, model: str = "", region: str = ""):
        self.action_id = action_id
        self.actual_tokens = actual_tokens
        self.model = model
        self.region = region

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "actual_tokens": self.actual_tokens,
            "model": self.model,
            "region": self.region,
        }


def _first(attrs: Dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for k in keys:
        if k in attrs and attrs[k] is not None:
            return float(attrs[k])
    return None


class OTelActualsConsumer:
    """Map OpenTelemetry spans to actual token usage for the auditor.

    Parameters
    ----------
    on_actuals:
        Callback invoked with each :class:`SpanActuals` extracted from a span —
        typically wired to :meth:`GreenSarcMCPService.post_action_auditor` or a
        direct auditor call.  The correlation id is read from the span attribute
        named by ``action_id_key`` (default ``"green_sarc.action_id"``), which a
        gate call would have set as OpenTelemetry baggage.
    """

    def __init__(
        self,
        on_actuals: Callable[[SpanActuals], None],
        *,
        action_id_key: str = "green_sarc.action_id",
    ) -> None:
        self.on_actuals = on_actuals
        self.action_id_key = action_id_key

    def ingest_span(self, span: Dict[str, Any]) -> Optional[SpanActuals]:
        """Extract actuals from a span dict and emit them; ``None`` if absent.

        ``span`` is the framework-neutral shape ``{"name": str, "attributes":
        {...}}`` — the same data an OTLP exporter carries.
        """
        attrs: Dict[str, Any] = span.get("attributes", {})
        action_id = attrs.get(self.action_id_key)
        if action_id is None:
            return None

        total = _first(attrs, _TOTAL_TOKEN_KEYS)
        if total is None:
            inp = _first(attrs, _INPUT_TOKEN_KEYS) or 0.0
            out = _first(attrs, _OUTPUT_TOKEN_KEYS) or 0.0
            total = inp + out
        if total <= 0.0:
            return None

        actuals = SpanActuals(
            action_id=str(action_id),
            actual_tokens=total,
            model=str(attrs.get("gen_ai.request.model", "")),
            region=str(attrs.get("green_sarc.region", "")),
        )
        self.on_actuals(actuals)
        return actuals

    def serve_otlp(self, endpoint: str) -> None:  # pragma: no cover - documented stub
        """Run an OTLP receiver that feeds :meth:`ingest_span` (not yet implemented).

        The mapping above is the stable part; standing up an OTLP collector
        endpoint is deferred until the KAOS OTel path is exercised end-to-end.
        """
        raise NotImplementedError(
            "Live OTLP ingestion is a documented stub; use ingest_span() to feed "
            "spans from your own collector for now."
        )
