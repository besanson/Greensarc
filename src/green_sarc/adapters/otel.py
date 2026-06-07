"""KAOS adapter — consume an OpenTelemetry actuals stream.

KAOS/PAIS emit per-request OpenTelemetry spans; the actual token usage of a
model call lands there (pydantic-ai's instrumentation records ``gen_ai.usage.*``
attributes).  The Post-Action Auditor's predicted-vs-actual log can therefore be
fed from the OTel stream instead of an explicit auditor tool call.

This module provides:

- the framework-agnostic mapping from a span to actuals
  (:class:`OTelActualsConsumer.ingest_span`) — testable with a plain dict and no
  ``opentelemetry`` dependency; and
- :class:`GreenSarcSpanProcessor`, a working OpenTelemetry ``SpanProcessor`` that
  feeds ended spans into the consumer **in process**.  It is duck-typed against
  the ``SpanProcessor`` interface, so it needs no ``opentelemetry`` import itself
  and is unit-testable, yet registers directly on a real ``TracerProvider``::

      provider.add_span_processor(GreenSarcSpanProcessor(consumer))

This is the supported live path when Green SARC runs in the same process as PAIS
(for example alongside the sidecar).  Consuming spans **across** processes from a
standalone OTLP collector is a separate deployment concern; the receiver hook
(:meth:`OTelActualsConsumer.serve_otlp`) is left as a documented stub for that.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "SpanActuals",
    "OTelActualsConsumer",
    "span_to_dict",
    "GreenSarcSpanProcessor",
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
        """Run a standalone OTLP receiver that feeds :meth:`ingest_span`.

        For **in-process** consumption use :class:`GreenSarcSpanProcessor` instead
        — it is the supported live path.  A cross-process OTLP collector endpoint
        is a deployment concern left as a documented stub here.
        """
        raise NotImplementedError(
            "Cross-process OTLP ingestion is a documented stub; register a "
            "GreenSarcSpanProcessor on your TracerProvider for in-process spans, "
            "or feed ingest_span() from your own collector."
        )


def span_to_dict(span: Any) -> Dict[str, Any]:
    """Convert an OpenTelemetry ``ReadableSpan`` to the neutral span dict.

    Reads ``span.name`` and ``span.attributes`` (a mapping), which is all
    :meth:`OTelActualsConsumer.ingest_span` needs.  Works on the real SDK span and
    on any object exposing those attributes.
    """
    attributes = getattr(span, "attributes", None) or {}
    return {"name": getattr(span, "name", ""), "attributes": dict(attributes)}


class GreenSarcSpanProcessor:
    """OpenTelemetry ``SpanProcessor`` that feeds ended spans to a consumer.

    Implements the ``SpanProcessor`` interface structurally (``on_start``,
    ``on_end``, ``shutdown``, ``force_flush``) so it can be registered with
    ``TracerProvider.add_span_processor`` without this module importing
    ``opentelemetry``.  On span end it maps the span to actuals and forwards them;
    extraction failures are logged and swallowed so telemetry never breaks the
    traced application.
    """

    def __init__(self, consumer: OTelActualsConsumer) -> None:
        self.consumer = consumer

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        return None

    def on_end(self, span: Any) -> None:
        try:
            self.consumer.ingest_span(span_to_dict(span))
        except Exception:  # pragma: no cover - defensive; telemetry must not raise
            logger.exception("GreenSarcSpanProcessor failed to ingest a span — suppressed")

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True
