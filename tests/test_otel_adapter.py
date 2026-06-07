"""Tests for the OTel actuals-consumer adapter (pure mapping logic)."""

from __future__ import annotations

import pytest

from green_sarc.adapters.otel import (
    GreenSarcSpanProcessor,
    OTelActualsConsumer,
    SpanActuals,
    span_to_dict,
)


class _FakeReadableSpan:
    """Stands in for an opentelemetry ReadableSpan (name + attributes)."""

    def __init__(self, name, attributes):
        self.name = name
        self.attributes = attributes


def test_ingest_span_extracts_total_tokens():
    seen = []
    consumer = OTelActualsConsumer(seen.append)
    span = {
        "name": "chat",
        "attributes": {
            "green_sarc.action_id": "a1",
            "gen_ai.usage.input_tokens": 100,
            "gen_ai.usage.output_tokens": 150,
            "gen_ai.request.model": "test-model",
            "green_sarc.region": "eu-west",
        },
    }
    out = consumer.ingest_span(span)
    assert isinstance(out, SpanActuals)
    assert out.actual_tokens == 250.0
    assert out.model == "test-model"
    assert out.region == "eu-west"
    assert len(seen) == 1


def test_ingest_span_prefers_total_when_present():
    consumer = OTelActualsConsumer(lambda a: None)
    span = {"attributes": {"green_sarc.action_id": "a1", "gen_ai.usage.total_tokens": 999}}
    out = consumer.ingest_span(span)
    assert out is not None and out.actual_tokens == 999.0


def test_ingest_span_without_correlation_id_is_ignored():
    consumer = OTelActualsConsumer(lambda a: None)
    out = consumer.ingest_span({"attributes": {"gen_ai.usage.total_tokens": 100}})
    assert out is None


def test_ingest_span_with_zero_tokens_is_ignored():
    consumer = OTelActualsConsumer(lambda a: None)
    span = {"attributes": {"green_sarc.action_id": "a1", "gen_ai.usage.total_tokens": 0}}
    assert consumer.ingest_span(span) is None


def test_serve_otlp_is_a_documented_stub():
    consumer = OTelActualsConsumer(lambda a: None)
    with pytest.raises(NotImplementedError):
        consumer.serve_otlp("localhost:4317")


def test_span_to_dict_reads_name_and_attributes():
    span = _FakeReadableSpan("chat", {"gen_ai.usage.total_tokens": 42})
    d = span_to_dict(span)
    assert d["name"] == "chat"
    assert d["attributes"]["gen_ai.usage.total_tokens"] == 42


def test_span_processor_feeds_consumer_on_end():
    seen = []
    processor = GreenSarcSpanProcessor(OTelActualsConsumer(seen.append))
    span = _FakeReadableSpan(
        "chat",
        {
            "green_sarc.action_id": "a1",
            "gen_ai.usage.total_tokens": 250,
            "gen_ai.request.model": "gpt-x",
        },
    )
    processor.on_start(span)  # no-op
    processor.on_end(span)
    assert len(seen) == 1
    assert seen[0].action_id == "a1"
    assert seen[0].actual_tokens == 250.0
    assert processor.force_flush() is True
    processor.shutdown()  # no-op, must not raise


def test_span_processor_swallows_extraction_errors():
    def boom(_actuals):
        raise RuntimeError("downstream failed")

    processor = GreenSarcSpanProcessor(OTelActualsConsumer(boom))
    span = _FakeReadableSpan(
        "chat", {"green_sarc.action_id": "a1", "gen_ai.usage.total_tokens": 5}
    )
    # Telemetry must never raise into the traced application.
    processor.on_end(span)
