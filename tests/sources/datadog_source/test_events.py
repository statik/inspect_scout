"""Tests for Datadog event conversion."""

import pytest
from inspect_ai.event import ModelEvent, SpanBeginEvent, SpanEndEvent, ToolEvent
from inspect_scout.sources._datadog.events import (
    _ms_to_datetime,
    spans_to_events,
    to_model_event,
    to_span_begin_event,
    to_span_end_event,
    to_tool_event,
)

from .mocks import (
    _BASE_MS,
    _SECOND_MS,
    create_agent_span,
    create_llm_span,
    create_multiturn_trace,
    create_tool_call_trace,
    create_tool_span,
)

pytestmark = pytest.mark.usefixtures("no_fallback_warnings")


class TestMsToDatetime:
    """Tests for _ms_to_datetime helper."""

    def test_converts_milliseconds(self) -> None:
        """Convert millisecond timestamp to UTC datetime."""
        dt = _ms_to_datetime(1700000000000)
        assert dt.year == 2023
        assert dt.month == 11

    def test_none_returns_min_utc(self) -> None:
        """Return UTC-aware datetime.min for None input."""
        from datetime import datetime, timezone

        dt = _ms_to_datetime(None)
        assert dt == datetime.min.replace(tzinfo=timezone.utc)
        assert dt.tzinfo is not None

    def test_invalid_returns_min_utc(self) -> None:
        """Return UTC-aware datetime.min for invalid input."""
        from datetime import datetime, timezone

        dt = _ms_to_datetime("not-a-number")
        assert dt == datetime.min.replace(tzinfo=timezone.utc)
        assert dt.tzinfo is not None


class TestToModelEvent:
    """Tests for to_model_event function."""

    @pytest.mark.asyncio
    async def test_basic_llm_span(self) -> None:
        """Convert basic LLM span to ModelEvent."""
        span = create_llm_span(model_name="gpt-4o-mini")
        event = await to_model_event(span)

        assert isinstance(event, ModelEvent)
        assert event.model == "gpt-4o-mini"
        assert event.timestamp is not None

    @pytest.mark.asyncio
    async def test_model_event_has_output(self) -> None:
        """ModelEvent should have output from span."""
        span = create_llm_span(
            output_messages=[{"role": "assistant", "content": "Hello! I can help."}]
        )
        event = await to_model_event(span)

        assert event.output is not None

    @pytest.mark.asyncio
    async def test_model_event_span_id(self) -> None:
        """ModelEvent should have span_id from parent."""
        span = create_llm_span(parent_id="parent-123")
        event = await to_model_event(span)

        assert event.span_id == "parent-123"

    @pytest.mark.asyncio
    async def test_model_event_config(self) -> None:
        """ModelEvent should include config from metadata."""
        span = create_llm_span(metadata={"temperature": 0.5, "max_tokens": 100})
        event = await to_model_event(span)

        assert event.config.temperature == 0.5
        assert event.config.max_tokens == 100


class TestToToolEvent:
    """Tests for to_tool_event function."""

    def test_basic_tool_span(self) -> None:
        """Convert basic tool span to ToolEvent."""
        span = create_tool_span(
            tool_name="get_weather",
            arguments={"city": "San Francisco"},
            result="Sunny, 72F",
        )
        event = to_tool_event(span)

        assert isinstance(event, ToolEvent)
        assert event.function == "get_weather"
        assert event.arguments == {"city": "San Francisco"}
        assert event.result == "Sunny, 72F"

    def test_tool_event_with_error(self) -> None:
        """Convert tool span with error to ToolEvent."""
        span = create_tool_span(
            tool_name="get_weather",
            error_message="City not found",
        )
        event = to_tool_event(span)

        assert isinstance(event, ToolEvent)
        assert event.error is not None
        assert "City not found" in event.error.message

    def test_tool_event_span_id(self) -> None:
        """ToolEvent should have span_id from parent."""
        span = create_tool_span(parent_id="llm-span-123")
        event = to_tool_event(span)

        assert event.span_id == "llm-span-123"


class TestToSpanEvents:
    """Tests for span begin/end event conversion."""

    def test_span_begin_event(self) -> None:
        """Convert agent span to SpanBeginEvent."""
        span = create_agent_span(span_id="agent-1", agent_name="my-agent")
        event = to_span_begin_event(span)

        assert isinstance(event, SpanBeginEvent)
        assert event.id == "agent-1"
        assert "my-agent" in event.name

    def test_span_end_event(self) -> None:
        """Convert agent span to SpanEndEvent."""
        span = create_agent_span(span_id="agent-1")
        event = to_span_end_event(span)

        assert isinstance(event, SpanEndEvent)
        assert event.id == "agent-1"


class TestMultiturnEvents:
    """Tests for multi-turn conversation event conversion."""

    @pytest.mark.asyncio
    async def test_multiturn_events_count(self) -> None:
        """Multi-turn conversation produces correct number of events."""
        spans = create_multiturn_trace()
        events = await spans_to_events(spans)

        model_events = [e for e in events if isinstance(e, ModelEvent)]
        assert len(model_events) == 3

    @pytest.mark.asyncio
    async def test_multiturn_events_ordered(self) -> None:
        """Multi-turn events are in chronological order."""
        spans = create_multiturn_trace()
        events = await spans_to_events(spans)

        model_events = [e for e in events if isinstance(e, ModelEvent)]
        assert len(model_events) == 3

        for i in range(len(model_events) - 1):
            assert model_events[i].timestamp <= model_events[i + 1].timestamp


class TestSpansToEvents:
    """Tests for spans_to_events function."""

    @pytest.mark.asyncio
    async def test_convert_mixed_spans(self) -> None:
        """Convert list of mixed span types to events."""
        llm_span = create_llm_span(span_id="llm-1")
        tool_span = create_tool_span(span_id="tool-1")
        agent_span = create_agent_span(span_id="agent-1")

        events = await spans_to_events([llm_span, tool_span, agent_span])

        model_events = [e for e in events if isinstance(e, ModelEvent)]
        tool_events = [e for e in events if isinstance(e, ToolEvent)]
        span_begin_events = [e for e in events if isinstance(e, SpanBeginEvent)]
        span_end_events = [e for e in events if isinstance(e, SpanEndEvent)]

        assert len(model_events) == 1
        assert len(tool_events) == 1
        assert len(span_begin_events) == 1
        assert len(span_end_events) == 1

    @pytest.mark.asyncio
    async def test_tool_call_trace_events(self) -> None:
        """Convert tool call trace to events."""
        spans = create_tool_call_trace()
        events = await spans_to_events(spans)

        model_events = [e for e in events if isinstance(e, ModelEvent)]
        tool_events = [e for e in events if isinstance(e, ToolEvent)]
        span_begin_events = [e for e in events if isinstance(e, SpanBeginEvent)]

        assert len(model_events) == 2
        assert len(tool_events) == 1
        assert len(span_begin_events) == 1

    @pytest.mark.asyncio
    async def test_events_sorted_by_timestamp(self) -> None:
        """Events should be sorted by timestamp."""
        span1 = create_llm_span(
            span_id="span-1",
            start_ns=_BASE_MS + 10 * _SECOND_MS,
        )
        span2 = create_llm_span(
            span_id="span-2",
            start_ns=_BASE_MS,
        )

        events = await spans_to_events([span1, span2])

        assert len(events) == 2
        assert events[0].timestamp <= events[1].timestamp

    @pytest.mark.asyncio
    async def test_empty_spans_returns_empty_events(self) -> None:
        """Empty span list returns empty event list."""
        events = await spans_to_events([])
        assert events == []
