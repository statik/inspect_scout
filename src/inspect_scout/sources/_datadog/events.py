"""Event conversion for Datadog LLM Observability spans.

Converts Datadog spans to Scout event types:
- LLM spans (span_kind="llm") -> ModelEvent
- Tool spans (span_kind="tool") -> ToolEvent
- Agent spans (span_kind in agent/workflow/task) -> SpanBeginEvent + SpanEndEvent
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from inspect_ai.event import (
    Event,
    ModelEvent,
    SpanBeginEvent,
    SpanEndEvent,
    ToolEvent,
)
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.tool._tool_call import ToolCallError

from .detection import (
    detect_provider,
    get_model_name,
    is_agent_span,
    is_llm_span,
    is_tool_span,
)
from .extraction import extract_input_messages, extract_output, extract_tools
from .tree import DATETIME_MIN_UTC


def _ms_to_datetime(ms: Any) -> datetime:
    """Convert millisecond timestamp to datetime.

    The Datadog Export API field is named ``start_ns`` but actually
    contains milliseconds (13-digit values, not 19-digit nanoseconds).

    Args:
        ms: Millisecond timestamp (int or None)

    Returns:
        UTC datetime, or datetime.min (UTC) if conversion fails
    """
    if ms is not None:
        try:
            return datetime.fromtimestamp(int(ms) / 1e3, tz=timezone.utc)
        except (ValueError, TypeError, OverflowError):
            pass
    return DATETIME_MIN_UTC


def _get_timestamp(span: dict[str, Any]) -> datetime:
    """Get start timestamp from a Datadog span."""
    return _ms_to_datetime(span.get("start_ns"))


def _get_end_timestamp(span: dict[str, Any]) -> datetime:
    """Get end timestamp from a Datadog span (start_ns + duration)."""
    start_ms = span.get("start_ns")
    duration = span.get("duration")
    if start_ms is not None and duration is not None:
        try:
            end_ms = int(start_ms) + int(duration)
            return datetime.fromtimestamp(end_ms / 1e3, tz=timezone.utc)
        except (ValueError, TypeError, OverflowError):
            pass
    return DATETIME_MIN_UTC


async def to_model_event(span: dict[str, Any]) -> ModelEvent:
    """Convert Datadog LLM span to ModelEvent.

    Args:
        span: Datadog span with meta.kind="llm"

    Returns:
        ModelEvent object
    """
    provider = detect_provider(span)
    input_messages = await extract_input_messages(span, provider)
    output = await extract_output(span)
    model_name = get_model_name(span) or "unknown"

    metadata = span.get("metadata") or {}
    config = GenerateConfig(
        temperature=metadata.get("temperature"),
        max_tokens=metadata.get("max_tokens"),
        top_p=metadata.get("top_p"),
    )

    return ModelEvent(
        model=model_name,
        input=input_messages,
        tools=extract_tools(span),
        tool_choice="auto",
        config=config,
        output=output,
        timestamp=_get_timestamp(span),
        completed=_get_end_timestamp(span),
        span_id=span.get("parent_id") or "",
    )


def to_tool_event(span: dict[str, Any]) -> ToolEvent:
    """Convert Datadog tool span to ToolEvent.

    Args:
        span: Datadog span with meta.kind="tool"

    Returns:
        ToolEvent object
    """
    input_data = span.get("input") or {}
    output_data = span.get("output") or {}

    error = None
    if str(span.get("status", "")).lower() == "error":
        error_meta = span.get("error") or {}
        error = ToolCallError(
            type="unknown",
            message=error_meta.get("message") or "Unknown error",
        )

    function_name = span.get("name") or "unknown_tool"

    arguments: dict[str, Any] = {}
    input_value = input_data.get("value")
    if input_value:
        if isinstance(input_value, str):
            try:
                parsed = json.loads(input_value)
                if isinstance(parsed, dict):
                    arguments = parsed
            except json.JSONDecodeError:
                pass
        elif isinstance(input_value, dict):
            arguments = input_value

    result = ""
    output_value = output_data.get("value")
    if output_value:
        if isinstance(output_value, dict):
            result = json.dumps(output_value)
        else:
            result = str(output_value)

    return ToolEvent(
        id=str(span.get("span_id") or uuid.uuid4()),
        type="function",
        function=str(function_name),
        arguments=arguments,
        result=result,
        timestamp=_get_timestamp(span),
        completed=_get_end_timestamp(span),
        error=error,
        span_id=span.get("parent_id") or "",
    )


def to_span_begin_event(span: dict[str, Any]) -> SpanBeginEvent:
    """Convert Datadog span to SpanBeginEvent.

    Args:
        span: Datadog span (agent/workflow/task kind)

    Returns:
        SpanBeginEvent object
    """
    name = span.get("name") or "span"

    return SpanBeginEvent(
        id=str(span.get("span_id", "")),
        name=str(name),
        parent_id=span.get("parent_id") or "",
        timestamp=_get_timestamp(span),
        working_start=0.0,
    )


def to_span_end_event(span: dict[str, Any]) -> SpanEndEvent:
    """Convert Datadog span end to SpanEndEvent.

    Args:
        span: Datadog span object

    Returns:
        SpanEndEvent object
    """
    return SpanEndEvent(
        id=str(span.get("span_id", "")),
        timestamp=_get_end_timestamp(span),
    )


async def spans_to_events(spans: list[dict[str, Any]]) -> list[Event]:
    """Convert Datadog spans to Scout events by type.

    Dispatches based on meta.kind field.

    Args:
        spans: List of Datadog spans

    Returns:
        List of Scout event objects sorted chronologically
    """
    events: list[Event] = []

    for span in spans:
        if is_llm_span(span):
            events.append(await to_model_event(span))
        elif is_tool_span(span):
            events.append(to_tool_event(span))
        elif is_agent_span(span):
            events.append(to_span_begin_event(span))
            duration = span.get("duration")
            if duration is not None:
                events.append(to_span_end_event(span))

    events.sort(key=lambda e: e.timestamp or DATETIME_MIN_UTC)

    return events
