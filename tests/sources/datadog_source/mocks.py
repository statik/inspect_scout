"""Mock Datadog span factories for testing.

Provides factory functions that create Datadog span dictionaries matching
the Export API's ``data[].attributes`` structure.
"""

from datetime import datetime, timezone
from typing import Any

# Base timestamp in nanoseconds (2023-11-14T22:13:20Z)
_BASE_NS = 1700000000000000000

# 1 second in nanoseconds
_SECOND_NS = 1_000_000_000


def create_llm_span(
    span_id: str = "span-llm-001",
    trace_id: str = "trace-001",
    parent_id: str = "",
    name: str = "chat gpt-4o",
    model_name: str = "gpt-4o",
    model_provider: str = "openai",
    input_messages: list[dict[str, Any]] | None = None,
    output_messages: list[dict[str, Any]] | None = None,
    input_value: str | None = None,
    output_value: str | None = None,
    input_tokens: int = 10,
    output_tokens: int = 5,
    total_tokens: int | None = None,
    start_ns: int = _BASE_NS,
    duration: int = _SECOND_NS,
    status: str = "ok",
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Create a mock Datadog LLM span.

    Args:
        span_id: Span identifier
        trace_id: Trace identifier
        parent_id: Parent span ID (empty for root)
        name: Span name
        model_name: Model name
        model_provider: Provider name (openai, anthropic, etc.)
        input_messages: Input messages array
        output_messages: Output messages array
        input_value: Fallback input text
        output_value: Fallback output text
        input_tokens: Input token count
        output_tokens: Output token count
        total_tokens: Total token count (defaults to input + output)
        start_ns: Start time in nanoseconds
        duration: Duration in nanoseconds
        status: Span status
        metadata: Additional metadata
        tags: Span tags

    Returns:
        Datadog span dictionary
    """
    if input_messages is None:
        input_messages = [{"role": "user", "content": "Hello"}]
    if output_messages is None:
        output_messages = [{"role": "assistant", "content": "Hi!"}]
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    meta_input: dict[str, Any] = {"messages": input_messages}
    if input_value:
        meta_input["value"] = input_value

    meta_output: dict[str, Any] = {"messages": output_messages}
    if output_value:
        meta_output["value"] = output_value

    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "name": name,
        "start_ns": start_ns,
        "duration": duration,
        "status": status,
        "model_name": model_name,
        "model_provider": model_provider,
        "meta": {
            "kind": "llm",
            "input": meta_input,
            "output": meta_output,
            "metadata": metadata or {"temperature": 0.7},
        },
        "metrics": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        },
        "tags": tags or ["ml_app:my-app", "env:prod"],
    }


def create_tool_span(
    span_id: str = "span-tool-001",
    trace_id: str = "trace-001",
    parent_id: str = "",
    tool_name: str = "get_weather",
    arguments: dict[str, Any] | None = None,
    result: str | None = None,
    start_ns: int = _BASE_NS,
    duration: int = _SECOND_NS // 2,
    status: str = "ok",
    error_message: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Create a mock Datadog tool span.

    Args:
        span_id: Span identifier
        trace_id: Trace identifier
        parent_id: Parent span ID
        tool_name: Tool function name
        arguments: Tool arguments
        result: Tool result
        start_ns: Start time in nanoseconds
        duration: Duration in nanoseconds
        status: Span status
        error_message: Error message (sets status to "error")
        tags: Span tags

    Returns:
        Datadog span dictionary
    """
    if arguments is None:
        arguments = {"city": "San Francisco"}
    if result is None:
        result = "Sunny, 72F"

    meta: dict[str, Any] = {
        "kind": "tool",
        "input": {"value": arguments},
        "output": {"value": result},
    }

    if error_message:
        status = "error"
        meta["error"] = {"message": error_message}

    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "name": tool_name,
        "start_ns": start_ns,
        "duration": duration,
        "status": status,
        "meta": meta,
        "metrics": {},
        "tags": tags or ["ml_app:my-app"],
    }


def create_agent_span(
    span_id: str = "span-agent-001",
    trace_id: str = "trace-001",
    parent_id: str = "",
    agent_name: str = "assistant",
    kind: str = "agent",
    start_ns: int = _BASE_NS,
    duration: int = 5 * _SECOND_NS,
    status: str = "ok",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Create a mock Datadog agent/workflow/task span.

    Args:
        span_id: Span identifier
        trace_id: Trace identifier
        parent_id: Parent span ID
        agent_name: Agent name
        kind: Span kind (agent, workflow, task)
        start_ns: Start time in nanoseconds
        duration: Duration in nanoseconds
        status: Span status
        tags: Span tags

    Returns:
        Datadog span dictionary
    """
    return {
        "span_id": span_id,
        "trace_id": trace_id,
        "parent_id": parent_id,
        "name": agent_name,
        "start_ns": start_ns,
        "duration": duration,
        "status": status,
        "meta": {
            "kind": kind,
            "input": {"value": "Hello!"},
            "output": {"value": "Hello! How can I help you?"},
        },
        "metrics": {},
        "tags": tags or ["ml_app:my-app"],
    }


def create_multiturn_trace(
    trace_id: str = "trace-multiturn",
    model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Create a multi-turn conversation trace.

    Three LLM spans with progressive conversation history.

    Args:
        trace_id: Trace identifier
        model: Model name

    Returns:
        List of span dictionaries
    """
    span1 = create_llm_span(
        span_id="span-turn-1",
        trace_id=trace_id,
        model_name=model,
        input_messages=[
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 2 + 2?"},
        ],
        output_messages=[
            {"role": "assistant", "content": "2 + 2 equals 4."},
        ],
        input_tokens=30,
        output_tokens=10,
        start_ns=_BASE_NS,
    )
    span2 = create_llm_span(
        span_id="span-turn-2",
        trace_id=trace_id,
        model_name=model,
        input_messages=[
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "What about 3 + 3?"},
        ],
        output_messages=[
            {"role": "assistant", "content": "3 + 3 equals 6."},
        ],
        input_tokens=50,
        output_tokens=10,
        start_ns=_BASE_NS + _SECOND_NS,
    )
    span3 = create_llm_span(
        span_id="span-turn-3",
        trace_id=trace_id,
        model_name=model,
        input_messages=[
            {"role": "system", "content": "You are a helpful math tutor."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 equals 4."},
            {"role": "user", "content": "What about 3 + 3?"},
            {"role": "assistant", "content": "3 + 3 equals 6."},
            {"role": "user", "content": "And 5 + 5?"},
        ],
        output_messages=[
            {"role": "assistant", "content": "5 + 5 equals 10."},
        ],
        input_tokens=70,
        output_tokens=10,
        start_ns=_BASE_NS + 2 * _SECOND_NS,
    )
    return [span1, span2, span3]


def create_tool_call_trace(
    trace_id: str = "trace-tool-call",
    model: str = "gpt-4o",
) -> list[dict[str, Any]]:
    """Create a trace with an agent span, LLM calls, and a tool execution.

    Args:
        trace_id: Trace identifier
        model: Model name

    Returns:
        List of span dictionaries
    """
    agent = create_agent_span(
        span_id="span-agent-root",
        trace_id=trace_id,
        start_ns=_BASE_NS,
    )
    llm1 = create_llm_span(
        span_id="span-llm-1",
        trace_id=trace_id,
        parent_id="span-agent-root",
        model_name=model,
        input_messages=[
            {"role": "user", "content": "What's the weather in SF?"},
        ],
        output_messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "San Francisco"}',
                        },
                    }
                ],
            }
        ],
        start_ns=_BASE_NS + _SECOND_NS // 10,
    )
    tool = create_tool_span(
        span_id="span-tool-1",
        trace_id=trace_id,
        parent_id="span-agent-root",
        tool_name="get_weather",
        arguments={"city": "San Francisco"},
        result="Sunny, 72F",
        start_ns=_BASE_NS + 2 * _SECOND_NS // 10,
    )
    llm2 = create_llm_span(
        span_id="span-llm-2",
        trace_id=trace_id,
        parent_id="span-agent-root",
        model_name=model,
        input_messages=[
            {"role": "user", "content": "What's the weather in SF?"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "Sunny, 72F"},
        ],
        output_messages=[
            {
                "role": "assistant",
                "content": "The weather in San Francisco is sunny, 72F.",
            }
        ],
        start_ns=_BASE_NS + 3 * _SECOND_NS // 10,
    )
    return [agent, llm1, tool, llm2]
