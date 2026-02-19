"""Tests for Datadog message/output/token extraction."""

import pytest
from inspect_scout.sources._datadog.detection import Provider
from inspect_scout.sources._datadog.extraction import (
    extract_input_messages,
    extract_output,
    extract_tools,
    extract_usage,
    sum_latency,
    sum_tokens,
)

from .mocks import _SECOND_NS, create_llm_span

pytestmark = pytest.mark.usefixtures("no_fallback_warnings")


class TestExtractInputMessages:
    """Tests for extract_input_messages function."""

    @pytest.mark.asyncio
    async def test_extract_from_messages(self) -> None:
        """Extract messages from meta.input.messages."""
        span = create_llm_span(
            input_messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        )
        messages = await extract_input_messages(span, Provider.OPENAI)

        assert len(messages) >= 1
        assert any(m.role == "user" for m in messages)

    @pytest.mark.asyncio
    async def test_extract_from_value_fallback(self) -> None:
        """Fall back to meta.input.value when no messages."""
        span = create_llm_span()
        span["meta"]["input"] = {"value": "What is 2+2?"}
        messages = await extract_input_messages(span, Provider.OPENAI)

        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "2+2" in messages[0].content

    @pytest.mark.asyncio
    async def test_extract_empty_input(self) -> None:
        """Return empty list when no input data."""
        span = create_llm_span()
        span["meta"]["input"] = {}
        messages = await extract_input_messages(span, Provider.OPENAI)

        assert messages == []


class TestExtractOutput:
    """Tests for extract_output function."""

    @pytest.mark.asyncio
    async def test_extract_output_messages(self) -> None:
        """Extract output from meta.output.messages."""
        span = create_llm_span(
            output_messages=[{"role": "assistant", "content": "The answer is 42."}]
        )
        output = await extract_output(span)

        assert output is not None
        assert output.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_extract_output_with_tool_calls(self) -> None:
        """Extract output with tool calls."""
        span = create_llm_span(
            output_messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "SF"}',
                            },
                        }
                    ],
                }
            ]
        )
        output = await extract_output(span)

        assert output is not None
        assert output.choices[0].stop_reason == "tool_calls"
        msg = output.choices[0].message
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].function == "get_weather"

    @pytest.mark.asyncio
    async def test_extract_output_value_fallback(self) -> None:
        """Fall back to meta.output.value."""
        span = create_llm_span()
        span["meta"]["output"] = {"value": "Simple response"}
        output = await extract_output(span)

        assert output is not None


class TestExtractUsage:
    """Tests for extract_usage function."""

    def test_extract_usage(self) -> None:
        """Extract token usage from metrics."""
        span = create_llm_span(input_tokens=100, output_tokens=50, total_tokens=150)
        usage = extract_usage(span)

        assert usage is not None
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_no_metrics_returns_none(self) -> None:
        """Return None when no token metrics."""
        span = create_llm_span()
        span["metrics"] = {}
        usage = extract_usage(span)

        assert usage is None

    def test_partial_metrics(self) -> None:
        """Handle partial metrics (only input_tokens)."""
        span = create_llm_span()
        span["metrics"] = {"input_tokens": 50}
        usage = extract_usage(span)

        assert usage is not None
        assert usage.input_tokens == 50
        assert usage.output_tokens == 0


class TestExtractTools:
    """Tests for extract_tools function."""

    def test_extract_tool_definitions(self) -> None:
        """Extract tool definitions from metadata."""
        span = create_llm_span(
            metadata={
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {
                                        "type": "string",
                                        "description": "City name",
                                    }
                                },
                                "required": ["city"],
                            },
                        },
                    }
                ]
            }
        )
        tools = extract_tools(span)

        assert len(tools) == 1
        assert tools[0].name == "get_weather"

    def test_no_tools_returns_empty(self) -> None:
        """Return empty list when no tools in metadata."""
        span = create_llm_span()
        tools = extract_tools(span)

        assert tools == []


class TestSumTokens:
    """Tests for sum_tokens function."""

    def test_sum_across_spans(self) -> None:
        """Sum tokens across multiple spans."""
        span1 = create_llm_span(input_tokens=10, output_tokens=5)
        span2 = create_llm_span(input_tokens=20, output_tokens=10)

        total = sum_tokens([span1, span2])
        assert total == 45

    def test_empty_spans(self) -> None:
        """Sum of empty list is 0."""
        assert sum_tokens([]) == 0


class TestSumLatency:
    """Tests for sum_latency function."""

    def test_sum_nanosecond_durations(self) -> None:
        """Sum duration from nanoseconds to seconds."""
        span1 = create_llm_span(duration=_SECOND_NS)
        span2 = create_llm_span(duration=2 * _SECOND_NS)

        total = sum_latency([span1, span2])
        assert total == pytest.approx(3.0)

    def test_empty_spans(self) -> None:
        """Sum of empty list is 0."""
        assert sum_latency([]) == pytest.approx(0.0)

    def test_missing_duration(self) -> None:
        """Skip spans without duration."""
        span = create_llm_span()
        del span["duration"]

        assert sum_latency([span]) == pytest.approx(0.0)
