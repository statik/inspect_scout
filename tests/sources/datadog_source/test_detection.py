"""Tests for Datadog provider detection."""

from typing import Any

from inspect_scout.sources._datadog.detection import (
    Provider,
    detect_provider,
    get_model_name,
    is_agent_span,
    is_llm_span,
    is_tool_span,
)

from .mocks import create_agent_span, create_llm_span, create_tool_span


class TestDetectProvider:
    """Tests for detect_provider function."""

    def test_detect_openai_from_provider_field(self) -> None:
        """Detect OpenAI from model_provider field."""
        span = create_llm_span(model_provider="openai")
        assert detect_provider(span) == Provider.OPENAI

    def test_detect_anthropic_from_provider_field(self) -> None:
        """Detect Anthropic from model_provider field."""
        span = create_llm_span(model_provider="anthropic", model_name="claude-3-sonnet")
        assert detect_provider(span) == Provider.ANTHROPIC

    def test_detect_google_from_provider_field(self) -> None:
        """Detect Google from model_provider field."""
        span = create_llm_span(model_provider="google", model_name="gemini-1.5-pro")
        assert detect_provider(span) == Provider.GOOGLE

    def test_detect_google_genai(self) -> None:
        """Detect Google from google_genai provider value."""
        span: dict[str, Any] = {"model_provider": "google_genai"}
        assert detect_provider(span) == Provider.GOOGLE

    def test_detect_vertex_ai(self) -> None:
        """Detect Google from vertex_ai provider value."""
        span: dict[str, Any] = {"model_provider": "vertex_ai"}
        assert detect_provider(span) == Provider.GOOGLE

    def test_detect_from_model_name_openai(self) -> None:
        """Detect OpenAI from model name when no provider field."""
        span: dict[str, Any] = {"model_name": "gpt-4o"}
        assert detect_provider(span) == Provider.OPENAI

    def test_detect_from_model_name_anthropic(self) -> None:
        """Detect Anthropic from model name when no provider field."""
        span: dict[str, Any] = {"model_name": "claude-3-sonnet"}
        assert detect_provider(span) == Provider.ANTHROPIC

    def test_detect_from_model_name_google(self) -> None:
        """Detect Google from model name when no provider field."""
        span: dict[str, Any] = {"model_name": "gemini-1.5-pro"}
        assert detect_provider(span) == Provider.GOOGLE

    def test_detect_unknown(self) -> None:
        """Return UNKNOWN when no detection signals present."""
        span: dict[str, Any] = {}
        assert detect_provider(span) == Provider.UNKNOWN

    def test_provider_field_case_insensitive(self) -> None:
        """Provider field lookup is case-insensitive."""
        span: dict[str, Any] = {"model_provider": "OpenAI"}
        assert detect_provider(span) == Provider.OPENAI


class TestGetModelName:
    """Tests for get_model_name function."""

    def test_get_model_name(self) -> None:
        """Get model from model_name field."""
        span = create_llm_span(model_name="gpt-4o-mini")
        assert get_model_name(span) == "gpt-4o-mini"

    def test_no_model_returns_none(self) -> None:
        """Return None when model_name not present."""
        span: dict[str, Any] = {}
        assert get_model_name(span) is None


class TestIsLLMSpan:
    """Tests for is_llm_span function."""

    def test_llm_span_detected(self) -> None:
        """Detect LLM span from meta.kind."""
        span = create_llm_span()
        assert is_llm_span(span) is True

    def test_tool_span_not_llm(self) -> None:
        """Tool span is not an LLM span."""
        span = create_tool_span()
        assert is_llm_span(span) is False

    def test_agent_span_not_llm(self) -> None:
        """Agent span is not an LLM span."""
        span = create_agent_span()
        assert is_llm_span(span) is False

    def test_case_insensitive(self) -> None:
        """meta.kind comparison is case-insensitive."""
        span: dict[str, Any] = {"meta": {"kind": "LLM"}}
        assert is_llm_span(span) is True


class TestIsToolSpan:
    """Tests for is_tool_span function."""

    def test_tool_span_detected(self) -> None:
        """Detect tool span from meta.kind."""
        span = create_tool_span()
        assert is_tool_span(span) is True

    def test_llm_span_not_tool(self) -> None:
        """LLM span is not a tool span."""
        span = create_llm_span()
        assert is_tool_span(span) is False


class TestIsAgentSpan:
    """Tests for is_agent_span function."""

    def test_agent_span_detected(self) -> None:
        """Detect agent span from meta.kind."""
        span = create_agent_span()
        assert is_agent_span(span) is True

    def test_workflow_span_detected(self) -> None:
        """Detect workflow span as agent span."""
        span = create_agent_span(kind="workflow")
        assert is_agent_span(span) is True

    def test_task_span_detected(self) -> None:
        """Detect task span as agent span."""
        span = create_agent_span(kind="task")
        assert is_agent_span(span) is True

    def test_llm_span_not_agent(self) -> None:
        """LLM span is not an agent span."""
        span = create_llm_span()
        assert is_agent_span(span) is False
