"""Tests for _build_transcript and datadog() with mocked client."""

from collections.abc import AsyncGenerator, Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from inspect_scout._transcript.types import Transcript
from inspect_scout.sources._datadog import (
    _build_transcript,
    _extract_metadata,
    _extract_model_options,
    _extract_root_messages,
    _from_query,
    _get_ml_app_from_tags,
    _get_tag_value,
    _matches_trace_filter,
    _root_duration,
    _should_deduplicate,
    datadog,
)
from inspect_scout.sources._datadog.client import DATADOG_SOURCE_TYPE

from .mocks import (
    _BASE_MS,
    _SECOND_MS,
    create_agent_span,
    create_llm_span,
    create_multiturn_trace,
    create_tool_call_trace,
)


def _pages(
    *page_lists: list[dict[str, Any]],
) -> Callable[..., AsyncGenerator[list[dict[str, Any]], None]]:
    """Create a mock ``iter_span_pages`` that yields the given pages."""

    async def _iter(**kwargs: Any) -> AsyncGenerator[list[dict[str, Any]], None]:
        for page in page_lists:
            yield page

    return _iter


def _error_pages(
    error: Exception,
) -> Callable[..., AsyncGenerator[list[dict[str, Any]], None]]:
    """Create a mock ``iter_span_pages`` that raises immediately."""

    async def _iter(**kwargs: Any) -> AsyncGenerator[list[dict[str, Any]], None]:
        raise error
        yield []  # unreachable; makes this an async generator

    return _iter


class TestBuildTranscript:
    """Tests for _build_transcript function."""

    pytestmark = pytest.mark.usefixtures("no_fallback_warnings")

    @pytest.mark.asyncio
    async def test_empty_spans_returns_none(self) -> None:
        """Return None for empty span list."""
        result = await _build_transcript([], "my-app", "trace-1", "datadoghq.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_single_llm_span_transcript(self) -> None:
        """Build transcript from a single LLM span."""
        span = create_llm_span(
            span_id="span-1",
            trace_id="trace-1",
            model_name="gpt-4o",
            input_messages=[{"role": "user", "content": "Hello"}],
            output_messages=[{"role": "assistant", "content": "Hi!"}],
            input_tokens=10,
            output_tokens=5,
        )
        transcript = await _build_transcript(
            [span], "my-app", "trace-1", "datadoghq.com"
        )

        assert transcript is not None
        assert transcript.transcript_id == "trace-1"
        assert transcript.source_type == DATADOG_SOURCE_TYPE
        assert transcript.source_id == "my-app"
        assert transcript.model == "gpt-4o"
        assert transcript.total_tokens == 15
        assert len(transcript.messages) >= 1
        assert len(transcript.events) >= 1

    @pytest.mark.asyncio
    async def test_total_time_uses_root_duration(self) -> None:
        """total_time uses root span duration, not sum of all spans."""
        agent = create_agent_span(
            span_id="root",
            trace_id="trace-1",
            duration=5 * _SECOND_MS,
            start_ns=_BASE_MS,
        )
        child1 = create_llm_span(
            span_id="child-1",
            trace_id="trace-1",
            parent_id="root",
            duration=2 * _SECOND_MS,
            start_ns=_BASE_MS + _SECOND_MS,
        )
        child2 = create_llm_span(
            span_id="child-2",
            trace_id="trace-1",
            parent_id="root",
            duration=2 * _SECOND_MS,
            start_ns=_BASE_MS + 3 * _SECOND_MS,
        )

        transcript = await _build_transcript(
            [agent, child1, child2], "my-app", "trace-1", "datadoghq.com"
        )

        assert transcript is not None
        assert transcript.total_time == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_tool_call_trace_transcript(self) -> None:
        """Build transcript from a tool-call trace."""
        spans = create_tool_call_trace(trace_id="trace-tool")
        transcript = await _build_transcript(
            spans, "my-app", "trace-tool", "datadoghq.com"
        )

        assert transcript is not None
        assert transcript.transcript_id == "trace-tool"
        assert len(transcript.events) > 0
        assert len(transcript.messages) > 0

    @pytest.mark.asyncio
    async def test_source_uri_format(self) -> None:
        """source_uri follows expected Datadog URL format."""
        span = create_llm_span(trace_id="trace-abc")
        transcript = await _build_transcript(
            [span], "my-app", "trace-abc", "datadoghq.eu"
        )

        assert transcript is not None
        assert transcript.source_uri == "https://app.datadoghq.eu/llm/traces/trace-abc"

    @pytest.mark.asyncio
    async def test_ml_app_from_tags_when_none(self) -> None:
        """Extract ml_app from span tags when ml_app param is None."""
        span = create_llm_span(
            trace_id="trace-1", tags=["ml_app:from-tags", "env:prod"]
        )
        transcript = await _build_transcript([span], None, "trace-1", "datadoghq.com")

        assert transcript is not None
        assert transcript.source_id == "from-tags"

    @pytest.mark.asyncio
    async def test_best_model_selects_longest_input(self) -> None:
        """best_model heuristic picks the ModelEvent with the most input messages."""
        spans = create_multiturn_trace(trace_id="trace-multi")
        transcript = await _build_transcript(
            spans, "my-app", "trace-multi", "datadoghq.com"
        )

        assert transcript is not None
        # span3 has 6 input messages (longest); output adds 1 more
        assert transcript.message_count == 7
        # Verify span3 was actually selected by checking unique content
        assert transcript.messages[-2].content == "And 5 + 5?"
        assert transcript.messages[-1].content == "5 + 5 equals 10."

    @pytest.mark.asyncio
    async def test_error_span_captured(self) -> None:
        """Transcript captures error from span with error status."""
        span = create_llm_span(trace_id="trace-err", status="error")
        span["error"] = {"message": "Rate limited"}
        transcript = await _build_transcript(
            [span], "my-app", "trace-err", "datadoghq.com"
        )

        assert transcript is not None
        assert transcript.error == "Rate limited"


@pytest.mark.usefixtures("no_fallback_warnings")
class TestExtractModelOptions:
    """Tests for _extract_model_options function."""

    def test_extracts_all_options(self) -> None:
        """Extract temperature, max_tokens, top_p, top_k from metadata."""
        span = create_llm_span(
            metadata={
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
                "top_k": 40,
            }
        )
        result = _extract_model_options(span)
        assert result == {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9,
            "top_k": 40,
        }

    def test_partial_options(self) -> None:
        """Extract only the options that are present."""
        span = create_llm_span(metadata={"temperature": 0.5})
        result = _extract_model_options(span)
        assert result == {"temperature": 0.5}

    def test_no_options_returns_none(self) -> None:
        """Return None when no model options are present."""
        span = create_llm_span(metadata={"other_key": "value"})
        result = _extract_model_options(span)
        assert result is None

    def test_missing_metadata_returns_none(self) -> None:
        """Return None when metadata is absent."""
        span: dict[str, Any] = {}
        result = _extract_model_options(span)
        assert result is None

    @pytest.mark.asyncio
    async def test_model_options_in_transcript(self) -> None:
        """model_options are passed through to the transcript."""
        span = create_llm_span(
            trace_id="trace-opts",
            metadata={"temperature": 0.3, "max_tokens": 512},
        )
        transcript = await _build_transcript(
            [span], "my-app", "trace-opts", "datadoghq.com"
        )
        assert transcript is not None
        assert transcript.model_options == {
            "temperature": 0.3,
            "max_tokens": 512,
        }


class TestDatadogGenerator:
    """Tests for the datadog() async generator with mocked client."""

    pytestmark = pytest.mark.usefixtures("no_fallback_warnings")

    @pytest.mark.asyncio
    async def test_yields_transcripts(self) -> None:
        """datadog() yields transcripts from mocked API response."""
        spans = [
            create_llm_span(span_id="s1", trace_id="t1"),
            create_llm_span(span_id="s2", trace_id="t2"),
        ]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(spans)
        mock_client.site = "datadoghq.com"
        mock_client.aclose = AsyncMock()

        with patch(
            "inspect_scout.sources._datadog.get_datadog_client",
            return_value=mock_client,
        ):
            transcripts = []
            async for t in datadog(ml_app="test-app"):
                transcripts.append(t)

        assert len(transcripts) == 2
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_respects_limit(self) -> None:
        """datadog() respects the limit parameter."""
        spans = [
            create_llm_span(span_id="s1", trace_id="t1"),
            create_llm_span(span_id="s2", trace_id="t2"),
            create_llm_span(span_id="s3", trace_id="t3"),
        ]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(spans)
        mock_client.site = "datadoghq.com"
        mock_client.aclose = AsyncMock()

        with patch(
            "inspect_scout.sources._datadog.get_datadog_client",
            return_value=mock_client,
        ):
            transcripts = []
            async for t in datadog(ml_app="test-app", limit=2):
                transcripts.append(t)

        assert len(transcripts) == 2
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_api_error_propagates(self) -> None:
        """API errors propagate instead of being swallowed."""
        mock_client = AsyncMock()
        mock_client.iter_span_pages = _error_pages(ValueError("Bad request"))
        mock_client.site = "datadoghq.com"
        mock_client.aclose = AsyncMock()

        with patch(
            "inspect_scout.sources._datadog.get_datadog_client",
            return_value=mock_client,
        ):
            with pytest.raises(ValueError, match="Bad request"):
                async for _ in datadog(ml_app="test-app"):
                    pass

        mock_client.aclose.assert_awaited_once()


class TestStrictImport:
    """Tests for DATADOG_STRICT_IMPORT env-var behaviour."""

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("no_fallback_warnings")
    @pytest.mark.parametrize("env_value", ["1", "true", "TRUE"])
    async def test_strict_import_propagates_trace_errors(
        self, monkeypatch: pytest.MonkeyPatch, env_value: str
    ) -> None:
        """DATADOG_STRICT_IMPORT re-raises per-trace processing errors."""
        monkeypatch.setenv("DATADOG_STRICT_IMPORT", env_value)

        spans = [create_llm_span(span_id="s1", trace_id="t1")]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(spans)
        mock_client.site = "datadoghq.com"

        with patch(
            "inspect_scout.sources._datadog._build_transcript",
            side_effect=RuntimeError("bad span data"),
        ):
            with pytest.raises(RuntimeError, match="bad span data"):
                async for _ in _from_query(
                    mock_client,
                    "test-app",
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ):
                    pass

    @pytest.mark.asyncio
    async def test_default_import_skips_trace_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without DATADOG_STRICT_IMPORT, per-trace errors are logged and skipped."""
        monkeypatch.delenv("DATADOG_STRICT_IMPORT", raising=False)

        warnings: list[str] = []

        def _capture_warning(msg: str, *args: object, **kwargs: object) -> None:
            warnings.append(msg % args if args else msg)

        import inspect_scout.sources._datadog as datadog_pkg
        import inspect_scout.sources._datadog.extraction as datadog_extraction

        monkeypatch.setattr(datadog_pkg.logger, "warning", _capture_warning)
        monkeypatch.setattr(datadog_extraction.logger, "warning", _capture_warning)

        spans = [
            create_llm_span(span_id="s1", trace_id="t1"),
            create_llm_span(span_id="s2", trace_id="t2"),
        ]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(spans)
        mock_client.site = "datadoghq.com"

        with patch(
            "inspect_scout.sources._datadog._build_transcript",
            side_effect=RuntimeError("bad span data"),
        ):
            results: list[object] = []
            async for t in _from_query(
                mock_client,
                "test-app",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ):
                results.append(t)

        assert results == []
        assert len(warnings) == 2


class TestExtractRootMessages:
    """Tests for _extract_root_messages function."""

    def test_extracts_input_and_output(self) -> None:
        """Extract user and assistant messages from root span."""
        span: dict[str, Any] = {
            "input": {"value": "Hello"},
            "output": {"value": "Hi there!"},
        }
        messages = _extract_root_messages(span)
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

    def test_input_only(self) -> None:
        """Extract only input when output is missing."""
        span: dict[str, Any] = {
            "input": {"value": "Hello"},
            "output": {},
        }
        messages = _extract_root_messages(span)
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    def test_empty_input_output(self) -> None:
        """Return empty list when span has no input/output."""
        span: dict[str, Any] = {}
        messages = _extract_root_messages(span)
        assert messages == []


class TestRootDuration:
    """Tests for _root_duration function."""

    def test_valid_duration(self) -> None:
        """Convert millisecond duration to seconds."""
        span: dict[str, Any] = {"duration": 5_000}
        assert _root_duration(span) == pytest.approx(5.0)

    def test_missing_duration_returns_none(self) -> None:
        """Return None when duration key is absent."""
        span: dict[str, Any] = {}
        assert _root_duration(span) is None

    def test_none_duration_returns_none(self) -> None:
        """Return None when duration is explicitly None."""
        span: dict[str, Any] = {"duration": None}
        assert _root_duration(span) is None

    def test_invalid_duration_returns_none(self) -> None:
        """Return None when duration is not convertible to int."""
        span: dict[str, Any] = {"duration": "not-a-number"}
        assert _root_duration(span) is None


class TestGetMlAppFromTags:
    """Tests for _get_ml_app_from_tags function."""

    def test_extracts_ml_app(self) -> None:
        """Extract ml_app value from tag list."""
        span: dict[str, Any] = {"tags": ["ml_app:my-app", "env:prod"]}
        assert _get_ml_app_from_tags(span) == "my-app"

    def test_no_ml_app_tag(self) -> None:
        """Return None when no ml_app tag exists."""
        span: dict[str, Any] = {"tags": ["env:prod"]}
        assert _get_ml_app_from_tags(span) is None

    def test_no_tags_key(self) -> None:
        """Return None when tags key is absent."""
        span: dict[str, Any] = {}
        assert _get_ml_app_from_tags(span) is None

    def test_dict_tags_returns_none(self) -> None:
        """Return None when tags is a dict instead of a list."""
        span: dict[str, Any] = {"tags": {"ml_app": "my-app"}}
        assert _get_ml_app_from_tags(span) is None

    def test_non_string_tag_skipped(self) -> None:
        """Skip non-string entries in tags list."""
        span: dict[str, Any] = {"tags": [123, "ml_app:found"]}
        assert _get_ml_app_from_tags(span) == "found"


class TestExtractMetadataSuccess:
    """Tests for success field validation in _extract_metadata."""

    def test_bool_success_preserved(self) -> None:
        """Boolean success values are preserved."""
        span: dict[str, Any] = {
            "metadata": {"success": True},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is True

    def test_string_success_coerced_to_bool(self) -> None:
        """String 'true' is coerced to bool True."""
        span: dict[str, Any] = {
            "metadata": {"success": "true"},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is True

    def test_string_false_coerced_to_false(self) -> None:
        """String 'false' is coerced to bool False."""
        span: dict[str, Any] = {
            "metadata": {"success": "false"},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is False

    def test_string_zero_coerced_to_false(self) -> None:
        """String '0' is coerced to bool False."""
        span: dict[str, Any] = {
            "metadata": {"success": "0"},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is False

    def test_int_success_coerced_to_bool(self) -> None:
        """Integer 1 is coerced to bool True."""
        span: dict[str, Any] = {
            "metadata": {"success": 1},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is True

    def test_zero_success_coerced_to_false(self) -> None:
        """Integer 0 is coerced to bool False."""
        span: dict[str, Any] = {
            "metadata": {"success": 0},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is False

    def test_none_success_preserved(self) -> None:
        """None success value is preserved."""
        span: dict[str, Any] = {
            "metadata": {"success": None},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is None


class TestFromQueryLimitValidation:
    """Tests for _from_query limit validation."""

    pytestmark = pytest.mark.usefixtures("no_fallback_warnings")

    @pytest.mark.asyncio
    async def test_negative_limit_raises(self) -> None:
        """Negative limit raises ValueError."""
        mock_client = AsyncMock()
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            async for _ in _from_query(
                mock_client, "app", None, None, None, None, None, None, -1
            ):
                pass

    @pytest.mark.asyncio
    async def test_zero_limit_raises(self) -> None:
        """Zero limit raises ValueError."""
        mock_client = AsyncMock()
        with pytest.raises(ValueError, match="limit must be a positive integer"):
            async for _ in _from_query(
                mock_client, "app", None, None, None, None, None, None, 0
            ):
                pass


class TestFromQueryStreaming:
    """Tests that _from_query streams traces across pages."""

    pytestmark = pytest.mark.usefixtures("no_fallback_warnings")

    @pytest.mark.asyncio
    async def test_traces_across_pages_assembled(self) -> None:
        """Spans for the same trace arriving on different pages are grouped."""
        page1 = [
            create_llm_span(span_id="s1", trace_id="t1"),
            create_llm_span(span_id="s2", trace_id="t2"),
        ]
        page2 = [
            create_llm_span(span_id="s3", trace_id="t2", parent_id="s2"),
            create_llm_span(span_id="s4", trace_id="t3"),
        ]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(page1, page2)
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client, "app", None, None, None, None, None, None, None
        ):
            results.append(t)

        ids = {t.transcript_id for t in results}
        assert ids == {"t1", "t2", "t3"}

    @pytest.mark.asyncio
    async def test_completed_traces_yielded_between_pages(self) -> None:
        """Traces absent from the next page are yielded before that page."""
        page1 = [create_llm_span(span_id="s1", trace_id="t1")]
        page2 = [create_llm_span(span_id="s2", trace_id="t2")]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(page1, page2)
        mock_client.site = "datadoghq.com"

        yield_order: list[str] = []
        async for t in _from_query(
            mock_client, "app", None, None, None, None, None, None, None
        ):
            yield_order.append(t.transcript_id)

        # t1 is completed after page 2 (absent), t2 after final flush
        assert yield_order == ["t1", "t2"]

    @pytest.mark.asyncio
    async def test_non_contiguous_trace_spans_assembled(self) -> None:
        """Spans for a trace on pages 1 and 3 (absent from page 2) are grouped."""
        page1 = [
            create_llm_span(span_id="a1", trace_id="tA"),
            create_llm_span(span_id="b1", trace_id="tB"),
        ]
        page2 = [
            create_llm_span(span_id="b2", trace_id="tB", parent_id="b1"),
        ]
        page3 = [
            create_llm_span(span_id="a2", trace_id="tA", parent_id="a1"),
            create_llm_span(span_id="c1", trace_id="tC"),
        ]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(page1, page2, page3)
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client, "app", None, None, None, None, None, None, None
        ):
            results.append(t)

        ids = {t.transcript_id for t in results}
        assert ids == {"tA", "tB", "tC"}

        # tA appears exactly once (not split across pages)
        ta_results = [t for t in results if t.transcript_id == "tA"]
        assert len(ta_results) == 1
        # Both spans contributed tokens (15 per span × 2)
        assert ta_results[0].total_tokens == 30

    @pytest.mark.asyncio
    async def test_trace_split_when_gap_exceeds_threshold(self) -> None:
        """A trace absent for >1 page is flushed and split if it reappears."""
        # tA appears on page 1, is absent from pages 2 and 3 (exceeding the
        # grace period of 1), so it gets flushed. When tA reappears on page 4
        # it starts a new partial transcript.
        page1 = [create_llm_span(span_id="a1", trace_id="tA")]
        page2 = [create_llm_span(span_id="b1", trace_id="tB")]
        page3 = [create_llm_span(span_id="b2", trace_id="tB", parent_id="b1")]
        page4 = [create_llm_span(span_id="a2", trace_id="tA")]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(page1, page2, page3, page4)
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client, "app", None, None, None, None, None, None, None
        ):
            results.append(t)

        # tA is flushed after being absent from pages 2 and 3, then appears
        # again on page 4 as a separate transcript.
        ta_results = [t for t in results if t.transcript_id == "tA"]
        assert len(ta_results) == 2
        # Each partial contains only one span's worth of tokens (15)
        for t in ta_results:
            assert t.total_tokens == 15

    @pytest.mark.asyncio
    async def test_single_page_yields_all(self) -> None:
        """A single-page response yields all traces in the final flush."""
        spans = [
            create_llm_span(span_id="s1", trace_id="t1"),
            create_llm_span(span_id="s2", trace_id="t2"),
        ]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(spans)
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client, "app", None, None, None, None, None, None, None
        ):
            results.append(t)

        assert len(results) == 2


class TestMatchesTraceFilter:
    """Tests for _matches_trace_filter function."""

    pytestmark = pytest.mark.usefixtures("no_fallback_warnings")

    @pytest.mark.asyncio
    async def test_no_filters_yields_all(self) -> None:
        """No filter params means no filtering."""
        span = create_llm_span(trace_id="t1", model_name="gpt-4o")
        transcript = await _build_transcript([span], "my-app", "t1", "datadoghq.com")
        assert transcript is not None
        assert _matches_trace_filter(transcript, None, None) is True

    @pytest.mark.asyncio
    async def test_min_messages_filters_short_traces(self) -> None:
        """Skip transcripts with fewer messages than min_messages."""
        span = create_llm_span(
            trace_id="t1",
            input_messages=[{"role": "user", "content": "Hi"}],
            output_messages=[{"role": "assistant", "content": "Hello"}],
        )
        transcript = await _build_transcript([span], "my-app", "t1", "datadoghq.com")
        assert transcript is not None
        assert transcript.message_count == 2
        assert (
            _matches_trace_filter(transcript, min_messages=5, exclude_models=None)
            is False
        )
        assert (
            _matches_trace_filter(transcript, min_messages=2, exclude_models=None)
            is True
        )
        assert (
            _matches_trace_filter(transcript, min_messages=1, exclude_models=None)
            is True
        )

    @pytest.mark.asyncio
    async def test_exclude_models_filters_matching(self) -> None:
        """Skip transcripts whose model matches an excluded entry."""
        span = create_llm_span(trace_id="t1", model_name="gpt-4o")
        transcript = await _build_transcript([span], "my-app", "t1", "datadoghq.com")
        assert transcript is not None
        assert _matches_trace_filter(transcript, None, ["gpt-4o"]) is False
        assert _matches_trace_filter(transcript, None, ["claude-3"]) is True

    @pytest.mark.asyncio
    async def test_exclude_models_case_insensitive(self) -> None:
        """Exclude model matching is case-insensitive."""
        span = create_llm_span(trace_id="t1", model_name="gpt-4o")
        transcript = await _build_transcript([span], "my-app", "t1", "datadoghq.com")
        assert transcript is not None
        assert _matches_trace_filter(transcript, None, ["GPT-4O"]) is False

    @pytest.mark.asyncio
    async def test_exclude_models_substring_match(self) -> None:
        """Exclude model matching uses substring."""
        span = create_llm_span(trace_id="t1", model_name="gpt-3.5-turbo")
        transcript = await _build_transcript([span], "my-app", "t1", "datadoghq.com")
        assert transcript is not None
        assert _matches_trace_filter(transcript, None, ["gpt-3.5"]) is False

    @pytest.mark.asyncio
    async def test_filters_combined(self) -> None:
        """Both filters active use AND semantics."""
        span = create_llm_span(
            trace_id="t1",
            model_name="gpt-4o",
            input_messages=[
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
            ],
            output_messages=[{"role": "assistant", "content": "A2"}],
        )
        transcript = await _build_transcript([span], "my-app", "t1", "datadoghq.com")
        assert transcript is not None
        # Passes min_messages but fails exclude_models
        assert (
            _matches_trace_filter(transcript, min_messages=2, exclude_models=["gpt-4o"])
            is False
        )
        # Fails min_messages but passes exclude_models
        assert (
            _matches_trace_filter(
                transcript, min_messages=100, exclude_models=["claude"]
            )
            is False
        )
        # Passes both
        assert (
            _matches_trace_filter(transcript, min_messages=2, exclude_models=["claude"])
            is True
        )

    def test_exclude_models_no_model_passes(self) -> None:
        """Transcript with no model passes exclude_models filter."""
        transcript = Transcript(
            transcript_id="t1",
            model=None,
            message_count=5,
        )
        assert _matches_trace_filter(transcript, None, ["gpt-4o"]) is True

    def test_min_messages_none_count_treated_as_zero(self) -> None:
        """Transcript with None message_count treated as 0."""
        transcript = Transcript(
            transcript_id="t1",
            message_count=None,
        )
        assert (
            _matches_trace_filter(transcript, min_messages=1, exclude_models=None)
            is False
        )


class TestTraceFilterIntegration:
    """Integration tests for filtering through _from_query."""

    pytestmark = pytest.mark.usefixtures("no_fallback_warnings")

    @pytest.mark.asyncio
    async def test_min_messages_via_from_query(self) -> None:
        """min_messages filters traces in _from_query."""
        # One trace with 2 messages, one with many (multi-turn)
        short_span = create_llm_span(
            span_id="s1",
            trace_id="t-short",
            input_messages=[{"role": "user", "content": "Hi"}],
            output_messages=[{"role": "assistant", "content": "Hello"}],
        )
        long_span = create_llm_span(
            span_id="s2",
            trace_id="t-long",
            input_messages=[
                {"role": "user", "content": "Q1"},
                {"role": "assistant", "content": "A1"},
                {"role": "user", "content": "Q2"},
                {"role": "assistant", "content": "A2"},
                {"role": "user", "content": "Q3"},
            ],
            output_messages=[{"role": "assistant", "content": "A3"}],
        )

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages([short_span, long_span])
        mock_client.site = "datadoghq.com"

        results = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            min_messages=5,
        ):
            results.append(t)

        assert len(results) == 1
        assert results[0].transcript_id == "t-long"

    @pytest.mark.asyncio
    async def test_exclude_models_via_from_query(self) -> None:
        """exclude_models filters traces in _from_query."""
        gpt_span = create_llm_span(span_id="s1", trace_id="t-gpt", model_name="gpt-4o")
        claude_span = create_llm_span(
            span_id="s2", trace_id="t-claude", model_name="claude-3-sonnet"
        )

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages([gpt_span, claude_span])
        mock_client.site = "datadoghq.com"

        results = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            exclude_models=["gpt-4"],
        ):
            results.append(t)

        assert len(results) == 1
        assert results[0].transcript_id == "t-claude"

    @pytest.mark.asyncio
    async def test_min_messages_with_limit(self) -> None:
        """Limit counts only matched transcripts, not filtered ones."""
        spans = [
            create_llm_span(
                span_id=f"s{i}",
                trace_id=f"t{i}",
                input_messages=[{"role": "user", "content": "Hi"}],
                output_messages=[{"role": "assistant", "content": "Hello"}],
            )
            for i in range(5)
        ]
        # Give one trace many messages
        spans.append(
            create_llm_span(
                span_id="s-long-1",
                trace_id="t-long-1",
                input_messages=[
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"},
                ],
                output_messages=[{"role": "assistant", "content": "A2"}],
            )
        )
        spans.append(
            create_llm_span(
                span_id="s-long-2",
                trace_id="t-long-2",
                input_messages=[
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "Q2"},
                ],
                output_messages=[{"role": "assistant", "content": "A2"}],
            )
        )

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(spans)
        mock_client.site = "datadoghq.com"

        results = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            limit=1,
            min_messages=3,
        ):
            results.append(t)

        # Only 1 yielded despite 2 matching (limit=1)
        assert len(results) == 1


class TestBuildTranscriptEmptySpans:
    """Tests for _build_transcript with spans that get dropped."""

    @pytest.mark.asyncio
    async def test_all_empty_span_ids_returns_none(self) -> None:
        """Return None when all spans have empty span_id."""
        span: dict[str, Any] = {
            "span_id": "",
            "trace_id": "trace-1",
            "span_kind": "llm",
        }
        result = await _build_transcript([span], "my-app", "trace-1", "datadoghq.com")
        assert result is None


class TestDeduplicateBy:
    """Tests for session-based deduplication via deduplicate_by parameter."""

    pytestmark = pytest.mark.usefixtures("no_fallback_warnings")

    @pytest.mark.asyncio
    async def test_dedup_keeps_highest_tokens(self) -> None:
        """Two traces with the same session_id → only higher-token one yielded."""
        span_low = create_llm_span(
            span_id="s1",
            trace_id="t1",
            input_tokens=10,
            output_tokens=5,
            tags=["session_id:abc123"],
        )
        span_high = create_llm_span(
            span_id="s2",
            trace_id="t2",
            input_tokens=100,
            output_tokens=50,
            tags=["session_id:abc123"],
        )

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages([span_low, span_high])
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            deduplicate_by="session_id",
        ):
            results.append(t)

        assert len(results) == 1
        assert results[0].transcript_id == "t2"
        assert results[0].total_tokens == 150

    @pytest.mark.asyncio
    async def test_dedup_one_per_session(self) -> None:
        """3 traces in 2 sessions → 2 transcripts."""
        span1 = create_llm_span(
            span_id="s1",
            trace_id="t1",
            input_tokens=10,
            output_tokens=5,
            tags=["session_id:sess-a"],
        )
        span2 = create_llm_span(
            span_id="s2",
            trace_id="t2",
            input_tokens=20,
            output_tokens=10,
            tags=["session_id:sess-a"],
        )
        span3 = create_llm_span(
            span_id="s3",
            trace_id="t3",
            input_tokens=5,
            output_tokens=3,
            tags=["session_id:sess-b"],
        )

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages([span1, span2, span3])
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            deduplicate_by="session_id",
        ):
            results.append(t)

        assert len(results) == 2
        ids = {t.transcript_id for t in results}
        assert ids == {"t2", "t3"}

    @pytest.mark.asyncio
    async def test_dedup_passthrough_without_tag(self) -> None:
        """Traces missing the dedup tag are yielded immediately."""
        span_tagged = create_llm_span(
            span_id="s1",
            trace_id="t1",
            input_tokens=10,
            output_tokens=5,
            tags=["session_id:abc"],
        )
        span_untagged = create_llm_span(
            span_id="s2",
            trace_id="t2",
            input_tokens=10,
            output_tokens=5,
            tags=["env:prod"],
        )

        mock_client = AsyncMock()
        # Two pages so t1 completes on page boundary, t2 on flush
        mock_client.iter_span_pages = _pages([span_tagged], [span_untagged])
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            deduplicate_by="session_id",
        ):
            results.append(t)

        assert len(results) == 2
        ids = {t.transcript_id for t in results}
        assert ids == {"t1", "t2"}

    @pytest.mark.asyncio
    async def test_dedup_disabled_when_none(self) -> None:
        """deduplicate_by=None → all traces yielded (existing behavior)."""
        span1 = create_llm_span(
            span_id="s1",
            trace_id="t1",
            tags=["session_id:abc"],
        )
        span2 = create_llm_span(
            span_id="s2",
            trace_id="t2",
            tags=["session_id:abc"],
        )

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages([span1, span2])
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            deduplicate_by=None,
        ):
            results.append(t)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_dedup_with_limit(self) -> None:
        """Limit applies to total output including dedup winners."""
        spans = [
            create_llm_span(
                span_id=f"s{i}",
                trace_id=f"t{i}",
                input_tokens=10 * (i + 1),
                output_tokens=5,
                tags=[f"session_id:sess-{i}"],
            )
            for i in range(5)
        ]

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages(spans)
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            limit=2,
            deduplicate_by="session_id",
        ):
            results.append(t)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_dedup_none_tokens_treated_as_zero(self) -> None:
        """total_tokens=None loses to any positive value."""
        # Span with zero token counts (transcript will have total_tokens=0)
        span_none = create_llm_span(
            span_id="s1",
            trace_id="t1",
            input_tokens=0,
            output_tokens=0,
            tags=["session_id:abc"],
        )
        span_some = create_llm_span(
            span_id="s2",
            trace_id="t2",
            input_tokens=1,
            output_tokens=1,
            tags=["session_id:abc"],
        )

        mock_client = AsyncMock()
        mock_client.iter_span_pages = _pages([span_none, span_some])
        mock_client.site = "datadoghq.com"

        results: list[Transcript] = []
        async for t in _from_query(
            mock_client,
            "app",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            deduplicate_by="session_id",
        ):
            results.append(t)

        assert len(results) == 1
        assert results[0].transcript_id == "t2"

    def test_get_tag_value(self) -> None:
        """Unit test of the tag extraction helper."""
        transcript = Transcript(
            transcript_id="t1",
            metadata={"tags": ["session_id:abc123", "env:prod", "ml_app:my-app"]},
        )
        assert _get_tag_value(transcript, "session_id") == "abc123"
        assert _get_tag_value(transcript, "env") == "prod"
        assert _get_tag_value(transcript, "missing") is None

        # No tags in metadata
        transcript_no_tags = Transcript(
            transcript_id="t2",
            metadata={},
        )
        assert _get_tag_value(transcript_no_tags, "session_id") is None

        # Missing metadata (defaults to empty dict)
        transcript_default = Transcript(
            transcript_id="t3",
        )
        assert _get_tag_value(transcript_default, "session_id") is None
