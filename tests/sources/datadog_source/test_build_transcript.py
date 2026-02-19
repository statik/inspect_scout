"""Tests for _build_transcript and datadog() with mocked client."""

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from inspect_scout.sources._datadog import (
    _build_transcript,
    _extract_metadata,
    _extract_model_options,
    _extract_root_messages,
    _from_query,
    _get_ml_app_from_tags,
    _root_duration,
    datadog,
)
from inspect_scout.sources._datadog.client import DATADOG_SOURCE_TYPE

from .mocks import (
    _BASE_NS,
    _SECOND_NS,
    create_agent_span,
    create_llm_span,
    create_tool_call_trace,
)


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
            duration=5 * _SECOND_NS,
            start_ns=_BASE_NS,
        )
        child1 = create_llm_span(
            span_id="child-1",
            trace_id="trace-1",
            parent_id="root",
            duration=2 * _SECOND_NS,
            start_ns=_BASE_NS + _SECOND_NS,
        )
        child2 = create_llm_span(
            span_id="child-2",
            trace_id="trace-1",
            parent_id="root",
            duration=2 * _SECOND_NS,
            start_ns=_BASE_NS + 3 * _SECOND_NS,
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
    async def test_error_span_captured(self) -> None:
        """Transcript captures error from span with error status."""
        span = create_llm_span(trace_id="trace-err", status="error")
        span["meta"]["error"] = {"message": "Rate limited"}
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
        """Return None when meta.metadata is absent."""
        span: dict[str, Any] = {"meta": {}}
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
        mock_client.list_spans = AsyncMock(return_value=spans)
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
        mock_client.list_spans = AsyncMock(return_value=spans)
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
        mock_client.list_spans = AsyncMock(side_effect=ValueError("Bad request"))
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
        mock_client.list_spans = AsyncMock(return_value=spans)
        mock_client.site = "datadoghq.com"
        mock_client.aclose = AsyncMock()

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
        mock_client.list_spans = AsyncMock(return_value=spans)
        mock_client.site = "datadoghq.com"
        mock_client.aclose = AsyncMock()

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
            "meta": {
                "input": {"value": "Hello"},
                "output": {"value": "Hi there!"},
            },
        }
        messages = _extract_root_messages(span)
        assert len(messages) == 2
        assert messages[0].content == "Hello"
        assert messages[1].content == "Hi there!"

    def test_input_only(self) -> None:
        """Extract only input when output is missing."""
        span: dict[str, Any] = {
            "meta": {
                "input": {"value": "Hello"},
                "output": {},
            },
        }
        messages = _extract_root_messages(span)
        assert len(messages) == 1
        assert messages[0].content == "Hello"

    def test_empty_meta(self) -> None:
        """Return empty list when meta has no input/output."""
        span: dict[str, Any] = {"meta": {}}
        messages = _extract_root_messages(span)
        assert messages == []

    def test_no_meta_key(self) -> None:
        """Return empty list when span has no meta."""
        span: dict[str, Any] = {}
        messages = _extract_root_messages(span)
        assert messages == []


class TestRootDuration:
    """Tests for _root_duration function."""

    def test_valid_duration(self) -> None:
        """Convert nanosecond duration to seconds."""
        span: dict[str, Any] = {"duration": 5_000_000_000}
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
            "meta": {"metadata": {"success": True}},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is True

    def test_string_success_coerced_to_bool(self) -> None:
        """String 'true' is coerced to bool True."""
        span: dict[str, Any] = {
            "meta": {"metadata": {"success": "true"}},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is True

    def test_string_false_coerced_to_false(self) -> None:
        """String 'false' is coerced to bool False."""
        span: dict[str, Any] = {
            "meta": {"metadata": {"success": "false"}},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is False

    def test_string_False_coerced_to_false(self) -> None:
        """String 'False' is coerced to bool False."""
        span: dict[str, Any] = {
            "meta": {"metadata": {"success": "False"}},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is False

    def test_int_success_coerced_to_bool(self) -> None:
        """Integer 1 is coerced to bool True."""
        span: dict[str, Any] = {
            "meta": {"metadata": {"success": 1}},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is True

    def test_zero_success_coerced_to_false(self) -> None:
        """Integer 0 is coerced to bool False."""
        span: dict[str, Any] = {
            "meta": {"metadata": {"success": 0}},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is False

    def test_none_success_preserved(self) -> None:
        """None success value is preserved."""
        span: dict[str, Any] = {
            "meta": {"metadata": {"success": None}},
            "tags": [],
        }
        metadata = _extract_metadata(span)
        assert metadata["success"] is None


class TestFromQueryForwardsLimit:
    """Tests that _from_query forwards a span limit to list_spans."""

    pytestmark = pytest.mark.usefixtures("no_fallback_warnings")

    @pytest.mark.asyncio
    async def test_limit_forwarded_to_list_spans(self) -> None:
        """_from_query passes heuristic span limit to client.list_spans."""
        mock_client = AsyncMock()
        mock_client.list_spans = AsyncMock(return_value=[])
        mock_client.site = "datadoghq.com"

        async for _ in _from_query(
            mock_client, "app", None, None, None, None, None, None, 5
        ):
            pass

        call_kwargs = mock_client.list_spans.call_args.kwargs
        assert call_kwargs["limit"] == 250  # 5 * 50

    @pytest.mark.asyncio
    async def test_limit_capped_at_10000(self) -> None:
        """Span limit is capped at 10,000."""
        mock_client = AsyncMock()
        mock_client.list_spans = AsyncMock(return_value=[])
        mock_client.site = "datadoghq.com"

        async for _ in _from_query(
            mock_client, "app", None, None, None, None, None, None, 500
        ):
            pass

        call_kwargs = mock_client.list_spans.call_args.kwargs
        assert call_kwargs["limit"] == 10_000

    @pytest.mark.asyncio
    async def test_no_limit_passes_none(self) -> None:
        """No limit passes None to list_spans."""
        mock_client = AsyncMock()
        mock_client.list_spans = AsyncMock(return_value=[])
        mock_client.site = "datadoghq.com"

        async for _ in _from_query(
            mock_client, "app", None, None, None, None, None, None, None
        ):
            pass

        call_kwargs = mock_client.list_spans.call_args.kwargs
        assert call_kwargs["limit"] is None
