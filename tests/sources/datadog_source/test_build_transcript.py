"""Tests for _build_transcript and datadog() with mocked client."""

from unittest.mock import AsyncMock, patch

import pytest
from inspect_scout.sources._datadog import _build_transcript, _from_query, datadog
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

        monkeypatch.setattr(datadog_pkg.logger, "warning", _capture_warning)

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
