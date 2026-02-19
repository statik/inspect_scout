"""Tests for Datadog client pagination and retry logic."""

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

httpx = pytest.importorskip("httpx")

from inspect_scout.sources._datadog.client import (  # noqa: E402
    DatadogClient,
    _get_retry_after,
    _is_rate_limit_error,
    _is_retryable_error,
    retry_api_call_async,
)


class TestIsRetryableError:
    """Tests for _is_retryable_error function."""

    def test_timeout_is_retryable(self) -> None:
        """httpx.TimeoutException is retryable."""
        exc = httpx.TimeoutException("timed out")
        assert _is_retryable_error(exc) is True

    def test_connect_error_is_retryable(self) -> None:
        """httpx.ConnectError is retryable."""
        exc = httpx.ConnectError("connection refused")
        assert _is_retryable_error(exc) is True

    def test_429_is_retryable(self) -> None:
        """HTTP 429 is retryable."""
        response = httpx.Response(429, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )
        assert _is_retryable_error(exc) is True

    def test_500_is_retryable(self) -> None:
        """HTTP 500 is retryable."""
        response = httpx.Response(500, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError(
            "server error", request=response.request, response=response
        )
        assert _is_retryable_error(exc) is True

    def test_400_is_not_retryable(self) -> None:
        """HTTP 400 is not retryable."""
        response = httpx.Response(400, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError(
            "bad request", request=response.request, response=response
        )
        assert _is_retryable_error(exc) is False

    def test_403_is_not_retryable(self) -> None:
        """HTTP 403 is not retryable."""
        response = httpx.Response(403, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError(
            "forbidden", request=response.request, response=response
        )
        assert _is_retryable_error(exc) is False

    def test_value_error_is_not_retryable(self) -> None:
        """ValueError is not retryable."""
        assert _is_retryable_error(ValueError("bad")) is False


class TestIsRateLimitError:
    """Tests for _is_rate_limit_error function."""

    def test_429_is_rate_limit(self) -> None:
        """HTTP 429 is a rate limit error."""
        response = httpx.Response(429, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )
        assert _is_rate_limit_error(exc) is True

    def test_500_is_not_rate_limit(self) -> None:
        """HTTP 500 is not a rate limit error."""
        response = httpx.Response(500, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError(
            "server error", request=response.request, response=response
        )
        assert _is_rate_limit_error(exc) is False


class TestGetRetryAfter:
    """Tests for _get_retry_after function."""

    def test_extracts_retry_after_header(self) -> None:
        """Extract Retry-After seconds from response header."""
        response = httpx.Response(
            429,
            request=httpx.Request("GET", "https://x"),
            headers={"Retry-After": "30"},
        )
        exc = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )
        assert _get_retry_after(exc) == 30.0

    def test_returns_none_without_header(self) -> None:
        """Return None when Retry-After header is missing."""
        response = httpx.Response(429, request=httpx.Request("GET", "https://x"))
        exc = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )
        assert _get_retry_after(exc) is None

    def test_returns_none_for_non_numeric_header(self) -> None:
        """Return None when Retry-After header is not numeric."""
        response = httpx.Response(
            429,
            request=httpx.Request("GET", "https://x"),
            headers={"Retry-After": "Wed, 21 Oct 2025 07:28:00 GMT"},
        )
        exc = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )
        assert _get_retry_after(exc) is None

    def test_returns_none_for_non_http_error(self) -> None:
        """Return None for non-HTTP exceptions."""
        assert _get_retry_after(ValueError("bad")) is None


class TestRetryApiCallAsync:
    """Tests for retry_api_call_async function."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self) -> None:
        """Return result when call succeeds immediately."""
        func = AsyncMock(return_value={"data": []})
        result = await retry_api_call_async(func)
        assert result == {"data": []}
        func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_server_error(self) -> None:
        """Retry on 500 and succeed on second attempt."""
        response = httpx.Response(500, request=httpx.Request("GET", "https://x"))
        error = httpx.HTTPStatusError(
            "server error", request=response.request, response=response
        )

        func = AsyncMock(side_effect=[error, {"data": []}])
        result = await retry_api_call_async(func)
        assert result == {"data": []}
        assert func.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_429_with_retry_after_header(self) -> None:
        """Retry on 429 and respect the Retry-After header."""
        response = httpx.Response(
            429,
            request=httpx.Request("GET", "https://x"),
            headers={"Retry-After": "1"},
        )
        error = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )

        func = AsyncMock(side_effect=[error, {"data": []}])
        with patch("asyncio.sleep") as mock_sleep:
            result = await retry_api_call_async(func)
        assert result == {"data": []}
        assert func.await_count == 2
        mock_sleep.assert_awaited_once_with(1.0)

    @pytest.mark.asyncio
    async def test_retries_on_429_with_retry_after_zero(self) -> None:
        """Retry on 429 with Retry-After: 0 (immediate retry)."""
        response = httpx.Response(
            429,
            request=httpx.Request("GET", "https://x"),
            headers={"Retry-After": "0"},
        )
        error = httpx.HTTPStatusError(
            "rate limited", request=response.request, response=response
        )

        func = AsyncMock(side_effect=[error, {"data": []}])
        with patch("asyncio.sleep") as mock_sleep:
            result = await retry_api_call_async(func)
        assert result == {"data": []}
        assert func.await_count == 2
        mock_sleep.assert_awaited_once_with(0.0)

    @pytest.mark.asyncio
    async def test_non_retryable_error_raises_immediately(self) -> None:
        """Non-retryable errors raise without retry."""
        response = httpx.Response(403, request=httpx.Request("GET", "https://x"))
        error = httpx.HTTPStatusError(
            "forbidden", request=response.request, response=response
        )

        func = AsyncMock(side_effect=error)
        with pytest.raises(httpx.HTTPStatusError):
            await retry_api_call_async(func)
        func.assert_awaited_once()


class TestPagination:
    """Tests for cursor-based pagination in list_spans."""

    @pytest.mark.asyncio
    async def test_single_page(self) -> None:
        """Fetch spans from single page response."""
        response_data = {
            "data": [
                {"attributes": {"span_id": "s1", "trace_id": "t1"}},
                {"attributes": {"span_id": "s2", "trace_id": "t1"}},
            ],
            "meta": {"page": {}},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        client = DatadogClient(http=mock_http, site="datadoghq.com")
        spans = await client.list_spans(ml_app="test-app")

        assert len(spans) == 2
        assert spans[0]["span_id"] == "s1"

    @pytest.mark.asyncio
    async def test_multiple_pages(self) -> None:
        """Fetch spans across multiple pages using cursor."""
        page1 = {
            "data": [{"attributes": {"span_id": "s1"}}],
            "meta": {"page": {"after": "cursor-abc"}},
        }
        page2 = {
            "data": [{"attributes": {"span_id": "s2"}}],
            "meta": {"page": {}},
        }

        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = page1
        mock_response_1.raise_for_status = MagicMock()

        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = page2
        mock_response_2.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=[mock_response_1, mock_response_2])

        client = DatadogClient(http=mock_http, site="datadoghq.com")
        spans = await client.list_spans()

        assert len(spans) == 2
        assert mock_http.get.await_count == 2

        second_call_params = mock_http.get.call_args_list[1][1]["params"]
        assert "page[cursor]" in second_call_params
        assert second_call_params["page[cursor]"] == "cursor-abc"

    @pytest.mark.asyncio
    async def test_pagination_stops_on_empty_data(self) -> None:
        """Stop paginating when data array is empty."""
        page1 = {
            "data": [{"attributes": {"span_id": "s1"}}],
            "meta": {"page": {"after": "cursor-abc"}},
        }
        page2: dict[str, Any] = {
            "data": [],
            "meta": {"page": {"after": "cursor-def"}},
        }

        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = page1
        mock_response_1.raise_for_status = MagicMock()

        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = page2
        mock_response_2.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(side_effect=[mock_response_1, mock_response_2])

        client = DatadogClient(http=mock_http, site="datadoghq.com")
        spans = await client.list_spans()

        assert len(spans) == 1
        assert mock_http.get.await_count == 2

    @pytest.mark.asyncio
    async def test_limit_truncates_results(self) -> None:
        """Respect limit parameter and truncate results."""
        response_data = {
            "data": [
                {"attributes": {"span_id": "s1"}},
                {"attributes": {"span_id": "s2"}},
                {"attributes": {"span_id": "s3"}},
            ],
            "meta": {"page": {"after": "cursor-abc"}},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        client = DatadogClient(http=mock_http, site="datadoghq.com")
        spans = await client.list_spans(limit=2)

        assert len(spans) == 2

    @pytest.mark.asyncio
    async def test_pagination_retries_on_transient_error(self) -> None:
        """Second page fails with 500, retries, and succeeds."""
        page1 = {
            "data": [{"attributes": {"span_id": "s1"}}],
            "meta": {"page": {"after": "cursor-abc"}},
        }
        page2 = {
            "data": [{"attributes": {"span_id": "s2"}}],
            "meta": {"page": {}},
        }

        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = page1
        mock_response_1.raise_for_status = MagicMock()

        error_response = httpx.Response(500, request=httpx.Request("GET", "https://x"))
        server_error = httpx.HTTPStatusError(
            "server error",
            request=error_response.request,
            response=error_response,
        )

        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = page2
        mock_response_2.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(
            side_effect=[mock_response_1, server_error, mock_response_2]
        )

        client = DatadogClient(http=mock_http, site="datadoghq.com")
        spans = await client.list_spans()

        assert len(spans) == 2
        assert spans[0]["span_id"] == "s1"
        assert spans[1]["span_id"] == "s2"
        assert mock_http.get.await_count == 3

    @pytest.mark.asyncio
    async def test_query_params_construction(self) -> None:
        """Verify filter params are sent correctly for all query options."""
        empty_response = {
            "data": [],
            "meta": {"page": {}},
        }

        mock_response = MagicMock()
        mock_response.json.return_value = empty_response
        mock_response.raise_for_status = MagicMock()

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_response)

        client = DatadogClient(http=mock_http, site="datadoghq.com")

        from_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        to_time = datetime(2024, 1, 16, 10, 0, 0, tzinfo=timezone.utc)

        await client.list_spans(
            ml_app="my-app",
            from_time=from_time,
            to_time=to_time,
            trace_id="trace-abc",
            span_kind="llm",
            span_name="chat gpt-4o",
            tags=["env:prod", "version:2"],
        )

        params = mock_http.get.call_args_list[0].kwargs["params"]
        assert params["filter[ml_app]"] == "my-app"
        assert params["filter[from]"] == from_time.isoformat()
        assert params["filter[to]"] == to_time.isoformat()
        assert params["filter[trace_id]"] == "trace-abc"
        assert params["filter[span_kind]"] == "llm"
        assert params["filter[span_name]"] == "chat gpt-4o"
        assert params["filter[tags]"] == "env:prod,version:2"
