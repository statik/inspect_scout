"""Tests for Datadog client pagination and retry logic."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from inspect_scout.sources._datadog.client import (
    DatadogClient,
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
