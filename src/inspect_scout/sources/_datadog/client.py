"""Datadog LLM Observability client and retry logic.

Uses the Datadog Export API (``/api/v2/llm-obs/v1/spans/events``)
with httpx for HTTP calls. No SDK dependency required.
"""

import os
from dataclasses import dataclass
from logging import getLogger
from typing import Any, Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
)

logger = getLogger(__name__)

T = TypeVar("T")

DATADOG_SOURCE_TYPE = "datadog"

RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})

RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_MIN_WAIT = 2  # seconds
RATE_LIMIT_MAX_WAIT = 60  # seconds

# DD Export API base path
_EXPORT_PATH = "/api/v2/llm-obs/v1/spans/events"


@dataclass
class DatadogClient:
    """HTTP client for the Datadog LLM Observability Export API.

    Attributes:
        http: Configured httpx.AsyncClient with auth headers
        site: Datadog site identifier (e.g. ``datadoghq.com``)
    """

    http: Any  # httpx.AsyncClient
    site: str

    async def list_spans(
        self,
        ml_app: str | None = None,
        from_time: Any | None = None,
        to_time: Any | None = None,
        trace_id: str | None = None,
        span_kind: str | None = None,
        span_name: str | None = None,
        tags: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch spans from the DD LLM Observability Export API.

        Handles cursor-based pagination and unwraps the response envelope.

        Args:
            ml_app: Filter by ml_app tag
            from_time: Only spans created on or after this time
            to_time: Only spans created before this time
            trace_id: Fetch specific trace by ID
            span_kind: Filter by span kind (llm, tool, agent, etc.)
            span_name: Filter by span name
            tags: Additional tag filters (``key:value`` format)
            limit: Maximum number of spans to return

        Returns:
            List of unwrapped span dictionaries
        """
        params: dict[str, Any] = {}
        if ml_app:
            params["filter[ml_app]"] = ml_app
        if from_time:
            params["filter[from]"] = str(from_time.isoformat())
        if to_time:
            params["filter[to]"] = str(to_time.isoformat())
        if trace_id:
            params["filter[trace_id]"] = trace_id
        if span_kind:
            params["filter[span_kind]"] = span_kind
        if span_name:
            params["filter[span_name]"] = span_name
        if tags:
            params["filter[tags]"] = ",".join(tags)

        all_spans: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            page_params = dict(params)
            if cursor:
                page_params["page[cursor]"] = cursor

            async def _fetch(p: dict[str, Any] = page_params) -> Any:
                response = await self.http.get(_EXPORT_PATH, params=p)
                response.raise_for_status()
                return response.json()

            result = await retry_api_call_async(_fetch)

            data = result.get("data", [])
            for item in data:
                attrs = item.get("attributes", {})
                all_spans.append(attrs)

            if limit and len(all_spans) >= limit:
                all_spans = all_spans[:limit]
                break

            next_cursor = result.get("meta", {}).get("page", {}).get("after")
            if not next_cursor or not data:
                break
            cursor = next_cursor

        return all_spans

    async def aclose(self) -> None:
        """Close the underlying HTTP client."""
        await self.http.aclose()


def get_datadog_client(
    api_key: str | None = None,
    app_key: str | None = None,
    site: str | None = None,
) -> DatadogClient:
    """Create a Datadog client for the LLM Observability Export API.

    Resolves credentials in this order:
    1. Explicit parameters
    2. ``DD_API_KEY`` / ``DD_APP_KEY`` / ``DD_SITE`` environment variables

    Args:
        api_key: Datadog API key
        app_key: Datadog application key
        site: Datadog site (e.g. ``datadoghq.com``, ``datadoghq.eu``)

    Returns:
        Configured DatadogClient

    Raises:
        ImportError: If httpx is not installed
        ValueError: If required credentials are missing
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError(
            "The httpx package is required for Datadog import. "
            "Install it with: pip install httpx"
        ) from e

    resolved_api_key = api_key or os.environ.get("DD_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Datadog API key is required. Provide api_key parameter "
            "or set DD_API_KEY environment variable."
        )

    resolved_app_key = app_key or os.environ.get("DD_APP_KEY")
    if not resolved_app_key:
        raise ValueError(
            "Datadog application key is required. Provide app_key parameter "
            "or set DD_APP_KEY environment variable."
        )

    resolved_site = site or os.environ.get("DD_SITE", "datadoghq.com")
    base_url = f"https://api.{resolved_site}"

    http_client = httpx.AsyncClient(
        base_url=base_url,
        headers={
            "DD-API-KEY": resolved_api_key,
            "DD-APPLICATION-KEY": resolved_app_key,
            "Accept": "application/json",
        },
        timeout=httpx.Timeout(30.0),
    )

    return DatadogClient(http=http_client, site=resolved_site)


def _is_retryable_error(exception: BaseException) -> bool:
    """Check if an exception is retryable (timeout, rate limit, server error).

    Args:
        exception: The exception to check

    Returns:
        True if the error is transient and should be retried
    """
    try:
        import httpx

        if isinstance(exception, (httpx.TimeoutException, httpx.ConnectError)):
            return True
        if isinstance(exception, httpx.HTTPStatusError):
            return exception.response.status_code in RETRYABLE_HTTP_CODES
    except ImportError:
        pass

    exc_name = type(exception).__name__
    if "Timeout" in exc_name or "ConnectionError" in exc_name:
        return True

    exc_str = str(exception).lower()
    if "rate limit" in exc_str or "429" in exc_str or "too many requests" in exc_str:
        return True

    return False


def _is_rate_limit_error(exception: BaseException) -> bool:
    """Check if an exception is specifically a rate limit error (HTTP 429).

    Args:
        exception: The exception to check

    Returns:
        True if this is a rate limit error that needs longer backoff
    """
    try:
        import httpx

        if isinstance(exception, httpx.HTTPStatusError):
            return exception.response.status_code == 429
    except ImportError:
        pass

    exc_str = str(exception).lower()
    return "rate limit" in exc_str or "429" in exc_str or "too many requests" in exc_str


def _get_retry_after(exception: BaseException) -> float | None:
    """Extract Retry-After header value from rate limit error if available.

    Args:
        exception: The exception to check

    Returns:
        Seconds to wait, or None if not available
    """
    try:
        import httpx

        if isinstance(exception, httpx.HTTPStatusError):
            retry_after = exception.response.headers.get("Retry-After")
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass
    except ImportError:
        pass
    return None


async def retry_api_call_async(func: Callable[[], Any]) -> Any:
    """Execute a Datadog API call with retry logic for transient errors.

    Uses adaptive retry strategy:
    - For rate limits (429): Up to 5 attempts with longer backoff (2-60s)
    - Respects Retry-After header when present
    - For other errors: 5 attempts with exponential backoff (1-30s)

    Args:
        func: Zero-argument async callable that makes the API call

    Returns:
        The result of the API call

    Raises:
        The original exception if all retries fail or error is not retryable
    """

    def _log_retry(retry_state: Any) -> None:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        exc_name = type(exc).__name__ if exc else "Unknown"
        sleep_time = retry_state.next_action.sleep if retry_state.next_action else 0
        is_rate_limit = _is_rate_limit_error(exc) if exc else False
        error_type = "rate limited" if is_rate_limit else "failed"
        logger.warning(
            f"Datadog API call {error_type} ({exc_name}), "
            f"retrying in {sleep_time:.1f}s... "
            f"(attempt {retry_state.attempt_number}/{RATE_LIMIT_MAX_ATTEMPTS})"
        )

    def _wait_with_rate_limit_handling(retry_state: Any) -> float:
        """Calculate wait time, respecting Retry-After for rate limits."""
        exc = retry_state.outcome.exception() if retry_state.outcome else None

        if exc and _is_rate_limit_error(exc):
            retry_after = _get_retry_after(exc)
            if retry_after:
                wait_time = min(retry_after, float(RATE_LIMIT_MAX_WAIT))
                logger.info(f"Rate limit: waiting {wait_time}s per Retry-After header")
                return wait_time
            attempt: int = retry_state.attempt_number
            return float(
                min(
                    RATE_LIMIT_MIN_WAIT * (2 ** (attempt - 1)),
                    RATE_LIMIT_MAX_WAIT,
                )
            )

        attempt_num: int = retry_state.attempt_number
        return float(min(1 * (2 ** (attempt_num - 1)), 30))

    @retry(
        retry=retry_if_exception(_is_retryable_error),
        stop=stop_after_attempt(RATE_LIMIT_MAX_ATTEMPTS),
        wait=_wait_with_rate_limit_handling,
        before_sleep=_log_retry,
        reraise=True,
    )
    async def _call_with_retry() -> Any:
        return await func()

    return await _call_with_retry()
