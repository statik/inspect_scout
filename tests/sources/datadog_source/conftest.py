"""Datadog test configuration and fixtures.

Provides skip decorators and fixtures for Datadog integration tests.

Authentication:
    Set DD_API_KEY, DD_APP_KEY, and optionally DD_SITE environment variables.
"""

import os
from typing import Any, Callable, TypeVar, cast

import pytest

F = TypeVar("F", bound=Callable[..., Any])


def skip_if_no_datadog(func: F) -> F:
    """Skip test if Datadog test environment is not configured.

    Requires:
    - DATADOG_RUN_TESTS=1 (explicit opt-in)

    Args:
        func: Test function to decorate

    Returns:
        Decorated function with skip marker
    """
    run_tests = os.environ.get("DATADOG_RUN_TESTS", "").lower() in ("1", "true")

    return cast(
        F,
        pytest.mark.api(
            pytest.mark.skipif(
                not run_tests,
                reason="Datadog tests require DATADOG_RUN_TESTS=1",
            )(func)
        ),
    )


@pytest.fixture
def no_fallback_warnings(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Assert no fallback warnings from extraction during tests.

    Monkeypatches logger.warning in extraction and __init__ modules to
    record calls. After the test, asserts no warnings were emitted.

    Yields:
        List of captured warning messages (empty if native path succeeded).
    """
    warnings: list[str] = []

    def _capture_warning(msg: str, *args: Any, **kwargs: Any) -> None:
        warnings.append(msg % args if args else msg)

    import inspect_scout.sources._datadog as datadog_pkg
    import inspect_scout.sources._datadog.extraction as datadog_extraction

    monkeypatch.setattr(datadog_extraction.logger, "warning", _capture_warning)
    monkeypatch.setattr(datadog_pkg.logger, "warning", _capture_warning)

    yield warnings

    assert not warnings, f"Extraction fell back to simple conversion: {warnings}"


@pytest.fixture
def datadog_client() -> Any:
    """Create a Datadog client for testing.

    Uses DD_API_KEY, DD_APP_KEY, and DD_SITE environment variables.

    Returns:
        DatadogClient instance

    Raises:
        pytest.skip: If credentials are not set
    """
    try:
        from inspect_scout.sources._datadog.client import get_datadog_client
    except ImportError:
        pytest.skip("httpx package not installed")

    try:
        return get_datadog_client()
    except ValueError as e:
        pytest.skip(str(e))
