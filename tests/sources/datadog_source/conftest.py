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
def fallback_warnings(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Capture fallback warnings from extraction during tests.

    Monkeypatches logger.warning in extraction and __init__ modules to
    record calls. Does NOT assert — use this in tests that expect warnings.

    Yields:
        List of captured warning messages.
    """
    warnings: list[str] = []

    def _capture_warning(msg: str, *args: Any, **kwargs: Any) -> None:
        warnings.append(msg % args if args else msg)

    import inspect_scout.sources._datadog as datadog_pkg
    import inspect_scout.sources._datadog.extraction as datadog_extraction

    monkeypatch.setattr(datadog_extraction.logger, "warning", _capture_warning)
    monkeypatch.setattr(datadog_pkg.logger, "warning", _capture_warning)

    yield warnings


@pytest.fixture
def no_fallback_warnings(fallback_warnings: list[str]) -> Any:
    """Assert no fallback warnings from extraction during tests.

    Wraps ``fallback_warnings`` and asserts the list is empty on teardown.

    Yields:
        List of captured warning messages (empty if native path succeeded).
    """
    yield fallback_warnings

    assert not fallback_warnings, (
        f"Extraction fell back to simple conversion: {fallback_warnings}"
    )


@pytest.fixture
def datadog_client() -> None:
    """Verify Datadog credentials are available for integration tests.

    Skips the test if httpx is not installed or credentials are missing.
    Does not create an actual client — integration tests use datadog()
    which manages its own client lifecycle.
    """
    try:
        import httpx  # noqa: F401
    except ImportError:
        pytest.skip("httpx package not installed")

    if not os.environ.get("DD_API_KEY"):
        pytest.skip("DD_API_KEY not set")
    if not os.environ.get("DD_APP_KEY"):
        pytest.skip("DD_APP_KEY not set")
