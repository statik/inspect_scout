"""Tests for Datadog client creation and configuration."""

import pytest

from inspect_scout.sources._datadog.client import (
    DATADOG_SOURCE_TYPE,
    get_datadog_client,
)


class TestDatadogSourceType:
    """Tests for source type constant."""

    def test_source_type_value(self) -> None:
        """Source type constant is 'datadog'."""
        assert DATADOG_SOURCE_TYPE == "datadog"


class TestGetDatadogClient:
    """Tests for get_datadog_client function."""

    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise ValueError when no API key provided."""
        monkeypatch.delenv("DD_API_KEY", raising=False)
        monkeypatch.delenv("DD_APP_KEY", raising=False)

        with pytest.raises(ValueError, match="API key"):
            get_datadog_client()

    def test_missing_app_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Raise ValueError when no application key provided."""
        monkeypatch.delenv("DD_APP_KEY", raising=False)

        with pytest.raises(ValueError, match="application key"):
            get_datadog_client(api_key="test-api-key")

    def test_client_created_with_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Create client successfully with explicit keys."""
        monkeypatch.delenv("DD_API_KEY", raising=False)
        monkeypatch.delenv("DD_APP_KEY", raising=False)
        monkeypatch.delenv("DD_SITE", raising=False)

        client = get_datadog_client(
            api_key="test-api-key",
            app_key="test-app-key",
            site="datadoghq.eu",
        )

        assert client.site == "datadoghq.eu"

    def test_default_site(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default site is datadoghq.com."""
        monkeypatch.delenv("DD_SITE", raising=False)

        client = get_datadog_client(
            api_key="test-api-key",
            app_key="test-app-key",
        )

        assert client.site == "datadoghq.com"

    def test_env_var_site(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Use DD_SITE environment variable."""
        monkeypatch.setenv("DD_SITE", "us5.datadoghq.com")

        client = get_datadog_client(
            api_key="test-api-key",
            app_key="test-app-key",
        )

        assert client.site == "us5.datadoghq.com"
