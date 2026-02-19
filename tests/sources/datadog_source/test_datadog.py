"""Integration tests for Datadog LLM Observability adapter.

These tests require real Datadog credentials and are skipped unless
DATADOG_RUN_TESTS=1 is set.
"""

import pytest

from .conftest import skip_if_no_datadog


@skip_if_no_datadog
class TestDatadogIntegration:
    """Integration tests for the Datadog adapter."""

    @pytest.mark.asyncio
    async def test_fetch_transcripts(self, datadog_client: None) -> None:
        """Fetch transcripts from Datadog."""
        from inspect_scout.sources._datadog import datadog

        count = 0
        async for transcript in datadog(limit=1):
            assert transcript.transcript_id
            assert transcript.source_type == "datadog"
            count += 1

        assert count <= 1

    @pytest.mark.asyncio
    async def test_fetch_with_ml_app(self, datadog_client: None) -> None:
        """Fetch transcripts filtered by ml_app."""
        import os

        from inspect_scout.sources._datadog import datadog

        ml_app = os.environ.get("DD_TEST_ML_APP")
        if not ml_app:
            pytest.skip("DD_TEST_ML_APP not set")

        async for transcript in datadog(ml_app=ml_app, limit=1):
            assert transcript.source_type == "datadog"
            break

    @pytest.mark.asyncio
    async def test_transcript_has_events(self, datadog_client: None) -> None:
        """Fetched transcripts should have events."""
        from inspect_scout.sources._datadog import datadog

        async for transcript in datadog(limit=1):
            assert len(transcript.events) > 0 or len(transcript.messages) > 0
            break
