"""Datadog LLM Observability transcript import functionality.

This module provides functions to import transcripts from Datadog LLM
Observability into an Inspect Scout transcript database. Uses the Datadog
Export API (``/api/v2/llm-obs/v1/spans/events``) with httpx.

Authentication:
    Set ``DD_API_KEY`` and ``DD_APP_KEY`` environment variables, or pass
    ``api_key`` and ``app_key`` parameters. Optionally set ``DD_SITE``
    (defaults to ``datadoghq.com``).
"""

import os
from collections import defaultdict
from datetime import datetime, timezone
from logging import getLogger
from typing import Any, AsyncIterator

from inspect_ai.event import ModelEvent
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    stable_message_ids,
)

from inspect_scout._transcript.types import Transcript

from .client import (
    DATADOG_SOURCE_TYPE,
    DatadogClient,
    get_datadog_client,
)
from .detection import detect_provider
from .events import spans_to_events
from .extraction import sum_tokens
from .tree import build_span_tree, flatten_tree_chronological, get_llm_spans

logger = getLogger(__name__)


async def datadog(
    ml_app: str | None = None,
    from_time: datetime | None = None,
    to_time: datetime | None = None,
    trace_id: str | None = None,
    span_kind: str | None = None,
    span_name: str | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
    api_key: str | None = None,
    app_key: str | None = None,
    site: str | None = None,
) -> AsyncIterator[Transcript]:
    """Read transcripts from Datadog LLM Observability traces.

    Each Datadog trace (collection of spans with same trace_id) becomes one
    Scout transcript. Child spans (LLM calls, tools) become events within
    the transcript.

    Args:
        ml_app: Datadog ml_app name to filter by
        from_time: Only fetch traces created on or after this time
        to_time: Only fetch traces created before this time
        trace_id: Fetch a specific trace by ID
        span_kind: Filter by span kind (llm, tool, agent, etc.)
        span_name: Filter by span name
        tags: Additional tag filters (``key:value`` format)
        limit: Maximum number of transcripts to fetch
        api_key: Datadog API key (or ``DD_API_KEY`` env var)
        app_key: Datadog application key (or ``DD_APP_KEY`` env var)
        site: Datadog site (or ``DD_SITE`` env var, defaults to
            ``datadoghq.com``)

    Yields:
        Transcript objects ready for insertion into transcript database

    Raises:
        ImportError: If httpx package is not installed
        ValueError: If required credentials are missing

    Environment:
        DATADOG_STRICT_IMPORT: Set to ``1`` or ``true`` to propagate
            per-trace processing errors instead of logging and skipping.
            Useful for debugging import failures.
    """
    client = get_datadog_client(api_key, app_key, site)

    try:
        async for transcript in _from_query(
            client,
            ml_app,
            from_time,
            to_time,
            trace_id,
            span_kind,
            span_name,
            tags,
            limit,
        ):
            yield transcript
    finally:
        await client.aclose()


async def _from_query(
    client: DatadogClient,
    ml_app: str | None,
    from_time: datetime | None,
    to_time: datetime | None,
    trace_id: str | None,
    span_kind: str | None,
    span_name: str | None,
    tags: list[str] | None,
    limit: int | None,
) -> AsyncIterator[Transcript]:
    """Fetch transcripts from Datadog query results.

    Args:
        client: DatadogClient instance
        ml_app: ml_app filter
        from_time: Start time filter
        to_time: End time filter
        trace_id: Trace ID filter
        span_kind: Span kind filter
        span_name: Span name filter
        tags: Tag filters
        limit: Max transcripts

    Yields:
        Transcript objects
    """
    all_spans = await client.list_spans(
        ml_app=ml_app,
        from_time=from_time,
        to_time=to_time,
        trace_id=trace_id,
        span_kind=span_kind,
        span_name=span_name,
        tags=tags,
    )

    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for span in all_spans:
        tid = span.get("trace_id")
        if tid:
            traces[str(tid)].append(span)

    count = 0
    for tid, trace_spans in traces.items():
        try:
            transcript = await _build_transcript(trace_spans, ml_app, tid, client.site)
            if transcript:
                yield transcript
                count += 1
                if limit and count >= limit:
                    return
        except Exception as e:
            if os.environ.get("DATADOG_STRICT_IMPORT", "").lower() in ("1", "true"):
                raise
            logger.warning("Failed to process trace %s: %s", tid, e)
            continue


async def _build_transcript(
    trace_spans: list[dict[str, Any]],
    ml_app: str | None,
    trace_id: str,
    site: str,
) -> Transcript | None:
    """Build a Transcript from a set of spans belonging to one trace.

    Args:
        trace_spans: All spans for a single trace
        ml_app: ml_app name
        trace_id: Trace ID
        site: Datadog site for constructing source URI

    Returns:
        Transcript object or None
    """
    if not trace_spans:
        return None

    tree = build_span_tree(trace_spans)
    ordered_spans = flatten_tree_chronological(tree)

    events = await spans_to_events(ordered_spans)

    llm_spans = get_llm_spans(ordered_spans)

    messages: list[ChatMessage] = []

    if llm_spans:
        model_events = [e for e in events if isinstance(e, ModelEvent)]
        if model_events:
            best_model = max(model_events, key=lambda e: len(e.input))
            messages = list(best_model.input)
            if best_model.output and best_model.output.message:
                messages.append(best_model.output.message)

    if not messages:
        root_span = ordered_spans[0] if ordered_spans else None
        if root_span:
            messages = _extract_root_messages(root_span)

    apply_ids = stable_message_ids()
    for event in events:
        if isinstance(event, ModelEvent):
            apply_ids(event)
    apply_ids(messages)

    model_name = str(llm_spans[0].get("model_name", "unknown")) if llm_spans else None

    root_span = ordered_spans[0] if ordered_spans else {}
    root_name = root_span.get("name") or "trace"

    date = None
    start_ns = root_span.get("start_ns")
    if start_ns is not None:
        try:
            dt = datetime.fromtimestamp(int(start_ns) / 1e9, tz=timezone.utc)
            date = dt.isoformat()
        except (ValueError, TypeError, OverflowError):
            pass

    error = None
    for span in ordered_spans:
        if str(span.get("status", "")).lower() == "error":
            error_meta = (span.get("meta") or {}).get("error") or {}
            error = error_meta.get("message") or "Unknown error"
            break

    metadata = _extract_metadata(root_span)

    source_uri = f"https://app.{site}/llm/traces/{trace_id}"
    source_id = ml_app or _get_ml_app_from_tags(root_span)

    return Transcript(
        transcript_id=trace_id,
        source_type=DATADOG_SOURCE_TYPE,
        source_id=source_id,
        source_uri=source_uri,
        date=date,
        task_set=source_id,
        task_id=str(root_name),
        task_repeat=None,
        agent=metadata.get("agent"),
        agent_args=metadata.get("agent_args"),
        model=model_name,
        model_options=_extract_model_options(llm_spans[0]) if llm_spans else None,
        score=metadata.get("score"),
        success=metadata.get("success"),
        message_count=len(messages),
        total_tokens=sum_tokens(llm_spans),
        total_time=_root_duration(root_span),
        error=error,
        limit=None,
        messages=messages,
        events=events,
        metadata=metadata,
    )


def _root_duration(root_span: dict[str, Any]) -> float:
    """Get the root span's duration in seconds.

    Args:
        root_span: Root span dictionary

    Returns:
        Duration in seconds, or 0.0 if unavailable
    """
    duration = root_span.get("duration")
    if duration is not None:
        try:
            return int(duration) / 1e9
        except (ValueError, TypeError):
            pass
    return 0.0


def _get_ml_app_from_tags(span: dict[str, Any]) -> str | None:
    """Extract ml_app value from span tags.

    Args:
        span: Datadog span dictionary

    Returns:
        ml_app value or None
    """
    tags = span.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("ml_app:"):
                return tag[len("ml_app:") :]
    return None


def _extract_root_messages(span: dict[str, Any]) -> list[ChatMessage]:
    """Extract messages from root span input/output.

    Args:
        span: Root span dictionary

    Returns:
        List of ChatMessage objects
    """
    messages: list[ChatMessage] = []
    meta = span.get("meta") or {}

    input_data = meta.get("input") or {}
    input_value = input_data.get("value")
    if input_value:
        messages.append(ChatMessageUser(content=str(input_value)))

    output_data = meta.get("output") or {}
    output_value = output_data.get("value")
    if output_value:
        messages.append(ChatMessageAssistant(content=str(output_value)))

    return messages


def _extract_metadata(span: dict[str, Any]) -> dict[str, Any]:
    """Extract metadata from span for Scout transcript.

    Args:
        span: Datadog span dictionary

    Returns:
        Metadata dictionary
    """
    metadata: dict[str, Any] = {}

    provider = detect_provider(span)
    if provider.value != "unknown":
        metadata["provider"] = provider.value

    meta = span.get("meta") or {}
    kind = meta.get("kind")
    if kind:
        metadata["span_kind"] = kind

    tags = span.get("tags")
    if isinstance(tags, list) and tags:
        metadata["tags"] = tags

    span_metadata = meta.get("metadata") or {}
    for key in ["agent", "agent_args", "score", "success"]:
        if key in span_metadata:
            metadata[key] = span_metadata[key]

    evaluations = meta.get("evaluations")
    if evaluations:
        metadata["evaluations"] = evaluations

    return metadata


def _extract_model_options(span: dict[str, Any]) -> dict[str, Any] | None:
    """Extract model generation options from span metadata.

    Args:
        span: LLM span dictionary

    Returns:
        Model options dict or None
    """
    meta = span.get("meta") or {}
    metadata = meta.get("metadata") or {}

    options: dict[str, Any] = {}
    for key in ["temperature", "max_tokens", "top_p", "top_k"]:
        if key in metadata:
            options[key] = metadata[key]

    return options if options else None


__all__ = ["datadog", "DATADOG_SOURCE_TYPE"]
