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
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from logging import getLogger
from typing import Any

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
from .detection import Provider, detect_provider, get_model_name
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
    min_messages: int | None = None,
    exclude_models: list[str] | None = None,
    deduplicate_by: str | None = None,
    api_key: str | None = None,
    app_key: str | None = None,
    site: str | None = None,
) -> AsyncGenerator[Transcript, None]:
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
        limit: Maximum number of transcripts to yield
        min_messages: Skip traces with fewer than this many messages.
            Applied after transcript assembly.
        exclude_models: Skip traces whose model matches any entry.
            Case-insensitive substring matching (e.g. ``"gpt-3.5"``
            matches ``"gpt-3.5-turbo"``). Transcripts with no model
            are always passed through.
        deduplicate_by: Tag key to deduplicate on. When set, only
            the transcript with the highest ``total_tokens`` per unique
            tag value is yielded. Transcripts missing the tag pass
            through immediately.
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
            min_messages,
            exclude_models,
            deduplicate_by,
        ):
            yield transcript
    finally:
        await client.aclose()


async def _try_build(
    trace_spans: list[dict[str, Any]],
    ml_app: str | None,
    trace_id: str,
    site: str,
    strict: bool,
) -> Transcript | None:
    """Build a transcript, suppressing errors unless strict mode is on.

    Args:
        trace_spans: All spans for a single trace
        ml_app: ml_app name
        trace_id: Trace ID
        site: Datadog site
        strict: If True, re-raise processing errors

    Returns:
        Transcript or None if building failed
    """
    try:
        return await _build_transcript(trace_spans, ml_app, trace_id, site)
    except Exception as e:
        if strict:
            raise
        logger.warning("Failed to process trace %s: %s", trace_id, e)
        return None


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
    min_messages: int | None = None,
    exclude_models: list[str] | None = None,
    deduplicate_by: str | None = None,
) -> AsyncGenerator[Transcript, None]:
    """Fetch transcripts from Datadog, streaming page by page.

    Processes API results incrementally to bound memory usage.
    Spans are grouped by trace_id; a trace is considered complete
    when a subsequent page arrives with no new spans for that trace.

    Args:
        client: DatadogClient instance
        ml_app: ml_app filter
        from_time: Start time filter
        to_time: End time filter
        trace_id: Trace ID filter
        span_kind: Span kind filter
        span_name: Span name filter
        tags: Tag filters
        limit: Max transcripts to yield
        min_messages: Skip traces with fewer messages
        exclude_models: Skip traces matching these models
        deduplicate_by: Tag key to deduplicate on. When set, only the
            transcript with the highest ``total_tokens`` per unique tag
            value is yielded. Transcripts missing the tag pass through.

    Yields:
        Transcript objects
    """
    if limit is not None and limit < 1:
        raise ValueError(f"limit must be a positive integer, got {limit}")

    strict = os.environ.get("DATADOG_STRICT_IMPORT", "").lower() in ("1", "true")
    traces: dict[str, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    dedup_buffer: dict[str, Transcript] = {}

    async for page in client.iter_span_pages(
        ml_app=ml_app,
        from_time=from_time,
        to_time=to_time,
        trace_id=trace_id,
        span_kind=span_kind,
        span_name=span_name,
        tags=tags,
    ):
        page_trace_ids: set[str] = set()
        for span in page:
            tid = span.get("trace_id")
            if tid:
                str_tid = str(tid)
                traces[str_tid].append(span)
                page_trace_ids.add(str_tid)

        # Traces present in the buffer but absent from this page
        # have received all their spans — yield and discard them.
        completed = [tid for tid in traces if tid not in page_trace_ids]
        for tid in completed:
            transcript = await _try_build(
                traces.pop(tid), ml_app, tid, client.site, strict
            )
            if transcript and _matches_trace_filter(
                transcript, min_messages, exclude_models
            ):
                result = _should_deduplicate(transcript, deduplicate_by, dedup_buffer)
                if result is not None:
                    yield result
                    count += 1
                    if limit and count >= limit:
                        return

    # Yield remaining buffered traces (from the last page)
    for tid in list(traces):
        transcript = await _try_build(traces.pop(tid), ml_app, tid, client.site, strict)
        if transcript and _matches_trace_filter(
            transcript, min_messages, exclude_models
        ):
            result = _should_deduplicate(transcript, deduplicate_by, dedup_buffer)
            if result is not None:
                yield result
                count += 1
                if limit and count >= limit:
                    return

    # Yield dedup winners
    for transcript in dedup_buffer.values():
        yield transcript
        count += 1
        if limit and count >= limit:
            return


def _matches_trace_filter(
    transcript: Transcript,
    min_messages: int | None,
    exclude_models: list[str] | None,
) -> bool:
    """Check whether a transcript passes client-side filters.

    All specified filters must pass (AND semantics).

    Args:
        transcript: Built transcript to check
        min_messages: Minimum message count required
        exclude_models: Model substrings to exclude (case-insensitive)

    Returns:
        True if the transcript passes all filters
    """
    if min_messages is not None:
        count = transcript.message_count or 0
        if count < min_messages:
            return False

    if exclude_models and transcript.model:
        model_lower = transcript.model.lower()
        for pattern in exclude_models:
            if pattern.lower() in model_lower:
                return False

    return True


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

    if not ordered_spans:
        logger.warning("Trace %s: all spans dropped during tree construction", trace_id)
        return None

    events = await spans_to_events(ordered_spans)

    llm_spans = get_llm_spans(ordered_spans)

    messages: list[ChatMessage] = []

    if llm_spans:
        model_events = [e for e in events if isinstance(e, ModelEvent)]
        if model_events:
            # Pick the model event with the longest input as the best
            # representation of the conversation. This works well when the
            # application accumulates full history into each LLM call, but
            # may select an arbitrary mid-conversation turn for apps using
            # sliding-window, summarization, or RAG patterns.
            best_model = max(model_events, key=lambda e: len(e.input))
            messages = list(best_model.input)
            if best_model.output and not best_model.output.empty:
                messages.append(best_model.output.message)

    if not messages:
        root_span = ordered_spans[0]
        messages = _extract_root_messages(root_span)

    apply_ids = stable_message_ids()
    for event in events:
        if isinstance(event, ModelEvent):
            apply_ids(event)
    apply_ids(messages)

    model_name = get_model_name(llm_spans[0]) if llm_spans else None

    root_span = ordered_spans[0]
    root_name = root_span.get("name") or "trace"

    date = None
    start_ns = root_span.get("start_ns")
    if start_ns is not None:
        try:
            # start_ns is milliseconds despite the field name
            dt = datetime.fromtimestamp(int(start_ns) / 1e3, tz=timezone.utc)
            date = dt.isoformat()
        except (ValueError, TypeError, OverflowError):
            pass

    error = None
    for span in ordered_spans:
        if str(span.get("status", "")).lower() == "error":
            error_meta = span.get("error") or {}
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


def _root_duration(root_span: dict[str, Any]) -> float | None:
    """Get the root span's duration in seconds.

    Args:
        root_span: Root span dictionary

    Returns:
        Duration in seconds, or None if unavailable
    """
    duration = root_span.get("duration")
    if duration is not None:
        try:
            # duration is milliseconds despite the naming convention
            return int(duration) / 1e3
        except (ValueError, TypeError):
            pass
    return None


def _get_tag_value(transcript: Transcript, tag_key: str) -> str | None:
    """Extract a tag value from transcript metadata tags.

    Looks for ``"key:value"`` entries in ``transcript.metadata["tags"]``.

    Args:
        transcript: Transcript with metadata potentially containing tags
        tag_key: Tag key to search for (e.g. ``"session_id"``)

    Returns:
        The tag value, or None if not found
    """
    metadata = transcript.metadata or {}
    tags = metadata.get("tags")
    if isinstance(tags, list):
        prefix = f"{tag_key}:"
        for tag in tags:
            if isinstance(tag, str) and tag.startswith(prefix):
                return tag[len(prefix) :]
    return None


def _should_deduplicate(
    transcript: Transcript,
    deduplicate_by: str | None,
    dedup_buffer: dict[str, Transcript],
) -> Transcript | None:
    """Buffer transcript for deduplication, returning it only if it should be yielded now.

    When ``deduplicate_by`` is set, transcripts sharing the same tag value
    compete — only the one with the highest ``total_tokens`` wins. Transcripts
    without the tag are passed through immediately.

    Args:
        transcript: The transcript to consider
        deduplicate_by: Tag key to deduplicate on, or None to disable
        dedup_buffer: Mutable buffer mapping tag values to best-so-far transcripts

    Returns:
        The transcript if it should be yielded immediately, or None if buffered
    """
    if deduplicate_by is None:
        return transcript

    tag_value = _get_tag_value(transcript, deduplicate_by)
    if tag_value is None:
        return transcript

    existing = dedup_buffer.get(tag_value)
    if existing is None:
        dedup_buffer[tag_value] = transcript
    else:
        new_tokens = transcript.total_tokens or 0
        existing_tokens = existing.total_tokens or 0
        if new_tokens > existing_tokens:
            dedup_buffer[tag_value] = transcript

    return None


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

    input_data = span.get("input") or {}
    input_value = input_data.get("value")
    if input_value:
        messages.append(ChatMessageUser(content=str(input_value)))

    output_data = span.get("output") or {}
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
    if provider != Provider.UNKNOWN:
        metadata["provider"] = provider.value

    kind = span.get("span_kind")
    if kind:
        metadata["span_kind"] = kind

    tags = span.get("tags")
    if isinstance(tags, list) and tags:
        metadata["tags"] = tags

    span_metadata = span.get("metadata") or {}
    for key in ["agent", "agent_args", "score"]:
        if key in span_metadata:
            metadata[key] = span_metadata[key]

    if "success" in span_metadata:
        val = span_metadata["success"]
        if val is None:
            metadata["success"] = None
        elif isinstance(val, str):
            metadata["success"] = val.lower() not in ("false", "0", "no", "")
        else:
            metadata["success"] = bool(val)

    evaluations = span.get("evaluations")
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
    metadata = span.get("metadata") or {}

    options: dict[str, Any] = {}
    for key in ["temperature", "max_tokens", "top_p", "top_k"]:
        if key in metadata:
            options[key] = metadata[key]

    return options if options else None


__all__ = ["datadog", "DATADOG_SOURCE_TYPE"]
