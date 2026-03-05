"""Trace tree reconstruction from Datadog spans.

Datadog spans use top-level ``span_id``, ``parent_id``, and ``trace_id`` fields
with millisecond timestamps (``start_ns``, ``duration``).

Note: Despite the field name ``start_ns``, the Datadog Export API returns
values in milliseconds, not nanoseconds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import getLogger
from typing import Any

from .detection import is_llm_span

logger = getLogger(__name__)

DATETIME_MIN_UTC = datetime.min.replace(tzinfo=timezone.utc)


def _ms_to_datetime(ms: Any) -> datetime:
    """Convert millisecond timestamp to datetime.

    The Datadog Export API field is named ``start_ns`` but actually
    contains milliseconds (13-digit values, not 19-digit nanoseconds).

    Args:
        ms: Millisecond timestamp (int or None)

    Returns:
        UTC datetime, or datetime.min (UTC) if conversion fails
    """
    if ms is not None:
        try:
            return datetime.fromtimestamp(int(ms) / 1e3, tz=timezone.utc)
        except (ValueError, TypeError, OverflowError):
            pass
    return DATETIME_MIN_UTC


@dataclass
class SpanNode:
    """A node in the span tree."""

    span: dict[str, Any]
    children: list["SpanNode"] = field(default_factory=list)

    @property
    def span_id(self) -> str:
        return str(self.span.get("span_id", ""))

    @property
    def parent_id(self) -> str | None:
        parent = self.span.get("parent_id")
        return str(parent) if parent else None

    @property
    def start_time(self) -> datetime:
        return _ms_to_datetime(self.span.get("start_ns"))

    @property
    def trace_id(self) -> str:
        return str(self.span.get("trace_id", ""))


def build_span_tree(spans: list[dict[str, Any]]) -> list[SpanNode]:
    """Build a tree structure from flat list of Datadog spans.

    Args:
        spans: Flat list of Datadog spans with parent_id references

    Returns:
        List of root SpanNode objects (spans without parents)
    """
    nodes: dict[str, SpanNode] = {}
    for span in spans:
        span_id = str(span.get("span_id", ""))
        if span_id:
            nodes[span_id] = SpanNode(span=span)
        else:
            logger.warning(
                "Dropping span with empty span_id (trace_id=%s)", span.get("trace_id")
            )

    roots: list[SpanNode] = []
    for node in nodes.values():
        parent_id = node.parent_id
        if parent_id and parent_id in nodes:
            nodes[parent_id].children.append(node)
        else:
            roots.append(node)

    def sort_children(node: SpanNode) -> None:
        node.children.sort(key=lambda n: n.start_time or DATETIME_MIN_UTC)
        for child in node.children:
            sort_children(child)

    for root in roots:
        sort_children(root)

    roots.sort(key=lambda n: n.start_time or DATETIME_MIN_UTC)

    return roots


def flatten_tree_chronological(
    roots: list[SpanNode],
) -> list[dict[str, Any]]:
    """Flatten tree to chronologically ordered list of spans.

    Performs a depth-first traversal, emitting spans in the order
    they would have executed.

    Args:
        roots: List of root SpanNode objects

    Returns:
        Chronologically ordered list of spans
    """
    result: list[dict[str, Any]] = []

    def visit(node: SpanNode) -> None:
        result.append(node.span)
        for child in node.children:
            visit(child)

    for root in roots:
        visit(root)

    return result


def get_llm_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter spans to only LLM operation spans.

    Args:
        spans: List of Datadog spans

    Returns:
        List of LLM spans
    """
    return [span for span in spans if is_llm_span(span)]
