"""Tests for Datadog trace tree reconstruction."""

import pytest
from inspect_scout.sources._datadog.tree import (
    build_span_tree,
    flatten_tree_chronological,
    get_llm_spans,
)

from .mocks import (
    _BASE_MS,
    _SECOND_MS,
    create_agent_span,
    create_llm_span,
    create_multiturn_trace,
    create_tool_call_trace,
    create_tool_span,
)


class TestBuildSpanTree:
    """Tests for build_span_tree function."""

    def test_single_span(self) -> None:
        """Build tree with single root span."""
        span = create_llm_span(span_id="span-1", trace_id="trace-1")
        roots = build_span_tree([span])

        assert len(roots) == 1
        assert roots[0].span_id == "span-1"
        assert roots[0].children == []

    def test_parent_child_relationship(self) -> None:
        """Build tree with parent-child relationship."""
        parent = create_agent_span(
            span_id="parent",
            trace_id="trace-1",
            start_ns=_BASE_MS,
        )
        child = create_llm_span(
            span_id="child",
            trace_id="trace-1",
            parent_id="parent",
            start_ns=_BASE_MS + _SECOND_MS,
        )

        roots = build_span_tree([parent, child])

        assert len(roots) == 1
        assert roots[0].span_id == "parent"
        assert len(roots[0].children) == 1
        assert roots[0].children[0].span_id == "child"

    def test_multiple_children_sorted_by_time(self) -> None:
        """Children should be sorted by start_ns timestamp."""
        parent = create_agent_span(
            span_id="parent",
            trace_id="trace-1",
            start_ns=_BASE_MS,
        )
        child1 = create_llm_span(
            span_id="child1",
            trace_id="trace-1",
            parent_id="parent",
            start_ns=_BASE_MS + 2 * _SECOND_MS,
        )
        child2 = create_tool_span(
            span_id="child2",
            trace_id="trace-1",
            parent_id="parent",
            start_ns=_BASE_MS + _SECOND_MS,
        )

        roots = build_span_tree([parent, child1, child2])

        assert len(roots[0].children) == 2
        assert roots[0].children[0].span_id == "child2"
        assert roots[0].children[1].span_id == "child1"

    def test_multiple_roots_sorted_by_time(self) -> None:
        """Multiple roots should be sorted by start_ns."""
        span1 = create_llm_span(
            span_id="span1",
            trace_id="trace-1",
            start_ns=_BASE_MS + _SECOND_MS,
        )
        span2 = create_llm_span(
            span_id="span2",
            trace_id="trace-1",
            start_ns=_BASE_MS,
        )

        roots = build_span_tree([span1, span2])

        assert len(roots) == 2
        assert roots[0].span_id == "span2"
        assert roots[1].span_id == "span1"

    def test_orphan_spans_become_roots(self) -> None:
        """Spans with missing parents become roots."""
        span = create_llm_span(
            span_id="orphan",
            trace_id="trace-1",
            parent_id="missing-parent",
        )
        roots = build_span_tree([span])

        assert len(roots) == 1
        assert roots[0].span_id == "orphan"

    def test_millisecond_timestamps_converted(self) -> None:
        """SpanNode.start_time converts milliseconds to datetime."""
        span = create_llm_span(start_ns=1700000000000)
        roots = build_span_tree([span])

        node = roots[0]
        assert node.start_time is not None
        assert node.start_time.year == 2023
        assert node.start_time.month == 11

    def test_empty_span_id_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Span with empty span_id is dropped with a warning."""
        span = {"span_id": "", "trace_id": "trace-1"}
        with caplog.at_level("WARNING", logger="inspect_scout.sources._datadog.tree"):
            roots = build_span_tree([span])

        assert roots == []
        assert "empty span_id" in caplog.text


class TestFlattenTreeChronological:
    """Tests for flatten_tree_chronological function."""

    def test_flatten_single_span(self) -> None:
        """Flatten tree with single span."""
        span = create_llm_span(span_id="span-1", trace_id="trace-1")
        roots = build_span_tree([span])
        flattened = flatten_tree_chronological(roots)

        assert len(flattened) == 1
        assert flattened[0]["span_id"] == "span-1"

    def test_flatten_preserves_depth_first_order(self) -> None:
        """Flatten should use depth-first traversal."""
        parent = create_agent_span(
            span_id="parent",
            trace_id="trace-1",
            start_ns=_BASE_MS,
        )
        child = create_llm_span(
            span_id="child",
            trace_id="trace-1",
            parent_id="parent",
            start_ns=_BASE_MS + _SECOND_MS,
        )
        grandchild = create_tool_span(
            span_id="grandchild",
            trace_id="trace-1",
            parent_id="child",
            start_ns=_BASE_MS + 2 * _SECOND_MS,
        )

        roots = build_span_tree([parent, child, grandchild])
        flattened = flatten_tree_chronological(roots)

        assert len(flattened) == 3
        assert flattened[0]["span_id"] == "parent"
        assert flattened[1]["span_id"] == "child"
        assert flattened[2]["span_id"] == "grandchild"


class TestMultiturnSpans:
    """Tests for multi-turn conversation span handling."""

    def test_multiturn_spans_ordered(self) -> None:
        """Multi-turn spans maintain chronological order."""
        spans = create_multiturn_trace()
        roots = build_span_tree(spans)
        flattened = flatten_tree_chronological(roots)

        assert len(flattened) == 3
        assert flattened[0]["span_id"] == "span-turn-1"
        assert flattened[1]["span_id"] == "span-turn-2"
        assert flattened[2]["span_id"] == "span-turn-3"

    def test_tool_call_trace_structure(self) -> None:
        """Tool call trace has correct parent-child structure."""
        spans = create_tool_call_trace()
        roots = build_span_tree(spans)

        assert len(roots) == 1
        assert roots[0].span_id == "span-agent-root"
        assert len(roots[0].children) >= 2


class TestSpanFilters:
    """Tests for span filtering functions."""

    def test_get_llm_spans(self) -> None:
        """Filter to only LLM spans."""
        llm_span = create_llm_span(span_id="llm", trace_id="trace-1")
        tool_span = create_tool_span(span_id="tool", trace_id="trace-1")
        agent_span = create_agent_span(span_id="agent", trace_id="trace-1")

        spans = [llm_span, tool_span, agent_span]
        llm_spans = get_llm_spans(spans)

        assert len(llm_spans) == 1
        assert llm_spans[0]["span_id"] == "llm"

    def test_get_llm_spans_multiturn(self) -> None:
        """Filter multi-turn spans to LLM spans."""
        spans = create_multiturn_trace()
        llm_spans = get_llm_spans(spans)

        assert len(llm_spans) == 3

    def test_tool_call_trace_filtering(self) -> None:
        """Filter tool call trace to LLM spans."""
        spans = create_tool_call_trace()

        llm_spans = get_llm_spans(spans)

        assert len(llm_spans) == 2
