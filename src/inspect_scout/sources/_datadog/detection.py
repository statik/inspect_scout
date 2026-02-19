"""Provider detection for Datadog LLM Observability spans.

Datadog provides explicit `model_provider` and `meta.kind` fields,
making detection straightforward compared to other adapters.
"""

from enum import Enum
from typing import Any


class Provider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    UNKNOWN = "unknown"


# Mapping from Datadog model_provider values to Provider enum
PROVIDER_MAP: dict[str, Provider] = {
    "openai": Provider.OPENAI,
    "anthropic": Provider.ANTHROPIC,
    "google": Provider.GOOGLE,
    "google_genai": Provider.GOOGLE,
    "vertex_ai": Provider.GOOGLE,
}


def detect_provider(span: dict[str, Any]) -> Provider:
    """Detect the LLM provider from a Datadog span.

    Detection priority:
    1. Check ``model_provider`` field (explicit in DD spans)
    2. Infer from model name patterns

    Args:
        span: Datadog span dictionary

    Returns:
        Detected Provider enum value
    """
    provider_str = span.get("model_provider")
    if provider_str:
        provider_lower = str(provider_str).lower()
        if provider_lower in PROVIDER_MAP:
            return PROVIDER_MAP[provider_lower]

    model_name = get_model_name(span)
    if model_name:
        model_lower = model_name.lower()
        if any(p in model_lower for p in ["gpt-", "o1-", "o3-", "text-davinci"]):
            return Provider.OPENAI
        if "claude" in model_lower:
            return Provider.ANTHROPIC
        if "gemini" in model_lower or "palm" in model_lower:
            return Provider.GOOGLE

    return Provider.UNKNOWN


def get_model_name(span: dict[str, Any]) -> str | None:
    """Get the model name from a Datadog span.

    Args:
        span: Datadog span dictionary

    Returns:
        Model name or None if not found
    """
    model = span.get("model_name")
    if model:
        return str(model)
    return None


def is_llm_span(span: dict[str, Any]) -> bool:
    """Check if a span represents an LLM operation.

    Args:
        span: Datadog span dictionary

    Returns:
        True if this is an LLM span
    """
    meta = span.get("meta") or {}
    return str(meta.get("kind", "")).lower() == "llm"


def is_tool_span(span: dict[str, Any]) -> bool:
    """Check if a span represents a tool execution.

    Args:
        span: Datadog span dictionary

    Returns:
        True if this is a tool span
    """
    meta = span.get("meta") or {}
    return str(meta.get("kind", "")).lower() == "tool"


def is_agent_span(span: dict[str, Any]) -> bool:
    """Check if a span represents an agent/workflow/task operation.

    Args:
        span: Datadog span dictionary

    Returns:
        True if this is an agent, workflow, or task span
    """
    meta = span.get("meta") or {}
    return str(meta.get("kind", "")).lower() in ("agent", "workflow", "task")
