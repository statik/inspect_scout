"""Input/output extraction for Datadog LLM Observability span data.

The Datadog Export API provides a flat attribute structure with
``input.messages`` and ``output.messages`` at the top level, along
with explicit ``model_provider`` fields.
"""

import json
from logging import getLogger
from typing import Any

from inspect_ai.model import ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
)
from inspect_ai.model._model_output import ChatCompletionChoice, ModelUsage
from inspect_ai.tool import ToolCall, ToolInfo, ToolParams

from .detection import Provider, get_model_name

logger = getLogger(__name__)


async def extract_input_messages(
    span: dict[str, Any], provider: Provider
) -> list[ChatMessage]:
    """Extract input messages from a Datadog span.

    Reads ``input.messages`` (array of ``{role, content}``), routing to
    provider-specific converters. Falls back to ``input.value`` as
    a single user message.

    Args:
        span: Datadog span dictionary
        provider: Detected LLM provider

    Returns:
        List of ChatMessage objects
    """
    input_data = span.get("input") or {}

    messages_raw = input_data.get("messages")
    if messages_raw and isinstance(messages_raw, list):
        return await _convert_messages(messages_raw, provider)

    value = input_data.get("value")
    if value:
        return [ChatMessageUser(content=str(value))]

    return []


async def _convert_messages(
    messages: list[dict[str, Any]], provider: Provider
) -> list[ChatMessage]:
    """Convert Datadog message dicts to ChatMessage objects.

    Routes to inspect_ai provider-specific converters when possible.

    Args:
        messages: List of ``{role, content}`` dictionaries
        provider: Detected provider for format-specific handling

    Returns:
        List of ChatMessage objects
    """
    if provider == Provider.ANTHROPIC:
        try:
            from inspect_ai.model import messages_from_anthropic

            system_text = _extract_system_text(messages)
            non_system = [m for m in messages if m.get("role") != "system"]
            # Datadog messages are list[dict[str, Any]], not the typed
            # dicts the converter expects; runtime format is compatible.
            return await messages_from_anthropic(
                non_system,  # type: ignore[arg-type]
                system_message=system_text,
            )
        except Exception:
            logger.warning(
                "messages_from_anthropic failed, using OpenAI converter", exc_info=True
            )

    if provider == Provider.GOOGLE:
        try:
            from inspect_ai.model import messages_from_openai

            normalized = _normalize_messages(messages)
            # Normalized dicts match OpenAI format at runtime; see note above.
            return await messages_from_openai(normalized)  # type: ignore[arg-type]
        except Exception:
            logger.warning(
                "messages_from_openai failed for Google, using simple conversion",
                exc_info=True,
            )

    try:
        from inspect_ai.model import messages_from_openai

        normalized = _normalize_messages(messages)
        # Normalized dicts match OpenAI format at runtime; see note above.
        return await messages_from_openai(normalized)  # type: ignore[arg-type]
    except Exception:
        logger.warning(
            "messages_from_openai failed, using simple conversion", exc_info=True
        )

    return _simple_message_conversion(messages)


def _extract_system_text(messages: list[dict[str, Any]]) -> str | None:
    """Extract system message text from message list.

    Args:
        messages: List of message dicts

    Returns:
        System message text or None
    """
    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif isinstance(block, str):
                        parts.append(block)
                return " ".join(parts) if parts else None
    return None


def _normalize_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize Datadog messages to OpenAI format for the converter.

    Args:
        messages: Raw message dicts from Datadog

    Returns:
        Normalized message list
    """
    normalized = []
    for msg in messages:
        if not isinstance(msg, dict) or "role" not in msg:
            continue
        new_msg = dict(msg)

        role = new_msg.get("role", "")
        if role == "model":
            new_msg["role"] = "assistant"

        if "tool_calls" in new_msg:
            tool_calls = new_msg["tool_calls"]
            if isinstance(tool_calls, list):
                normalized_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        new_tc = dict(tc)
                        if "type" not in new_tc:
                            new_tc["type"] = "function"
                        if "function" not in new_tc and "name" in new_tc:
                            args = new_tc.pop("args", None)
                            raw_arguments = new_tc.pop("arguments", None)
                            if args is None:
                                args = raw_arguments
                            if isinstance(args, dict):
                                args = json.dumps(args)
                            elif not args:
                                args = "{}"
                            elif not isinstance(args, str):
                                args = json.dumps(args)
                            new_tc["function"] = {
                                "name": new_tc.pop("name"),
                                "arguments": args,
                            }
                        normalized_calls.append(new_tc)
                new_msg["tool_calls"] = normalized_calls

        normalized.append(new_msg)

    return normalized


def _simple_message_conversion(
    messages: list[dict[str, Any]],
) -> list[ChatMessage]:
    """Simple fallback message conversion.

    Args:
        messages: List of message dictionaries

    Returns:
        List of ChatMessage objects
    """
    result: list[ChatMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if role == "system":
            result.append(ChatMessageSystem(content=content))
        elif role == "user":
            result.append(ChatMessageUser(content=content))
        elif role == "assistant":
            result.append(ChatMessageAssistant(content=content))
    return result


async def extract_output(span: dict[str, Any]) -> ModelOutput:
    """Extract output from a Datadog span.

    Reads ``output.messages`` to construct a ModelOutput.

    Args:
        span: Datadog span dictionary

    Returns:
        ModelOutput object
    """
    output_data = span.get("output") or {}
    model_name = get_model_name(span) or "unknown"

    messages_raw = output_data.get("messages")
    if messages_raw and isinstance(messages_raw, list):
        last_msg = messages_raw[-1] if messages_raw else {}
        content = str(last_msg.get("content", ""))
        tool_calls_data = last_msg.get("tool_calls")

        if tool_calls_data and isinstance(tool_calls_data, list):
            tool_calls = _extract_tool_calls(tool_calls_data)
            output = ModelOutput(
                model=str(model_name),
                choices=[
                    ChatCompletionChoice(
                        message=ChatMessageAssistant(
                            content=content,
                            tool_calls=tool_calls,
                        ),
                        stop_reason="tool_calls",
                    )
                ],
            )
        else:
            output = ModelOutput.from_content(model=str(model_name), content=content)

        usage = extract_usage(span)
        if usage:
            output.usage = usage
        return output

    value = output_data.get("value")
    if value:
        output = ModelOutput.from_content(model=str(model_name), content=str(value))
        usage = extract_usage(span)
        if usage:
            output.usage = usage
        return output

    output = ModelOutput.from_content(model=str(model_name), content="")
    usage = extract_usage(span)
    if usage:
        output.usage = usage
    return output


def _extract_tool_calls(
    tool_calls_data: list[dict[str, Any]],
) -> list[ToolCall]:
    """Extract tool calls from output message data.

    Args:
        tool_calls_data: List of tool call dictionaries

    Returns:
        List of ToolCall objects
    """
    tool_calls: list[ToolCall] = []
    for tc in tool_calls_data:
        if not isinstance(tc, dict):
            logger.debug("Skipping non-dict tool call entry: %s", type(tc).__name__)
            continue
        func = tc.get("function", tc)
        if not isinstance(func, dict):
            logger.debug(
                "Skipping tool call with non-dict function: %s", tc.get("id", "unknown")
            )
            continue
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
        except json.JSONDecodeError:
            args = {}

        tool_calls.append(
            ToolCall(
                id=str(tc.get("id", "")),
                function=str(func.get("name", "")),
                arguments=args if isinstance(args, dict) else {},
                type="function",
            )
        )
    return tool_calls


def extract_usage(span: dict[str, Any]) -> ModelUsage | None:
    """Extract token usage from Datadog span metrics.

    Args:
        span: Datadog span dictionary

    Returns:
        ModelUsage object or None
    """
    metrics = span.get("metrics") or {}

    input_tokens = metrics.get("input_tokens")
    output_tokens = metrics.get("output_tokens")
    total_tokens = metrics.get("total_tokens")

    if input_tokens is not None or output_tokens is not None:
        input_t = int(input_tokens) if input_tokens is not None else 0
        output_t = int(output_tokens) if output_tokens is not None else 0
        total_t = int(total_tokens) if total_tokens is not None else input_t + output_t
        return ModelUsage(
            input_tokens=input_t,
            output_tokens=output_t,
            total_tokens=total_t,
        )

    return None


def extract_tools(span: dict[str, Any]) -> list[ToolInfo]:
    """Extract tool definitions from span metadata.

    Args:
        span: Datadog span dictionary

    Returns:
        List of ToolInfo objects
    """
    metadata = span.get("metadata") or {}
    tools_data = metadata.get("tools")

    if not tools_data or not isinstance(tools_data, list):
        return []

    tools: list[ToolInfo] = []
    for tool_schema in tools_data:
        tool_info = _parse_tool_schema(tool_schema)
        if tool_info:
            tools.append(tool_info)

    return tools


def _parse_tool_schema(schema: Any) -> ToolInfo | None:
    """Parse a tool JSON schema into ToolInfo.

    Args:
        schema: Tool schema (string or dict)

    Returns:
        ToolInfo or None
    """
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            return None

    if not isinstance(schema, dict):
        return None

    func = schema.get("function", schema)
    if not isinstance(func, dict):
        return None

    name = func.get("name", "")
    if not name:
        return None

    params = func.get("parameters", {})
    properties = params.get("properties", {}) if isinstance(params, dict) else {}
    required = params.get("required", []) if isinstance(params, dict) else []

    return ToolInfo(
        name=str(name),
        description=str(func.get("description", "")),
        parameters=ToolParams(
            type="object",
            properties=properties,
            required=required,
        ),
    )


def sum_tokens(spans: list[dict[str, Any]]) -> int:
    """Sum tokens across all spans.

    Args:
        spans: List of Datadog spans

    Returns:
        Total token count
    """
    total = 0
    for span in spans:
        metrics = span.get("metrics") or {}
        total_t = metrics.get("total_tokens")
        if total_t is not None:
            total += int(total_t)
        else:
            input_t = metrics.get("input_tokens") or 0
            output_t = metrics.get("output_tokens") or 0
            total += int(input_t) + int(output_t)
    return total
