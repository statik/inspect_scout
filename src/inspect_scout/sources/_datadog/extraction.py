"""Input/output extraction for Datadog LLM Observability span data.

The Datadog Export API provides a flat attribute structure with
``input.messages`` and ``output.messages`` at the top level, along
with explicit ``model_provider`` fields.
"""

import json
from logging import getLogger
from typing import Any

from inspect_ai._util.content import Content
from inspect_ai.model import ModelOutput
from inspect_ai.model._chat_message import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
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
                new_msg["tool_calls"] = [
                    _normalize_tool_call(tc)
                    for tc in tool_calls
                    if isinstance(tc, dict)
                ]

        normalized.append(new_msg)

    return normalized


def _normalize_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """Normalize a single tool call dict to OpenAI format.

    Ensures ``type`` is set and wraps bare ``name``/``args`` into the
    ``{"function": {"name": ..., "arguments": ...}}`` structure that
    ``messages_from_openai`` expects.

    Args:
        tc: Raw tool call dictionary

    Returns:
        Normalized tool call dictionary
    """
    new_tc = dict(tc)
    if "type" not in new_tc:
        new_tc["type"] = "function"
    if "function" not in new_tc and "name" in new_tc:
        args = new_tc.pop("args", None)
        raw_arguments = new_tc.pop("arguments", None)
        if args is None or args == "" or args == {}:
            args = raw_arguments
        if isinstance(args, dict):
            args = json.dumps(args)
        elif args is None or args == "":
            args = "{}"
        elif not isinstance(args, str):
            args = json.dumps(args)
        new_tc["function"] = {
            "name": new_tc.pop("name"),
            "arguments": args,
        }
    return new_tc


def _simple_message_conversion(
    messages: list[dict[str, Any]],
) -> list[ChatMessage]:
    """Fallback message conversion when provider-specific converters fail.

    Preserves tool calls as structured ``ToolCall`` objects rather than
    dropping them silently. Logs a warning so degraded conversion is visible.

    Args:
        messages: List of message dictionaries

    Returns:
        List of ChatMessage objects
    """
    logger.warning(
        "Using fallback message conversion; structured content types "
        "may be less accurate than provider-specific converters"
    )
    result: list[ChatMessage] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = _convert_content(msg.get("content", ""))
        if role == "system":
            result.append(ChatMessageSystem(content=content))
        elif role == "user":
            result.append(ChatMessageUser(content=content))
        elif role in ("assistant", "model"):
            tool_calls = _extract_fallback_tool_calls(msg.get("tool_calls"))
            result.append(
                ChatMessageAssistant(
                    content=content,
                    tool_calls=tool_calls if tool_calls else None,
                )
            )
        elif role == "tool":
            # Tool messages are always plain text
            text_content = str(msg.get("content", ""))
            result.append(
                ChatMessageTool(
                    content=text_content,
                    tool_call_id=msg.get("tool_call_id", ""),
                    function=msg.get("name"),
                )
            )
        else:
            logger.warning("Skipping message with unhandled role: %s", role)
    return result


def _extract_fallback_tool_calls(
    tool_calls_raw: Any,
) -> list[ToolCall]:
    """Extract tool calls from raw message data in the fallback path.

    Args:
        tool_calls_raw: Raw tool_calls value from a message dict

    Returns:
        List of ToolCall objects (empty if none found)
    """
    if not tool_calls_raw or not isinstance(tool_calls_raw, list):
        return []

    tool_calls: list[ToolCall] = []
    for tc in tool_calls_raw:
        if not isinstance(tc, dict):
            continue
        func = tc.get("function", tc)
        if not isinstance(func, dict):
            continue
        name = str(func.get("name", ""))
        if not name:
            continue
        args_raw = func.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}
        tool_calls.append(
            ToolCall(
                id=str(tc.get("id", "")),
                function=name,
                arguments=args if isinstance(args, dict) else {},
                type="function",
            )
        )
    return tool_calls


def _convert_content(raw: Any) -> str | list[Content]:
    """Convert raw message content to the appropriate type.

    Strings pass through directly. Lists of content blocks are converted
    using inspect_ai's ``content_from_openai`` so that structured types
    like ``ContentImage`` and ``ContentAudio`` are preserved.

    Args:
        raw: Content field from a message dict (str, list, or other)

    Returns:
        Plain string or list of Content objects
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        try:
            from inspect_ai.model._openai import content_from_openai

            parts: list[Content] = []
            for block in raw:
                if isinstance(block, dict):
                    # Datadog dicts match OpenAI format at runtime.
                    parts.extend(content_from_openai(block))  # type: ignore[arg-type]
                elif isinstance(block, str):
                    from inspect_ai._util.content import ContentText

                    parts.append(ContentText(text=block))
            return parts if parts else ""
        except Exception:
            logger.debug(
                "content_from_openai failed for content list, stringifying",
                exc_info=True,
            )
    return str(raw) if raw else ""


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

    output: ModelOutput
    messages_raw = output_data.get("messages")
    if messages_raw and isinstance(messages_raw, list):
        last_msg = messages_raw[-1] if messages_raw else {}
        content = _convert_content(last_msg.get("content", ""))
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
    elif value := output_data.get("value"):
        output = ModelOutput.from_content(model=str(model_name), content=str(value))
    else:
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
