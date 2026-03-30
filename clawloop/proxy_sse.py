"""Pure-function parsers for OpenAI SSE streaming and JSON responses.

Used by the proxy to reconstruct the full assistant message from tee'd
response bytes without any external dependencies beyond the stdlib.
"""

from __future__ import annotations

import json


def parse_sse_bytes(
    raw: bytes,
) -> tuple[dict | None, dict | None, bool]:
    """Parse OpenAI SSE streaming bytes into a reconstructed message.

    Returns ``(assistant_message_dict, usage_dict, is_complete)``.

    * Accumulates ``choices[0].delta.content`` into a single string.
    * Accumulates ``choices[0].delta.tool_calls`` (index-based, arguments
      are incremental strings that get concatenated).
    * Extracts ``usage`` from the final chunk when present.
    * ``is_complete`` is ``True`` only when ``data: [DONE]`` was seen.
    * Returns ``(None, None, False)`` for empty / unparseable input.
    """
    if not raw or not raw.strip():
        return None, None, False

    text = raw.decode("utf-8", errors="replace")

    chunks: list[dict] = []
    is_complete = False

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            is_complete = True
            continue
        try:
            chunks.append(json.loads(payload))
        except (json.JSONDecodeError, ValueError):
            continue

    if not chunks:
        return None, None, is_complete

    # --- accumulate message ------------------------------------------------
    role: str | None = None
    content_parts: list[str] = []
    # tool_calls keyed by index → {id, type, function: {name, arguments}}
    tool_calls_by_index: dict[int, dict] = {}
    usage: dict | None = None
    model: str | None = None

    for chunk in chunks:
        # model
        if "model" in chunk and chunk["model"]:
            model = chunk["model"]

        # usage (typically on the last real chunk)
        if "usage" in chunk and chunk["usage"]:
            usage = chunk["usage"]

        choices = chunk.get("choices")
        if not choices:
            continue
        delta = choices[0].get("delta", {})

        # role
        if "role" in delta:
            role = delta["role"]

        # content
        if "content" in delta and delta["content"] is not None:
            content_parts.append(delta["content"])

        # tool_calls
        if "tool_calls" in delta:
            for tc_delta in delta["tool_calls"]:
                idx = tc_delta["index"]
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = {
                        "id": tc_delta.get("id", ""),
                        "type": tc_delta.get("type", "function"),
                        "function": {"name": "", "arguments": ""},
                    }
                existing = tool_calls_by_index[idx]

                # id (set once)
                if tc_delta.get("id"):
                    existing["id"] = tc_delta["id"]
                if tc_delta.get("type"):
                    existing["type"] = tc_delta["type"]

                fn_delta = tc_delta.get("function", {})
                if fn_delta.get("name"):
                    existing["function"]["name"] = fn_delta["name"]
                if "arguments" in fn_delta:
                    existing["function"]["arguments"] += fn_delta["arguments"]

    # Build the message dict
    msg: dict = {"role": role or "assistant"}

    content = "".join(content_parts) if content_parts else None
    msg["content"] = content

    if tool_calls_by_index:
        msg["tool_calls"] = [
            tool_calls_by_index[i] for i in sorted(tool_calls_by_index)
        ]

    if model:
        msg["model"] = model

    return msg, usage, is_complete


def parse_json_response(
    raw: bytes,
) -> tuple[dict | None, dict | None, str | None]:
    """Parse a non-streaming OpenAI chat completion JSON response.

    Returns ``(message_dict, usage_dict, model)``.
    Returns ``(None, None, None)`` on parse failure.
    """
    if not raw:
        return None, None, None

    try:
        body = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None, None, None

    choices = body.get("choices")
    if not choices:
        return None, None, None

    message = choices[0].get("message")
    if message is None:
        return None, None, None

    usage = body.get("usage")  # may be None
    model = body.get("model")

    return message, usage, model
