"""Shared parsing utilities for OpenAI-format data structures."""

from __future__ import annotations

from typing import Any

from lfx.core.episode import TokenLogProb, ToolCall


def parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall] | None:
    """Convert OpenAI-format tool_call dicts to ToolCall objects."""
    if not raw:
        return None
    result = []
    for tc in raw:
        fn = tc.get("function", {})
        result.append(
            ToolCall(
                id=tc.get("id", ""),
                name=fn.get("name", "") if isinstance(fn, dict) else "",
                arguments=fn.get("arguments", "{}") if isinstance(fn, dict) else "{}",
            )
        )
    return result


def parse_logprobs(raw: list[dict[str, Any]] | None) -> list[TokenLogProb] | None:
    """Convert logprob dicts to TokenLogProb objects."""
    if not raw:
        return None
    return [
        TokenLogProb(
            token=lp.get("token", ""),
            token_id=lp.get("token_id"),
            logprob=lp.get("logprob", 0.0),
            top_logprobs=lp.get("top_logprobs"),
        )
        for lp in raw
    ]


def _safe_session_hash(content: Any) -> str:
    """Hash content for session_id, handling list/dict content (vision messages)."""
    import hashlib

    if not isinstance(content, str):
        content = str(content)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
