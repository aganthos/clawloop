"""Shared parsing utilities for OpenAI-format data structures."""

from __future__ import annotations

import logging
import re
from typing import Any

from clawloop.core.episode import TokenLogProb, ToolCall

_log = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Strip markdown code fences if present, return raw JSON string.

    LLMs frequently wrap JSON in ```json ... ``` fences. This strips
    them so ``json.loads`` can parse the payload.
    """
    m = _JSON_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


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


def resolve_oi_span_kind() -> tuple[str, str]:
    """Return (attribute_name, LLM_value) for OpenInference span kind.

    Imports ``openinference.semconv.trace`` if available; falls back to
    string literals so callers work without the optional dependency.
    """
    try:
        from openinference.semconv.trace import (  # type: ignore[import-untyped]
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )

        return SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value
    except ImportError:
        _log.debug("openinference not installed; using string literals for span kind")
        return "openinference.span.kind", "LLM"


def _safe_session_hash(content: Any) -> str:
    """Hash content for session_id, handling list/dict content (vision messages)."""
    import hashlib

    if not isinstance(content, str):
        content = str(content)
    return hashlib.sha256(content.encode()).hexdigest()[:16]
