"""lfx.wrap() — SDK wrapper that intercepts LLM calls for learning."""

from __future__ import annotations

import time
from typing import Any
from uuid import uuid4

from lfx.collector import EpisodeCollector
from lfx.completion import CompletionResult
from lfx.core.episode import Message
from lfx.core.parse import parse_tool_calls, _safe_session_hash


class WrappedClient:
    """Drop-in LLMClient replacement that intercepts calls for learning."""

    def __init__(self, client: Any, collector: EpisodeCollector) -> None:
        self._client = client
        self._collector = collector

    def complete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> CompletionResult:
        # Extract lfx-specific kwargs before forwarding to the client
        task_id = kwargs.pop("task_id", uuid4().hex)

        start = time.monotonic()
        response = self._client.complete(messages, **kwargs)
        elapsed_ms = (time.monotonic() - start) * 1000

        # Normalize response to CompletionResult
        if isinstance(response, CompletionResult):
            result = response
            # Use client-reported latency if available, else our measurement
            if result.latency_ms is None:
                result = CompletionResult(
                    text=result.text,
                    model=result.model,
                    tool_calls=result.tool_calls,
                    usage=result.usage,
                    logprobs=result.logprobs,
                    latency_ms=elapsed_ms,
                    raw_response=result.raw_response,
                )
        else:
            result = CompletionResult(text=str(response), latency_ms=elapsed_ms)

        # Build rich Message objects from input messages
        ep_messages: list[Message] = []
        for m in messages:
            ep_messages.append(
                Message(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    name=m.get("name"),
                    tool_calls=parse_tool_calls(m.get("tool_calls")),
                    tool_call_id=m.get("tool_call_id"),
                )
            )

        # Assistant response message with rich metadata
        ep_messages.append(
            Message(
                role="assistant",
                content=result.text,
                model=result.model,
                tool_calls=result.tool_calls,
                logprobs=result.logprobs,
                token_count=(
                    result.usage.completion_tokens if result.usage else None
                ),
                timestamp=time.time(),
            )
        )

        # Stable session_id from first user message
        session_id = ""
        for m in messages:
            if m.get("role") == "user":
                session_id = _safe_session_hash(m.get("content", ""))
                break

        self._collector.ingest(
            ep_messages,
            task_id=task_id,
            session_id=session_id,
            usage=result.usage,
            timing_ms=result.latency_ms,
            model=result.model,
        )

        return result


def wrap(client: Any, collector: EpisodeCollector) -> WrappedClient:
    """Wrap an LLMClient with live-mode episode collection.

    Usage::

        wrapped = lfx.wrap(my_client, collector=collector)
        result = wrapped.complete(messages)  # works exactly like before
    """
    return WrappedClient(client, collector)
