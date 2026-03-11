"""LfxCallback — litellm custom logger for Mode B trajectory capture.

Registers as a litellm callback to automatically capture all LLM calls
and feed them into the lfx learning pipeline.

Usage::

    import litellm
    from lfx.callbacks.litellm_cb import LfxCallback

    collector = EpisodeCollector(pipeline=RewardPipeline.with_defaults(), ...)
    litellm.callbacks = [LfxCallback(collector=collector)]

    # All subsequent litellm.completion() calls are now captured
    litellm.completion(model="gpt-4o", messages=[...])
"""

from __future__ import annotations

import logging
from typing import Any

from lfx.collector import EpisodeCollector
from lfx.core.episode import Message, TokenLogProb, TokenUsage, ToolCall, cap_logprobs
from lfx.core.parse import parse_tool_calls, _safe_session_hash

log = logging.getLogger(__name__)


class LfxCallback:
    """litellm callback that captures completions into EpisodeCollector.

    Each litellm completion becomes a single-step episode. For multi-turn
    grouping, pass ``session_id`` in the litellm ``metadata`` dict.
    """

    def __init__(
        self,
        collector: EpisodeCollector,
        bench: str = "litellm",
    ) -> None:
        self.collector = collector
        self.bench = bench

    def log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Called by litellm after a successful completion."""
        try:
            self._process(kwargs, response_obj, start_time, end_time)
        except Exception:
            log.exception("LfxCallback: failed to process completion")

    def log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        exception: Exception | None = None,
    ) -> None:
        """Called by litellm after a failed completion. Currently a no-op."""
        pass

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Async variant — delegates to sync."""
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
        exception: Exception | None = None,
    ) -> None:
        """Async variant — delegates to sync."""
        self.log_failure_event(
            kwargs, response_obj, start_time, end_time, exception,
        )

    def _process(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: Any,
        end_time: Any,
    ) -> None:
        """Extract messages and metadata from litellm response."""
        input_messages = kwargs.get("messages", [])
        if not input_messages:
            return

        choice = response_obj.choices[0]
        text = choice.message.content or ""
        model = getattr(response_obj, "model", kwargs.get("model"))

        # Build input Message objects (including tool_calls from prior turns)
        ep_messages: list[Message] = []
        for m in input_messages:
            raw_content = m.get("content")
            if raw_content is None:
                content = ""
            elif isinstance(raw_content, str):
                content = raw_content
            else:
                content = str(raw_content)  # list/dict (vision) → string
            ep_messages.append(
                Message(
                    role=m.get("role", "user"),
                    content=content,
                    name=m.get("name"),
                    tool_calls=parse_tool_calls(m.get("tool_calls")),
                    tool_call_id=m.get("tool_call_id"),
                )
            )

        # Extract tool calls from response
        raw_tc = getattr(choice.message, "tool_calls", None)
        tool_calls = None
        if raw_tc:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in raw_tc
            ]

        # Extract logprobs (with cap)
        logprobs = None
        raw_logprobs = getattr(choice, "logprobs", None)
        if raw_logprobs and hasattr(raw_logprobs, "content") and raw_logprobs.content:
            logprobs = cap_logprobs([
                TokenLogProb(
                    token=lp.token,
                    token_id=getattr(lp, "token_id", None),
                    logprob=lp.logprob,
                )
                for lp in raw_logprobs.content
            ])

        # Build assistant message
        ep_messages.append(
            Message(
                role="assistant",
                content=text,
                model=model,
                tool_calls=tool_calls,
                logprobs=logprobs,
            )
        )

        # Extract usage
        usage = None
        raw_usage = getattr(response_obj, "usage", None)
        if raw_usage:
            usage = TokenUsage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", 0),
                completion_tokens=getattr(raw_usage, "completion_tokens", 0),
                total_tokens=getattr(raw_usage, "total_tokens", 0),
            )

        # Timing — litellm passes datetime objects, not floats
        if hasattr(start_time, "timestamp"):
            timing_ms = (end_time.timestamp() - start_time.timestamp()) * 1000
        else:
            timing_ms = (end_time - start_time) * 1000

        # Session ID from metadata or hash of first user message
        metadata = kwargs.get("metadata", {}) or {}
        session_id = metadata.get("session_id", "")
        if not session_id:
            for m in input_messages:
                if m.get("role") == "user":
                    session_id = _safe_session_hash(m.get("content", ""))
                    break

        self.collector.ingest(
            ep_messages,
            session_id=session_id,
            usage=usage,
            timing_ms=timing_ms,
            model=model,
            bench=self.bench,
        )
