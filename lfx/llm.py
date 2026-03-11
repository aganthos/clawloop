"""LLM client abstraction — Protocol + LiteLLM and Mock implementations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from lfx.completion import CompletionResult
from lfx.core.episode import TokenLogProb, TokenUsage, ToolCall, cap_logprobs


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM completion clients."""

    def complete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> CompletionResult:
        """Send messages to an LLM and return a rich completion result."""
        ...


@dataclass
class LiteLLMClient:
    """Production LLM client backed by litellm.completion().

    Supports 100+ providers via LiteLLM's unified API.
    Pass api_base to route through a proxy (e.g. CLIProxyAPI).
    Automatically requests logprobs unless caller opts out.
    """

    model: str
    api_key: str | None = None
    api_base: str | None = None
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.default_kwargs = kwargs

    def complete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> CompletionResult:
        """Call litellm.completion() and return a CompletionResult."""
        import litellm

        start = time.monotonic()

        merged = {**self.default_kwargs, **kwargs}
        if self.api_key is not None:
            merged["api_key"] = self.api_key
        if self.api_base is not None:
            merged["api_base"] = self.api_base

        response = litellm.completion(
            model=self.model, messages=messages, **merged
        )
        elapsed_ms = (time.monotonic() - start) * 1000

        choice = response.choices[0]
        text = choice.message.content or ""

        # Extract tool calls
        raw_tool_calls = getattr(choice.message, "tool_calls", None)
        tool_calls = None
        if raw_tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=tc.function.arguments,
                )
                for tc in raw_tool_calls
            ]

        # Extract per-token logprobs
        token_logprobs = None
        raw_logprobs = getattr(choice, "logprobs", None)
        if raw_logprobs and hasattr(raw_logprobs, "content") and raw_logprobs.content:
            token_logprobs = [
                TokenLogProb(
                    token=lp.token,
                    token_id=getattr(lp, "token_id", None),
                    logprob=lp.logprob,
                    top_logprobs=(
                        {t.token: t.logprob for t in lp.top_logprobs}
                        if getattr(lp, "top_logprobs", None)
                        else None
                    ),
                )
                for lp in raw_logprobs.content
            ]

        # Extract usage
        usage = None
        raw_usage = getattr(response, "usage", None)
        if raw_usage:
            usage = TokenUsage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", 0),
                completion_tokens=getattr(raw_usage, "completion_tokens", 0),
                total_tokens=getattr(raw_usage, "total_tokens", 0),
            )

        return CompletionResult(
            text=text,
            model=getattr(response, "model", self.model),
            tool_calls=tool_calls,
            usage=usage,
            logprobs=cap_logprobs(token_logprobs),
            latency_ms=elapsed_ms,
            raw_response=response,
        )


@dataclass
class MockLLMClient:
    """Deterministic mock LLM client for testing.

    Cycles through the provided responses list and records every call.
    Optionally provides mock tool_calls and logprobs per response.
    """

    responses: list[str] = field(default_factory=lambda: ["mock response"])
    model: str | None = field(default=None)
    tool_calls: list[list[ToolCall] | None] | None = field(default=None)
    logprobs: list[list[TokenLogProb] | None] | None = field(default=None)
    call_log: list[tuple[list[dict[str, str]], dict[str, Any]]] = field(
        default_factory=list
    )
    _call_idx: int = 0

    def complete(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> CompletionResult:
        """Return the next canned response and log the call."""
        self.call_log.append((messages, kwargs))
        idx = self._call_idx % len(self.responses)
        text = self.responses[idx]

        tc = None
        if self.tool_calls and idx < len(self.tool_calls):
            tc = self.tool_calls[idx]

        lp = None
        if self.logprobs and idx < len(self.logprobs):
            lp = self.logprobs[idx]

        self._call_idx += 1

        return CompletionResult(
            text=text,
            model=self.model,
            tool_calls=tc,
            logprobs=lp,
        )
