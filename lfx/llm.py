"""LLM client abstraction — Protocol + LiteLLM and Mock implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for LLM completion clients."""

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Send messages to an LLM and return the text response."""
        ...


@dataclass
class LiteLLMClient:
    """Production LLM client backed by litellm.completion().

    Supports 100+ providers via LiteLLM's unified API.
    Pass api_base to route through a proxy (e.g. CLIProxyAPI).
    """

    model: str
    api_key: str | None = None
    api_base: str | None = None
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self, model: str, api_key: str | None = None, api_base: str | None = None, **kwargs: Any
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.default_kwargs = kwargs

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Call litellm.completion() and return the assistant message text."""
        import litellm

        merged = {**self.default_kwargs, **kwargs}
        if self.api_key is not None:
            merged["api_key"] = self.api_key
        if self.api_base is not None:
            merged["api_base"] = self.api_base

        response = litellm.completion(
            model=self.model, messages=messages, **merged
        )
        return response.choices[0].message.content


@dataclass
class MockLLMClient:
    """Deterministic mock LLM client for testing.

    Cycles through the provided responses list and records every call.
    """

    responses: list[str] = field(default_factory=lambda: ["mock response"])
    call_log: list[tuple[list[dict[str, str]], dict[str, Any]]] = field(
        default_factory=list
    )
    _call_idx: int = 0

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Return the next canned response and log the call."""
        self.call_log.append((messages, kwargs))
        response = self.responses[self._call_idx % len(self.responses)]
        self._call_idx += 1
        return response
