"""CompletionResult — rich response type from LLM calls.

Wraps the text response with metadata: model ID, tool calls,
token usage, per-token log probabilities, and latency.  Behaves
like a ``str`` for backward compatibility (equality, hashing,
len, contains, format).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lfx.core.episode import TokenLogProb, TokenUsage, ToolCall


@dataclass
class CompletionResult:
    """Rich response from an LLM completion call.

    Carries the generated text plus all available metadata.
    String-compatible: ``str()``, ``==``, ``hash()``, ``len()``,
    ``in``, ``+``, and f-string formatting all delegate to ``.text``.

    **Equality and hashing are text-only by design.** Two results with
    the same text but different metadata (model, logprobs, etc.) are
    considered equal. Metadata is auxiliary, not identity.

    **Not a real str.** ``isinstance(result, str)`` returns ``False``.
    Methods like ``.startswith()``, ``.split()``, ``re.search()`` require
    ``str(result)`` or ``result.text``. Use those when you need a real str.

    **raw_response is ephemeral.** It holds the provider's raw response
    object for immediate inspection but should not be persisted — it can
    pin large objects in memory.
    """

    text: str
    model: str | None = None
    tool_calls: list[ToolCall] | None = None
    usage: TokenUsage | None = None
    logprobs: list[TokenLogProb] | None = None
    latency_ms: float | None = None
    raw_response: Any = field(default=None, repr=False)

    # -- String compatibility --

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"CompletionResult(text={self.text!r}, model={self.model!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.text == other
        if isinstance(other, CompletionResult):
            return self.text == other.text
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.text)

    def __bool__(self) -> bool:
        return bool(self.text)

    def __contains__(self, item: str) -> bool:
        return item in self.text

    def __len__(self) -> int:
        return len(self.text)

    def __add__(self, other: str) -> str:
        return self.text + other

    def __radd__(self, other: str) -> str:
        return other + self.text

    def __format__(self, format_spec: str) -> str:
        return format(self.text, format_spec)
