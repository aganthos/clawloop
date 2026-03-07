"""Episode data structures — the spine of the LfX system.

An Episode records one complete agent interaction: a sequence of
state->action->reward transitions within a single trajectory.  Messages are
stored once in OpenAI chat format; ``step_boundaries`` index into the message
list to delineate agent turns without duplicating content.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class ToolCall:
    """A single tool invocation within an assistant message."""

    id: str
    name: str
    arguments: str  # JSON string
    result: str | None = None
    success: bool | None = None
    latency_ms: float | None = None
    error: str | None = None


@dataclass
class Message:
    """OpenAI-format chat message with optional tool-call metadata."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None  # tool name when role="tool"
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    timestamp: float | None = None
    token_count: int | None = None
    model: str | None = None  # which model generated this

    def to_openai_dict(self) -> dict[str, Any]:
        """Serialize to the dict format expected by OpenAI-compatible APIs."""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            d["name"] = self.name
        if self.tool_calls is not None:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class StepMeta:
    """Per-step metadata for one agent turn."""

    t: int  # step index
    reward: float  # 0.0 for intermediate, actual for terminal
    done: bool
    timing_ms: float
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Aggregate token counts for an episode."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Timing:
    """Wall-clock timing for an episode."""

    total_ms: float = 0.0
    per_step_ms: list[float] = field(default_factory=list)


@dataclass
class EpisodeSummary:
    """Aggregate metrics for a completed episode."""

    total_reward: float
    score_breakdown: dict[str, float] | None = None
    token_usage: TokenUsage | None = None
    timing: Timing | None = None


@dataclass
class Episode:
    """One complete agent trajectory.

    ``messages`` holds the full conversation in OpenAI format.
    ``step_boundaries`` lists the message indices where each agent turn begins,
    avoiding any duplication of message content.
    """

    id: str
    state_id: str  # hash of layers used
    task_id: str
    bench: str  # "entropic" | "car" | "tau2" | ...
    messages: list[Message]
    step_boundaries: list[int]  # indices into messages where each agent turn starts
    steps: list[StepMeta]
    summary: EpisodeSummary

    # -- Convenience helpers --

    @staticmethod
    def new_id() -> str:
        """Generate a unique episode identifier."""
        return uuid.uuid4().hex

    def n_steps(self) -> int:
        return len(self.steps)

    def terminal_reward(self) -> float:
        """Return the reward of the final (terminal) step."""
        if not self.steps:
            return 0.0
        return self.steps[-1].reward

    def messages_for_step(self, t: int) -> list[Message]:
        """Return the slice of messages belonging to step *t*."""
        start = self.step_boundaries[t]
        if t + 1 < len(self.step_boundaries):
            end = self.step_boundaries[t + 1]
        else:
            end = len(self.messages)
        return self.messages[start:end]

    def to_openai_messages(self) -> list[dict[str, Any]]:
        """Serialize the full conversation to OpenAI dict format."""
        return [m.to_openai_dict() for m in self.messages]


@dataclass
class LearningUpdate:
    """A proposed modification to one of the three learning layers."""

    layer_type: Literal["harness", "router", "weights"]
    state_id_before: str
    state_id_after: str
    proposal: dict[str, Any]  # diff-like payload
    evidence: list[Episode]
    decision: Literal["accept", "reject", "defer"]
    created_at: float = field(default_factory=time.time)
