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
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from lfx.core.reward import RewardSignal


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


class EpisodeSummary:
    """Aggregate metrics for a completed episode.

    Primary reward storage is ``signals``, a dict of named
    :class:`~lfx.core.reward.RewardSignal` instances.  The ``total_reward``
    property provides backward compatibility with the old float field: reading
    it returns :meth:`normalized_reward` (in [0, 1]), and writing it stores the
    value as an ``"outcome"`` signal converted to [-1, 1].
    """

    # -- Priority order for effective_reward (highest first) --
    _PRIORITY: tuple[str, ...] = ("user", "outcome", "execution", "judge")

    def __init__(
        self,
        *,
        total_reward: float | None = None,
        signals: dict[str, RewardSignal] | None = None,
        score_breakdown: dict[str, float] | None = None,
        token_usage: TokenUsage | None = None,
        timing: Timing | None = None,
        filtered: bool = False,
    ) -> None:
        self.signals: dict[str, RewardSignal] = signals if signals is not None else {}
        self.score_breakdown = score_breakdown
        self.token_usage = token_usage
        self.timing = timing
        self.filtered = filtered
        if total_reward is not None:
            self.total_reward = total_reward  # calls the setter

    # -- total_reward property (backward compat) --------------------------

    @property
    def total_reward(self) -> float:
        """Return :meth:`normalized_reward` so old code sees [0, 1]."""
        return self.normalized_reward()

    @total_reward.setter
    def total_reward(self, value: float) -> None:
        """Accept a [0, 1] value and store as outcome signal in [-1, 1]."""
        from lfx.core.reward import RewardSignal

        mapped = float(value) * 2.0 - 1.0
        self.signals["outcome"] = RewardSignal(
            name="outcome", value=mapped, confidence=1.0,
        )

    # -- Core reward methods ----------------------------------------------

    def effective_reward(self) -> float:
        """Priority-based effective reward in [-1, 1].

        Priority: user > outcome > execution (confidence >= 0.7) > judge.
        Falls back to 0.0 (neutral) when no qualifying signal exists.
        """
        for key in self._PRIORITY:
            sig = self.signals.get(key)
            if sig is None:
                continue
            if key == "execution" and sig.confidence < 0.7:
                continue
            return sig.value
        return 0.0

    def normalized_reward(self) -> float:
        """Map :meth:`effective_reward` from [-1, 1] to [0, 1]."""
        return (self.effective_reward() + 1.0) / 2.0

    def needs_judge(self) -> bool:
        """Return whether this episode still needs a judge signal.

        False when an ``outcome`` or ``user`` signal is present, or when
        ``execution`` has confidence >= 0.7.
        """
        if "outcome" in self.signals:
            return False
        if "user" in self.signals:
            return False
        ex = self.signals.get("execution")
        if ex is not None and ex.confidence >= 0.7:
            return False
        return True


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
