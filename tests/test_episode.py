"""Tests for lfx.core.episode."""

from lfx.core.episode import (
    Episode,
    EpisodeSummary,
    LearningUpdate,
    Message,
    StepMeta,
    Timing,
    TokenUsage,
    ToolCall,
)


def _make_episode(
    n_steps: int = 3,
    reward: float = 1.0,
    bench: str = "test",
) -> Episode:
    """Helper to build a minimal Episode for testing."""
    messages: list[Message] = []
    step_boundaries: list[int] = []
    steps: list[StepMeta] = []

    # System prompt
    messages.append(Message(role="system", content="You are a helpful assistant."))

    for t in range(n_steps):
        step_boundaries.append(len(messages))
        # User turn
        messages.append(Message(role="user", content=f"Task step {t}"))
        # Assistant turn
        messages.append(
            Message(
                role="assistant",
                content=f"Response {t}",
                model="test-model",
                token_count=10,
            )
        )
        is_terminal = t == n_steps - 1
        steps.append(
            StepMeta(
                t=t,
                reward=reward if is_terminal else 0.0,
                done=is_terminal,
                timing_ms=100.0,
                info={},
            )
        )

    return Episode(
        id=Episode.new_id(),
        state_id="test-state",
        task_id="task-001",
        bench=bench,
        messages=messages,
        step_boundaries=step_boundaries,
        steps=steps,
        summary=EpisodeSummary(
            total_reward=reward,
            score_breakdown={"functional": 0.8, "efficiency": 0.6},
            token_usage=TokenUsage(prompt_tokens=30, completion_tokens=30, total_tokens=60),
            timing=Timing(total_ms=300.0, per_step_ms=[100.0] * n_steps),
        ),
    )


class TestMessage:
    def test_to_openai_dict_basic(self) -> None:
        msg = Message(role="user", content="Hello")
        d = msg.to_openai_dict()
        assert d == {"role": "user", "content": "Hello"}

    def test_to_openai_dict_with_tool_calls(self) -> None:
        tc = ToolCall(
            id="tc-1",
            name="search",
            arguments='{"q": "test"}',
            result="found it",
            success=True,
        )
        msg = Message(role="assistant", content="Let me search.", tool_calls=[tc])
        d = msg.to_openai_dict()
        assert d["role"] == "assistant"
        assert len(d["tool_calls"]) == 1
        assert d["tool_calls"][0]["function"]["name"] == "search"

    def test_to_openai_dict_tool_result(self) -> None:
        msg = Message(role="tool", content="result", name="search", tool_call_id="tc-1")
        d = msg.to_openai_dict()
        assert d["role"] == "tool"
        assert d["name"] == "search"
        assert d["tool_call_id"] == "tc-1"


class TestEpisode:
    def test_new_id_unique(self) -> None:
        ids = {Episode.new_id() for _ in range(100)}
        assert len(ids) == 100

    def test_n_steps(self) -> None:
        ep = _make_episode(n_steps=5)
        assert ep.n_steps() == 5

    def test_terminal_reward(self) -> None:
        ep = _make_episode(reward=0.75)
        assert ep.terminal_reward() == 0.75

    def test_terminal_reward_empty(self) -> None:
        ep = _make_episode(n_steps=0, reward=0.0)
        # Override with empty steps
        ep.steps = []
        assert ep.terminal_reward() == 0.0

    def test_messages_for_step(self) -> None:
        ep = _make_episode(n_steps=3)
        step_0_msgs = ep.messages_for_step(0)
        # Each step has user + assistant = 2 messages
        assert len(step_0_msgs) == 2
        assert step_0_msgs[0].role == "user"
        assert step_0_msgs[1].role == "assistant"

    def test_messages_for_last_step(self) -> None:
        ep = _make_episode(n_steps=3)
        last_msgs = ep.messages_for_step(2)
        assert len(last_msgs) == 2

    def test_to_openai_messages(self) -> None:
        ep = _make_episode(n_steps=2)
        msgs = ep.to_openai_messages()
        assert isinstance(msgs, list)
        # 1 system + 2 steps x 2 messages = 5
        assert len(msgs) == 5
        assert all(isinstance(m, dict) for m in msgs)

    def test_step_boundaries_consistency(self) -> None:
        ep = _make_episode(n_steps=4)
        assert len(ep.step_boundaries) == 4
        # Each boundary should be within message list bounds
        for idx in ep.step_boundaries:
            assert 0 <= idx < len(ep.messages)


class TestStepMeta:
    def test_fields(self) -> None:
        step = StepMeta(t=0, reward=0.5, done=True, timing_ms=42.0, info={"key": "val"})
        assert step.t == 0
        assert step.reward == 0.5
        assert step.done is True
        assert step.info["key"] == "val"


class TestLearningUpdate:
    def test_creation(self) -> None:
        ep = _make_episode()
        update = LearningUpdate(
            layer_type="harness",
            state_id_before="aaa",
            state_id_after="bbb",
            proposal={"prompts": {"test": "new prompt"}},
            evidence=[ep],
            decision="accept",
        )
        assert update.layer_type == "harness"
        assert update.decision == "accept"
        assert len(update.evidence) == 1
        assert update.created_at > 0
