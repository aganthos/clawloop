"""Tests for clawloop.core.episode."""

from clawloop.core.episode import (
    MAX_LOGPROBS_PER_MESSAGE,
    Episode,
    EpisodeSummary,
    LearningUpdate,
    Message,
    StepMeta,
    Timing,
    TokenLogProb,
    TokenUsage,
    ToolCall,
    cap_logprobs,
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


class TestEpisodeOptionalFields:
    """New optional fields have defaults and don't break existing call sites."""

    def test_defaults(self) -> None:
        ep = _make_episode()
        assert ep.session_id == ""
        assert ep.model is None
        assert ep.created_at is None
        assert ep.metadata == {}

    def test_explicit_values(self) -> None:
        ep = _make_episode()
        # Override after creation (dataclass fields)
        ep2 = Episode(
            id=ep.id,
            state_id=ep.state_id,
            task_id=ep.task_id,
            bench=ep.bench,
            messages=ep.messages,
            step_boundaries=ep.step_boundaries,
            steps=ep.steps,
            summary=ep.summary,
            session_id="sess-abc",
            model="claude-opus-4-6",
            created_at=1234567890.0,
            metadata={"source": "test"},
        )
        assert ep2.session_id == "sess-abc"
        assert ep2.model == "claude-opus-4-6"
        assert ep2.created_at == 1234567890.0
        assert ep2.metadata == {"source": "test"}

    def test_existing_call_sites_unaffected(self) -> None:
        """Creating Episode with only required fields still works."""
        ep = Episode(
            id="ep-test",
            state_id="s0",
            task_id="t0",
            bench="test",
            messages=[Message(role="user", content="hi")],
            step_boundaries=[0],
            steps=[],
            summary=EpisodeSummary(),
        )
        assert ep.id == "ep-test"
        assert ep.session_id == ""


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


class TestTokenLogProb:
    def test_basic_construction(self) -> None:
        lp = TokenLogProb(token="Hello", token_id=1234, logprob=-0.5)
        assert lp.token == "Hello"
        assert lp.token_id == 1234
        assert lp.logprob == -0.5
        assert lp.top_logprobs is None

    def test_with_top_logprobs(self) -> None:
        lp = TokenLogProb(
            token="Hello",
            logprob=-0.5,
            top_logprobs={"Hello": -0.5, "Hi": -1.2},
        )
        assert lp.top_logprobs == {"Hello": -0.5, "Hi": -1.2}

    def test_defaults(self) -> None:
        lp = TokenLogProb(token="x")
        assert lp.token_id is None
        assert lp.logprob == 0.0
        assert lp.top_logprobs is None

    def test_frozen(self) -> None:
        import pytest

        lp = TokenLogProb(token="x", logprob=-0.1)
        with pytest.raises(AttributeError):
            lp.token = "y"  # type: ignore[misc]


class TestCapLogprobs:
    def test_cap_under_limit(self) -> None:
        lps = [TokenLogProb(token=f"t{i}", logprob=-0.1) for i in range(10)]
        assert cap_logprobs(lps) is lps  # no copy needed

    def test_cap_over_limit(self) -> None:
        lps = [
            TokenLogProb(token=f"t{i}", logprob=-0.1)
            for i in range(MAX_LOGPROBS_PER_MESSAGE + 100)
        ]
        capped = cap_logprobs(lps)
        assert len(capped) == MAX_LOGPROBS_PER_MESSAGE

    def test_cap_none(self) -> None:
        assert cap_logprobs(None) is None


class TestMessageLogprobs:
    def test_message_default_no_logprobs(self) -> None:
        msg = Message(role="assistant", content="hello")
        assert msg.logprobs is None

    def test_message_with_logprobs(self) -> None:
        lps = [
            TokenLogProb(token="hello", logprob=-0.3),
            TokenLogProb(token=" world", logprob=-0.7),
        ]
        msg = Message(role="assistant", content="hello world", logprobs=lps)
        assert len(msg.logprobs) == 2
        assert msg.logprobs[0].logprob == -0.3

    def test_logprobs_not_in_openai_dict(self) -> None:
        """logprobs are internal metadata, not part of the OpenAI wire format."""
        lps = [TokenLogProb(token="hi", logprob=-0.1)]
        msg = Message(role="assistant", content="hi", logprobs=lps)
        d = msg.to_openai_dict()
        assert "logprobs" not in d


class TestMessageReasoningContent:
    def test_default_is_none(self) -> None:
        msg = Message(role="assistant", content="hi")
        assert msg.reasoning_content is None

    def test_stored_alongside_content(self) -> None:
        msg = Message(
            role="assistant",
            content="final answer",
            reasoning_content="step by step thinking",
        )
        assert msg.content == "final answer"
        assert msg.reasoning_content == "step by step thinking"

    def test_empty_string_preserved(self) -> None:
        msg = Message(role="assistant", content="x", reasoning_content="")
        assert msg.reasoning_content == ""

    def test_not_in_openai_dict(self) -> None:
        """to_openai_dict() is the OpenAI Chat Completions request shape.
        reasoning_content is an internal record field — must not be emitted.
        """
        msg = Message(role="assistant", content="x", reasoning_content="y")
        d = msg.to_openai_dict()
        assert "reasoning_content" not in d
        assert "reasoning" not in d
        assert d == {"role": "assistant", "content": "x"}

    def test_openai_dict_roundtrip_is_lossy(self) -> None:
        """Document the contract: Message -> to_openai_dict -> Message loses
        reasoning_content. Future maintainers must not assume lossless
        round-trips through the OpenAI wire format."""
        original = Message(role="assistant", content="x", reasoning_content="y")
        d = original.to_openai_dict()
        reconstructed = Message(role=d["role"], content=d["content"])
        assert reconstructed.reasoning_content is None
