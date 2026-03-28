"""Tests for built-in reward extractors."""

from clawloop.core.env import EvalResult, Sample, StaticTaskEnvironment
from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.reward import RewardSignal
from clawloop.extractors.execution import ExecutionExtractor
from clawloop.extractors.formatting import FormattingFilter
from clawloop.extractors.outcome import OutcomeExtractor
from clawloop.extractors.user_feedback import UserFeedbackExtractor


def _make_episode(messages: list[Message]) -> Episode:
    """Build a minimal Episode with the given messages."""
    return Episode(
        id=Episode.new_id(),
        state_id="test-state",
        task_id="task-001",
        bench="test",
        messages=messages,
        step_boundaries=[0] if messages else [],
        steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=1.0)],
        summary=EpisodeSummary(total_reward=0.0),
    )


class TestExecutionExtractor:
    def setup_method(self) -> None:
        self.extractor = ExecutionExtractor()

    def test_name_is_execution(self) -> None:
        assert self.extractor.name == "execution"

    def test_no_tool_messages_returns_none(self) -> None:
        ep = _make_episode([
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ])
        assert self.extractor.extract(ep) is None

    def test_empty_messages_returns_none(self) -> None:
        ep = _make_episode([])
        assert self.extractor.extract(ep) is None

    def test_error_keyword_gives_negative(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="Error: file not found", tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == -1.0
        assert result.confidence == 0.9

    def test_exception_keyword(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="Traceback (most recent call last):\n  ...", tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == -1.0

    def test_failure_keyword(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="Request failed with timeout", tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == -1.0

    def test_http_error_code_4xx(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="HTTP 404 Not Found", tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == -1.0
        assert result.confidence == 0.85

    def test_http_error_code_500(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="Status 503 Service Unavailable", tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == -1.0

    def test_empty_content_gives_negative(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="", tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == -0.5
        assert result.confidence == 0.5

    def test_minimal_content_gives_neutral(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="OK", tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == 0.0
        assert result.confidence == 0.3

    def test_content_exactly_50_chars(self) -> None:
        content = "x" * 50
        ep = _make_episode([
            Message(role="tool", content=content, tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == 0.0

    def test_substantial_content_gives_positive(self) -> None:
        ep = _make_episode([
            Message(
                role="tool",
                content="Here is a detailed result that contains more than fifty characters of output.",
                tool_call_id="tc-1",
            ),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == 0.5
        assert result.confidence == 0.6

    def test_mixed_signals_aggregated(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="Error: something broke", tool_call_id="tc-1"),
            Message(
                role="tool",
                content="Success! The operation completed with the following detailed output data.",
                tool_call_id="tc-2",
            ),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert abs(result.value - (-0.4)) < 1e-9
        assert abs(result.confidence - 0.75) < 1e-9

    def test_value_clamped_to_range(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="Error: first", tool_call_id="tc-1"),
            Message(role="tool", content="Exception thrown", tool_call_id="tc-2"),
            Message(role="tool", content="failure in system", tool_call_id="tc-3"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert -1.0 <= result.value <= 1.0

    def test_tool_message_with_none_content_skipped(self) -> None:
        ep = _make_episode([
            Message(role="tool", content=None, tool_call_id="tc-1"),  # type: ignore[arg-type]
        ])
        result = self.extractor.extract(ep)
        assert result is None

    def test_error_keyword_precedence_over_http_code(self) -> None:
        ep = _make_episode([
            Message(role="tool", content="Error 500: server failed", tool_call_id="tc-1"),
        ])
        result = self.extractor.extract(ep)
        assert result is not None
        assert result.value == -1.0
        assert result.confidence == 0.9


# ── Helper for extractor tests with assistant messages ────────────────


def _make_full_episode(
    question: str = "What is 2+2?",
    response: str = "The answer is 4.",
    reward: float = 1.0,
    signals: dict[str, RewardSignal] | None = None,
) -> Episode:
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content=question),
        Message(role="assistant", content=response),
    ]
    step = StepMeta(t=0, reward=reward, done=True, timing_ms=0.0)
    return Episode(
        id=Episode.new_id(),
        state_id="test",
        task_id=question,
        bench="test",
        messages=messages,
        step_boundaries=[0],
        steps=[step],
        summary=EpisodeSummary(
            total_reward=reward,
            signals=signals if signals is not None else {},
        ),
    )


# ── OutcomeExtractor ─────────────────────────────────────────────────


class TestOutcomeExtractor:
    def test_returns_signal_with_env(self) -> None:
        sample = Sample(question="What is 2+2?", ground_truth="4")
        env = StaticTaskEnvironment(
            tasks=[sample],
            evaluate_fn=lambda s, r: EvalResult(score=1.0),
        )
        extractor = OutcomeExtractor(env=env)
        ep = _make_full_episode(question="What is 2+2?", response="4")
        signal = extractor.extract(ep)
        assert signal is not None
        assert signal.name == "outcome"
        assert signal.value == 1.0
        assert signal.confidence == 1.0

    def test_zero_score_maps_to_minus_one(self) -> None:
        sample = Sample(question="What is 2+2?")
        env = StaticTaskEnvironment(
            tasks=[sample],
            evaluate_fn=lambda s, r: EvalResult(score=0.0),
        )
        extractor = OutcomeExtractor(env=env)
        ep = _make_full_episode(question="What is 2+2?", response="wrong")
        signal = extractor.extract(ep)
        assert signal is not None
        assert signal.value == -1.0

    def test_no_env_returns_none(self) -> None:
        extractor = OutcomeExtractor(env=None)
        ep = _make_full_episode()
        assert extractor.extract(ep) is None

    def test_no_user_message_returns_none(self) -> None:
        env = StaticTaskEnvironment(
            tasks=[Sample(question="q")],
            evaluate_fn=lambda s, r: EvalResult(score=1.0),
        )
        extractor = OutcomeExtractor(env=env)
        ep = Episode(
            id=Episode.new_id(),
            state_id="test",
            task_id="t",
            bench="test",
            messages=[Message(role="assistant", content="answer")],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=0.0)],
            summary=EpisodeSummary(),
        )
        assert extractor.extract(ep) is None

    def test_no_assistant_message_returns_none(self) -> None:
        env = StaticTaskEnvironment(
            tasks=[Sample(question="q")],
            evaluate_fn=lambda s, r: EvalResult(score=1.0),
        )
        extractor = OutcomeExtractor(env=env)
        ep = Episode(
            id=Episode.new_id(),
            state_id="test",
            task_id="t",
            bench="test",
            messages=[Message(role="user", content="q")],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=0.0)],
            summary=EpisodeSummary(),
        )
        assert extractor.extract(ep) is None


# ── UserFeedbackExtractor ────────────────────────────────────────────


class TestUserFeedbackExtractor:
    def test_no_feedback_returns_none(self) -> None:
        extractor = UserFeedbackExtractor()
        ep = _make_full_episode(signals={})
        assert extractor.extract(ep) is None

    def test_feedback_present_returns_signal(self) -> None:
        feedback = RewardSignal(name="user", value=0.8, confidence=1.0)
        extractor = UserFeedbackExtractor()
        ep = _make_full_episode(signals={"user": feedback})
        result = extractor.extract(ep)
        assert result is not None
        assert result.name == "user"
        assert result.value == 0.8
        assert result.confidence == 1.0

    def test_name_is_user(self) -> None:
        assert UserFeedbackExtractor().name == "user"


# ── FormattingFilter ─────────────────────────────────────────────────


class TestFormattingFilter:
    def test_passes_normal_response(self) -> None:
        filt = FormattingFilter()
        ep = _make_full_episode(response="This is a normal response that is long enough.")
        assert filt.passes(ep) is True

    def test_fails_empty_response(self) -> None:
        filt = FormattingFilter()
        ep = _make_full_episode(response="")
        assert filt.passes(ep) is False

    def test_fails_too_short_response(self) -> None:
        filt = FormattingFilter(min_response_length=20)
        ep = _make_full_episode(response="Short")
        assert filt.passes(ep) is False

    def test_fails_no_assistant_message(self) -> None:
        messages = [
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
        ]
        ep = Episode(
            id=Episode.new_id(),
            state_id="test",
            task_id="task",
            bench="test",
            messages=messages,
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=0.0)],
            summary=EpisodeSummary(),
        )
        filt = FormattingFilter()
        assert filt.passes(ep) is False

    def test_custom_min_length(self) -> None:
        filt = FormattingFilter(min_response_length=6)
        ep = _make_full_episode(response="Hello")
        assert filt.passes(ep) is False
        ep2 = _make_full_episode(response="Hello!")
        assert filt.passes(ep2) is True
