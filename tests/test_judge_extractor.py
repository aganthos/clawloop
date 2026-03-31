"""Tests for JudgeExtractor (LLM-as-judge reward)."""
from dataclasses import dataclass

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.reward import RewardPipeline
from clawloop.reward_extractors.judge import JudgeExtractor


@dataclass
class FakeCompletion:
    text: str


class FakeLLM:
    """Deterministic mock LLM for testing."""
    def __init__(self, responses: list[str]):
        self._responses = iter(responses)

    def complete(self, messages, **kwargs):
        return FakeCompletion(text=next(self._responses))


def _make_episode(instruction: str, response: str) -> Episode:
    return Episode(
        id=Episode.new_id(),
        state_id="test",
        task_id="test",
        bench="test",
        messages=[
            Message(role="user", content=instruction),
            Message(role="assistant", content=response),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=1.0)],
        summary=EpisodeSummary(),
    )


class TestJudgeExtractor:
    def test_positive_majority(self):
        llm = FakeLLM(["1", "1", "0"])
        judge = JudgeExtractor(client=llm, n_votes=3)
        ep = _make_episode("Say hello", "Hello! How can I help?")
        sig = judge.extract(ep)
        assert sig is not None
        assert sig.name == "judge"
        assert sig.value == 1.0
        assert sig.confidence > 0.0

    def test_negative_majority(self):
        llm = FakeLLM(["-1", "-1", "0"])
        judge = JudgeExtractor(client=llm, n_votes=3)
        ep = _make_episode("Write Python code", "I cannot help with that.")
        sig = judge.extract(ep)
        assert sig is not None
        assert sig.value == -1.0

    def test_neutral_majority(self):
        llm = FakeLLM(["0", "0", "1"])
        judge = JudgeExtractor(client=llm, n_votes=3)
        ep = _make_episode("Explain X", "X is a thing.")
        sig = judge.extract(ep)
        assert sig is not None
        assert sig.value == 0.0

    def test_no_assistant_message_returns_none(self):
        llm = FakeLLM(["1"])
        judge = JudgeExtractor(client=llm, n_votes=1)
        ep = Episode(
            id=Episode.new_id(), state_id="", task_id="", bench="",
            messages=[Message(role="user", content="hi")],
            step_boundaries=[0], steps=[], summary=EpisodeSummary(),
        )
        assert judge.extract(ep) is None

    def test_unparseable_responses_skipped(self):
        llm = FakeLLM(["great job!", "1", "awesome"])
        judge = JudgeExtractor(client=llm, n_votes=3)
        ep = _make_episode("Say hi", "Hi!")
        sig = judge.extract(ep)
        # Only 1 valid vote out of 3
        assert sig is not None
        assert sig.value == 1.0

    def test_confidence_capped(self):
        llm = FakeLLM(["1", "1", "1"])
        judge = JudgeExtractor(client=llm, n_votes=3)
        ep = _make_episode("X", "Y")
        sig = judge.extract(ep)
        # 3/3 agreement → confidence = 1.0 * 0.8 cap = 0.8
        assert sig.confidence == pytest.approx(0.8)

    def test_integrates_with_pipeline(self):
        """JudgeExtractor works in the RewardPipeline."""
        llm = FakeLLM(["1"])
        judge = JudgeExtractor(client=llm, n_votes=1)
        pipeline = RewardPipeline([judge])
        ep = _make_episode("Say hello", "Hello!")
        pipeline.enrich(ep)
        assert "judge" in ep.summary.signals
        assert ep.summary.signals["judge"].value == 1.0

    def test_pipeline_skips_judge_when_not_needed(self):
        """Pipeline skips judge when high-confidence signals exist."""
        from clawloop.reward_extractors.execution import ExecutionExtractor

        llm = FakeLLM(["should not be called"])
        judge = JudgeExtractor(client=llm, n_votes=1)
        pipeline = RewardPipeline([ExecutionExtractor(), judge])

        # Episode with tool message containing error → execution extractor fires
        ep = Episode(
            id=Episode.new_id(), state_id="", task_id="", bench="",
            messages=[
                Message(role="user", content="run code"),
                Message(role="tool", content="Error: file not found"),
                Message(role="assistant", content="There was an error."),
            ],
            step_boundaries=[0], steps=[], summary=EpisodeSummary(),
        )
        pipeline.enrich(ep)
        # Execution extractor should fire, judge should be skipped
        assert "execution" in ep.summary.signals
