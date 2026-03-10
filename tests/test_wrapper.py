"""Tests for lfx.wrap() — SDK wrapper for live mode."""

import time

from lfx.collector import EpisodeCollector
from lfx.core.loop import AgentState
from lfx.core.reward import RewardPipeline
from lfx.learner import AsyncLearner
from lfx.llm import MockLLMClient
from lfx.wrapper import wrap


class TestWrap:
    def test_wrap_returns_callable(self) -> None:
        client = MockLLMClient(responses=["hello"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        assert hasattr(wrapped, "complete")

    def test_wrap_passes_through_response(self) -> None:
        client = MockLLMClient(responses=["The answer is 42"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        result = wrapped.complete([{"role": "user", "content": "What is 6*7?"}])
        assert result == "The answer is 42"

    def test_wrap_creates_episode(self) -> None:
        client = MockLLMClient(responses=["The answer is 42"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        wrapped.complete([{"role": "user", "content": "What is 6*7?"}])
        assert collector.metrics["episodes_collected"] == 1

    def test_wrap_with_learner_triggers_batch(self) -> None:
        client = MockLLMClient(responses=["Here is a detailed response."])
        state = AgentState()
        learner = AsyncLearner(agent_state=state)
        learner.start()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=learner.on_batch,
        )
        wrapped = wrap(client, collector=collector)

        wrapped.complete([{"role": "user", "content": "q1"}])
        wrapped.complete([{"role": "user", "content": "q2"}])

        time.sleep(0.5)
        assert learner.metrics["batches_trained"] >= 1
        learner.stop()
