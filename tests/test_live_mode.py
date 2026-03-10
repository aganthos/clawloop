"""Integration test: full live mode pipeline end-to-end."""

import time

from lfx.collector import EpisodeCollector
from lfx.core.loop import AgentState
from lfx.core.reward import RewardPipeline
from lfx.extractors.execution import ExecutionExtractor
from lfx.extractors.user_feedback import UserFeedbackExtractor
from lfx.layers.harness import Playbook, PlaybookEntry
from lfx.learner import AsyncLearner
from lfx.llm import MockLLMClient
from lfx.wrapper import wrap


class TestLiveModeEndToEnd:
    def test_wrap_collect_learn_cycle(self) -> None:
        state = AgentState()
        state.harness.playbook = Playbook(entries=[
            PlaybookEntry(id="tip-1", content="Be concise"),
        ])

        learner = AsyncLearner(agent_state=state, active_layers=["harness"])
        learner.start()

        pipeline = RewardPipeline([
            ExecutionExtractor(),
            UserFeedbackExtractor(),
        ])
        collector = EpisodeCollector(
            pipeline=pipeline,
            batch_size=3,
            on_batch=learner.on_batch,
        )

        client = MockLLMClient(responses=["Here is a helpful response with details."])
        wrapped = wrap(client, collector=collector)

        for i in range(3):
            result = wrapped.complete([{"role": "user", "content": f"Question {i}"}])
            assert result == "Here is a helpful response with details."

        time.sleep(1.0)

        assert learner.metrics["batches_trained"] >= 1
        assert collector.metrics["episodes_collected"] == 3

        learner.stop()

    def test_user_feedback_overrides_computed_reward(self) -> None:
        pipeline = RewardPipeline([ExecutionExtractor()])
        collector = EpisodeCollector(
            pipeline=pipeline,
            batch_size=100,
        )

        client = MockLLMClient(responses=["response"])
        wrapped = wrap(client, collector=collector)

        wrapped.complete([{"role": "user", "content": "test"}])
        ep_id = list(collector._episode_index.keys())[0]

        assert collector.submit_feedback(ep_id, -1.0) is True

        ep = collector._episode_index[ep_id]
        assert ep.summary.effective_reward() == -1.0
