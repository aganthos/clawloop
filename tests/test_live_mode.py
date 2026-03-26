"""Integration test: full live mode pipeline end-to-end."""

import time

from clawloop.collector import EpisodeCollector
from clawloop.completion import CompletionResult
from clawloop.core.episode import TokenLogProb, ToolCall, TokenUsage
from clawloop.core.loop import AgentState
from clawloop.core.reward import RewardPipeline
from clawloop.exporters.skyrl import SkyRLExporter
from clawloop.extractors.execution import ExecutionExtractor
from clawloop.extractors.user_feedback import UserFeedbackExtractor
from clawloop.layers.harness import Playbook, PlaybookEntry
from clawloop.learner import AsyncLearner
from clawloop.llm import MockLLMClient
from clawloop.wrapper import wrap


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


class TestRichPipelineEndToEnd:
    """Full pipeline: MockLLMClient with rich metadata → wrap → collector → exporter."""

    def test_logprobs_flow_through_to_exporter(self) -> None:
        """Logprobs on MockLLMClient → wrapper → Episode → SkyRL rollout_logprobs."""
        lps = [
            TokenLogProb(token="Here", logprob=-0.1),
            TokenLogProb(token=" is", logprob=-0.2),
            TokenLogProb(token=" help", logprob=-0.3),
        ]
        client = MockLLMClient(
            responses=["Here is help"],
            model="gpt-4o",
            logprobs=[lps],
        )

        collected_episodes = []

        pipeline = RewardPipeline([])
        collector = EpisodeCollector(
            pipeline=pipeline,
            batch_size=100,
            on_batch=lambda eps: collected_episodes.extend(eps),
        )
        wrapped = wrap(client, collector=collector)
        result = wrapped.complete([{"role": "user", "content": "help me"}])

        # Result is CompletionResult
        assert isinstance(result, CompletionResult)
        assert result.model == "gpt-4o"
        assert result.logprobs is not None

        # Episode captured with rich metadata
        ep = list(collector._episode_index.values())[0]
        assert ep.model == "gpt-4o"

        assistant_msg = [m for m in ep.messages if m.role == "assistant"][0]
        assert assistant_msg.logprobs is not None
        assert len(assistant_msg.logprobs) == 3

        # Exporter wires logprobs through
        from tests.test_skyrl_export import FakeTokenizer
        exporter = SkyRLExporter(tokenizer=FakeTokenizer())
        exported = exporter.export([ep])
        assert exported["rollout_logprobs"] is not None
        assert exported["rollout_logprobs"][0] == [-0.1, -0.2, -0.3]

    def test_tool_calls_captured_end_to_end(self) -> None:
        """Tool calls on response flow through to Episode messages."""
        tc = ToolCall(id="tc-1", name="search", arguments='{"q":"x"}')
        client = MockLLMClient(
            responses=["I found x"],
            tool_calls=[[tc]],
        )
        pipeline = RewardPipeline([])
        collector = EpisodeCollector(pipeline=pipeline, batch_size=100)
        wrapped = wrap(client, collector=collector)
        result = wrapped.complete([{"role": "user", "content": "find x"}])

        assert result.tool_calls is not None
        ep = list(collector._episode_index.values())[0]
        assistant_msg = [m for m in ep.messages if m.role == "assistant"][0]
        assert assistant_msg.tool_calls[0].name == "search"

    def test_provider_without_logprobs_works(self) -> None:
        """A client returning no logprobs should work fine — None flows through."""
        client = MockLLMClient(responses=["hello"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        result = wrapped.complete([{"role": "user", "content": "hi"}])

        assert result.logprobs is None
        ep = list(collector._episode_index.values())[0]
        assistant_msg = [m for m in ep.messages if m.role == "assistant"][0]
        assert assistant_msg.logprobs is None
