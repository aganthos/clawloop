"""Tests for lfx.wrap() — SDK wrapper for live mode."""

import time
import uuid

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


class TestWrapTaskIdAndSessionId:
    def test_task_id_is_uuid(self) -> None:
        """task_id should be a uuid hex, not a content hash."""
        captured = []
        client = MockLLMClient(responses=["ok", "ok"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        # Patch ingest to capture episodes
        orig_ingest = collector.ingest
        def capturing_ingest(messages, *, task_id="", session_id="", **kwargs):
            ep = orig_ingest(messages, task_id=task_id, session_id=session_id, **kwargs)
            captured.append(ep)
            return ep
        collector.ingest = capturing_ingest

        wrapped = wrap(client, collector=collector)
        wrapped.complete([{"role": "user", "content": "same question"}])
        wrapped.complete([{"role": "user", "content": "same question"}])

        # Two calls with same content should produce different task_ids
        assert captured[0].task_id != captured[1].task_id
        # task_id should be a valid hex string (uuid4)
        assert len(captured[0].task_id) == 32
        int(captured[0].task_id, 16)  # should not raise

    def test_session_id_populated(self) -> None:
        captured = []
        client = MockLLMClient(responses=["ok"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        orig_ingest = collector.ingest
        def capturing_ingest(messages, *, task_id="", session_id="", **kwargs):
            ep = orig_ingest(messages, task_id=task_id, session_id=session_id, **kwargs)
            captured.append(ep)
            return ep
        collector.ingest = capturing_ingest

        wrapped = wrap(client, collector=collector)
        wrapped.complete([{"role": "user", "content": "Hello world"}])

        assert captured[0].session_id != ""
        assert len(captured[0].session_id) == 16


class TestCollectorStateIdProvider:
    def test_default_state_id(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        from lfx.core.episode import Message
        ep = collector.ingest(
            [Message(role="user", content="hi")],
            task_id="t1", session_id="s1",
        )
        assert ep.state_id == "live"

    def test_string_state_id(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]), batch_size=100, state_id="custom-v1",
        )
        from lfx.core.episode import Message
        ep = collector.ingest(
            [Message(role="user", content="hi")],
            task_id="t1", session_id="s1",
        )
        assert ep.state_id == "custom-v1"

    def test_callable_state_id(self) -> None:
        counter = [0]
        def state_provider() -> str:
            counter[0] += 1
            return f"state-{counter[0]}"

        collector = EpisodeCollector(
            pipeline=RewardPipeline([]), batch_size=100, state_id=state_provider,
        )
        from lfx.core.episode import Message
        ep1 = collector.ingest(
            [Message(role="user", content="hi")],
            task_id="t1", session_id="s1",
        )
        ep2 = collector.ingest(
            [Message(role="user", content="bye")],
            task_id="t2", session_id="s1",
        )
        assert ep1.state_id == "state-1"
        assert ep2.state_id == "state-2"


from lfx.completion import CompletionResult
from lfx.core.episode import TokenLogProb, TokenUsage, ToolCall


class TestRichWrapperCapture:
    """Verify wrapper extracts rich metadata from CompletionResult."""

    def test_returns_completion_result(self) -> None:
        client = MockLLMClient(responses=["hello"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        result = wrapped.complete([{"role": "user", "content": "hi"}])
        assert isinstance(result, CompletionResult)
        assert result == "hello"

    def test_captures_model_on_episode(self) -> None:
        client = MockLLMClient(responses=["ok"], model="gpt-4o")
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        wrapped.complete([{"role": "user", "content": "hi"}])
        ep = list(collector._episode_index.values())[0]
        assert ep.model == "gpt-4o"

    def test_captures_logprobs_on_assistant_message(self) -> None:
        lps = [TokenLogProb(token="ok", logprob=-0.1)]
        client = MockLLMClient(responses=["ok"], logprobs=[lps])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        wrapped.complete([{"role": "user", "content": "hi"}])
        ep = list(collector._episode_index.values())[0]
        assistant_msg = [m for m in ep.messages if m.role == "assistant"][0]
        assert assistant_msg.logprobs is not None
        assert assistant_msg.logprobs[0].logprob == -0.1

    def test_captures_tool_calls_on_assistant_message(self) -> None:
        tc = ToolCall(id="tc-1", name="search", arguments='{"q":"x"}')
        client = MockLLMClient(responses=["searching"], tool_calls=[[tc]])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        wrapped.complete([{"role": "user", "content": "find x"}])
        ep = list(collector._episode_index.values())[0]
        assistant_msg = [m for m in ep.messages if m.role == "assistant"][0]
        assert assistant_msg.tool_calls is not None
        assert assistant_msg.tool_calls[0].name == "search"

    def test_captures_timing(self) -> None:
        client = MockLLMClient(responses=["ok"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        wrapped.complete([{"role": "user", "content": "hi"}])
        ep = list(collector._episode_index.values())[0]
        assert ep.summary.timing is not None
        assert ep.summary.timing.total_ms >= 0

    def test_captures_created_at(self) -> None:
        import time as _time
        client = MockLLMClient(responses=["ok"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        before = _time.time()
        wrapped.complete([{"role": "user", "content": "hi"}])
        after = _time.time()
        ep = list(collector._episode_index.values())[0]
        assert ep.created_at is not None
        assert before <= ep.created_at <= after

    def test_parses_tool_calls_from_input_messages(self) -> None:
        client = MockLLMClient(responses=["done"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        wrapped.complete([
            {"role": "user", "content": "search for x"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc-1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q":"x"}'},
                    }
                ],
            },
            {"role": "tool", "content": "found x", "tool_call_id": "tc-1", "name": "search"},
        ])
        ep = list(collector._episode_index.values())[0]
        asst_msgs = [m for m in ep.messages if m.role == "assistant"]
        assert asst_msgs[0].tool_calls is not None
        assert asst_msgs[0].tool_calls[0].name == "search"
        tool_msgs = [m for m in ep.messages if m.role == "tool"]
        assert tool_msgs[0].tool_call_id == "tc-1"
        assert tool_msgs[0].name == "search"

    def test_plain_string_client_still_works(self) -> None:
        class PlainClient:
            def complete(self, messages, **kwargs):
                return "plain text"

        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(PlainClient(), collector=collector)
        result = wrapped.complete([{"role": "user", "content": "hi"}])
        assert result == "plain text"
        assert isinstance(result, CompletionResult)
