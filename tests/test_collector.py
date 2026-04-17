"""Tests for EpisodeCollector — live mode episode construction."""

import threading

from clawloop.collector import EpisodeCollector
from clawloop.core.episode import Message
from clawloop.core.reward import RewardPipeline
from clawloop.reward_extractors.formatting import FormattingFilter


class _TrackingCallback:
    """Records batches passed to on_batch."""
    def __init__(self):
        self.batches = []

    def __call__(self, episodes):
        self.batches.append(list(episodes))


class TestEpisodeCollector:
    def test_ingest_creates_episode(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        ep = collector.ingest(msgs, task_id="t1", session_id="s1")
        assert ep.id
        assert ep.bench == "live"
        assert ep.task_id == "t1"
        assert ep.session_id == "s1"
        assert len(ep.messages) == 2

    def test_batch_triggers_callback(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=cb,
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 20),
        ]
        collector.ingest(msgs, session_id="s1")
        assert len(cb.batches) == 0
        collector.ingest(msgs, session_id="s2")
        assert len(cb.batches) == 1
        assert len(cb.batches[0]) == 2

    def test_filtered_episodes_not_buffered(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=cb,
            formatting_filter=FormattingFilter(min_response_length=100),
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="short"),
        ]
        ep = collector.ingest(msgs, session_id="s1")
        assert ep.summary.filtered is True
        collector.ingest(msgs, session_id="s2")
        assert len(cb.batches) == 0

    def test_submit_feedback_updates_signal(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a long enough response here"),
        ]
        ep = collector.ingest(msgs, session_id="s1")
        assert collector.submit_feedback(ep.id, 1.0) is True
        assert "user" in ep.summary.signals
        assert ep.summary.signals["user"].value == 1.0

    def test_submit_feedback_unknown_id(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        assert collector.submit_feedback("nonexistent", 1.0) is False

    def test_thread_safety(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=10,
            on_batch=cb,
        )

        def ingest_many():
            for i in range(5):
                msgs = [
                    Message(role="user", content=f"q-{i}"),
                    Message(role="assistant", content="a" * 20),
                ]
                collector.ingest(msgs, session_id=f"s-{threading.current_thread().name}-{i}")

        threads = [threading.Thread(target=ingest_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_eps = sum(len(b) for b in cb.batches)
        assert total_eps == 20

    def test_metrics(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
            formatting_filter=FormattingFilter(min_response_length=100),
        )
        msgs_good = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 200),
        ]
        msgs_bad = [
            Message(role="user", content="q"),
            Message(role="assistant", content="short"),
        ]
        collector.ingest(msgs_good, session_id="s1")
        collector.ingest(msgs_bad, session_id="s2")
        collector.submit_feedback("nonexistent", 1.0)

        m = collector.metrics
        assert m["episodes_collected"] == 2
        assert m["episodes_filtered"] == 1
        assert m["feedback_received"] == 0
        assert m["feedback_missed"] == 1


from clawloop.core.episode import TokenUsage, Timing, TokenLogProb, ToolCall


class TestCollectorRichMetadata:
    def test_ingest_with_usage(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        ep = collector.ingest(msgs, task_id="t1", session_id="s1", usage=usage)
        assert ep.summary.token_usage is not None
        assert ep.summary.token_usage.total_tokens == 30

    def test_ingest_with_timing(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        ep = collector.ingest(msgs, task_id="t1", session_id="s1", timing_ms=150.5)
        assert ep.summary.timing is not None
        assert ep.summary.timing.total_ms == 150.5

    def test_ingest_with_model(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        ep = collector.ingest(msgs, task_id="t1", session_id="s1", model="gpt-4o")
        assert ep.model == "gpt-4o"

    def test_ingest_sets_created_at(self) -> None:
        import time
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        before = time.time()
        ep = collector.ingest(msgs, task_id="t1", session_id="s1")
        after = time.time()
        assert ep.created_at is not None
        assert before <= ep.created_at <= after


class TestIngestExternal:
    def test_basic_openai_messages(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        ep = collector.ingest_external(
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ],
            task_id="math-1",
            model="gpt-4o",
        )
        assert ep.task_id == "math-1"
        assert ep.model == "gpt-4o"
        assert ep.bench == "external"
        assert len(ep.messages) == 3

    def test_with_tool_calls(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        ep = collector.ingest_external(
            messages=[
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
                {"role": "assistant", "content": "Here is x."},
            ],
        )
        asst = ep.messages[1]
        assert asst.tool_calls is not None
        assert asst.tool_calls[0].name == "search"
        tool_msg = ep.messages[2]
        assert tool_msg.tool_call_id == "tc-1"

    def test_with_response_logprobs(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        ep = collector.ingest_external(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there friend"},
            ],
            response_logprobs=[
                {"token": "hello", "logprob": -0.3, "token_id": 1234},
            ],
        )
        assistant_msg = ep.messages[1]
        assert assistant_msg.logprobs is not None
        assert assistant_msg.logprobs[0].token == "hello"
        assert assistant_msg.logprobs[0].logprob == -0.3

    def test_with_per_message_logprobs(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        ep = collector.ingest_external(
            messages=[
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "hello there friend",
                    "logprobs": [{"token": "hello", "logprob": -0.3}],
                },
            ],
        )
        assert ep.messages[1].logprobs is not None
        assert ep.messages[1].logprobs[0].logprob == -0.3

    def test_with_usage_dict(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        ep = collector.ingest_external(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there friend"},
            ],
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )
        assert ep.summary.token_usage is not None
        assert ep.summary.token_usage.total_tokens == 8

    def test_custom_bench(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        ep = collector.ingest_external(
            messages=[
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello there friend"},
            ],
            bench="openclaw",
        )
        assert ep.bench == "openclaw"

    def test_external_episodes_trigger_batch(self) -> None:
        batches = []
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=lambda eps: batches.append(eps),
        )
        collector.ingest_external(
            messages=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a" * 20},
            ],
        )
        collector.ingest_external(
            messages=[
                {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "a" * 20},
            ],
        )
        assert len(batches) == 1

    def test_empty_messages_no_step_mismatch(self) -> None:
        """Ingesting empty messages should produce empty steps and step_boundaries."""
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        ep = collector.ingest([], task_id="t1", session_id="s1")
        assert ep.messages == []
        assert ep.step_boundaries == []
        assert ep.steps == []

    def test_external_episodes_get_reward_pipeline(self) -> None:
        from clawloop.reward_extractors.execution import ExecutionExtractor
        collector = EpisodeCollector(
            pipeline=RewardPipeline([ExecutionExtractor()]),
            batch_size=100,
        )
        ep = collector.ingest_external(
            messages=[
                {"role": "user", "content": "do something"},
                {"role": "assistant", "content": "calling tool"},
                {"role": "tool", "content": "Error: file not found"},
                {"role": "assistant", "content": "sorry, that failed"},
            ],
        )
        assert "execution" in ep.summary.signals
        assert ep.summary.signals["execution"].value < 0


class TestReasoningContentIngestion:
    """Reasoning normalization at the collector boundary."""

    def _collector(self):
        return EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)

    def test_preserves_canonical_reasoning_content_key(self):
        ep = self._collector().ingest_external([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "final",
             "reasoning_content": "step-by-step thinking"},
        ])
        asst = ep.messages[-1]
        assert asst.content == "final"
        assert asst.reasoning_content == "step-by-step thinking"

    def test_normalizes_legacy_reasoning_key(self):
        ep = self._collector().ingest_external([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "final", "reasoning": "legacy"},
        ])
        assert ep.messages[-1].reasoning_content == "legacy"

    def test_empty_content_falls_back_to_reasoning_for_compat(self):
        ep = self._collector().ingest_external([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "reasoning_content": "T"},
        ])
        asst = ep.messages[-1]
        assert asst.content == "T"
        assert asst.reasoning_content == "T"

    def test_both_content_and_reasoning_preserved_independently(self):
        ep = self._collector().ingest_external([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "final",
             "reasoning_content": "thinking"},
        ])
        asst = ep.messages[-1]
        assert asst.content == "final"
        assert asst.reasoning_content == "thinking"

    def test_empty_string_reasoning_preserved_not_coerced(self):
        ep = self._collector().ingest_external([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "reasoning_content": ""},
        ])
        assert ep.messages[-1].reasoning_content == ""

    def test_canonical_key_wins_over_legacy_when_both_present(self):
        ep = self._collector().ingest_external([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x",
             "reasoning_content": "canonical", "reasoning": "legacy"},
        ])
        assert ep.messages[-1].reasoning_content == "canonical"

    def test_no_reasoning_keys_leaves_field_none(self):
        ep = self._collector().ingest_external([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x"},
        ])
        assert ep.messages[-1].reasoning_content is None

    def test_canonical_none_falls_back_to_legacy(self):
        """When reasoning_content is explicitly None, fall back to the
        legacy reasoning key if it has a non-None value."""
        ep = self._collector().ingest_external([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x",
             "reasoning_content": None, "reasoning": "from-legacy"},
        ])
        assert ep.messages[-1].reasoning_content == "from-legacy"

    def test_content_fallback_only_for_assistant_role(self):
        """Reasoning-into-content fallback is assistant-only. A malformed
        non-assistant message with content=None doesn't get reasoning
        injected into content."""
        ep = self._collector().ingest_external([
            {"role": "user", "content": None, "reasoning": "leak"},
            {"role": "assistant", "content": "ok"},
        ])
        assert ep.messages[0].content == ""
        assert ep.messages[0].reasoning_content == "leak"


class TestReasoningContentE2E:
    """Integration: SSE parser → collector → Message. Covers the real
    regression boundary for reasoning preservation end-to-end."""

    def _collector(self):
        return EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)

    def test_sse_stream_with_reasoning_content_key(self):
        """OpenAI-style stream using reasoning_content delta key."""
        from clawloop.proxy_sse import parse_sse_bytes

        sse = (
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b'data: {"choices":[{"delta":{"reasoning_content":"let me think"}}]}\n\n'
            b'data: {"choices":[{"delta":{"reasoning_content":" carefully"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"42"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        msg, _usage, complete = parse_sse_bytes(sse)
        assert complete
        assert msg is not None

        ep = self._collector().ingest_external([
            {"role": "user", "content": "q"},
            msg,
        ])
        asst = ep.messages[-1]
        assert asst.content == "42"
        assert asst.reasoning_content == "let me think carefully"

    def test_sse_stream_with_ollama_reasoning_key(self):
        """Ollama-style stream using legacy reasoning delta key."""
        from clawloop.proxy_sse import parse_sse_bytes

        sse = (
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b'data: {"choices":[{"delta":{"reasoning":"ollama thinking"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"ans"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        msg, _usage, _ = parse_sse_bytes(sse)
        assert msg is not None

        ep = self._collector().ingest_external([
            {"role": "user", "content": "q"},
            msg,
        ])
        asst = ep.messages[-1]
        assert asst.content == "ans"
        assert asst.reasoning_content == "ollama thinking"

    def test_sse_reasoning_only_turn_back_compat(self):
        """When the SSE stream has reasoning but no content (thinking-only
        turn), the parser inlines reasoning into content. The collector
        should also set reasoning_content on the Message."""
        from clawloop.proxy_sse import parse_sse_bytes

        sse = (
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n'
            b'data: {"choices":[{"delta":{"reasoning_content":"deep thought"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        msg, _, _ = parse_sse_bytes(sse)
        assert msg is not None
        # Parser inlines reasoning into content when content is empty
        assert msg["content"] == "deep thought"

        ep = self._collector().ingest_external([
            {"role": "user", "content": "q"},
            msg,
        ])
        asst = ep.messages[-1]
        assert asst.content == "deep thought"
        assert asst.reasoning_content == "deep thought"
