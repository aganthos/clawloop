"""Tests for LfxCallback — litellm integration for Mode B capture."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

from clawloop.callbacks.litellm_cb import LfxCallback
from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline


def _make_mock_response(
    content: str = "Hello",
    model: str = "gpt-4o",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    tool_calls: list | None = None,
    logprobs_content: list | None = None,
) -> MagicMock:
    """Build a mock litellm ModelResponse."""
    response = MagicMock()
    response.model = model

    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = tool_calls

    if logprobs_content is not None:
        lp_mock = MagicMock()
        lp_mock.content = logprobs_content
        choice.logprobs = lp_mock
    else:
        choice.logprobs = None

    response.choices = [choice]

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = prompt_tokens + completion_tokens
    response.usage = usage

    return response


class TestLfxCallbackBasic:
    def test_log_success_creates_episode(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {"messages": [{"role": "user", "content": "What is 2+2?"}], "model": "gpt-4o"}
        response = _make_mock_response(content="4")
        start = time.time()
        cb.log_success_event(kwargs, response, start, start + 0.1)
        assert collector.metrics["episodes_collected"] == 1

    def test_captures_model(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_mock_response(model="gpt-4o-2024-08-06")
        cb.log_success_event(kwargs, response, time.time(), time.time())
        ep = list(collector._episode_index.values())[0]
        assert ep.model == "gpt-4o-2024-08-06"

    def test_captures_usage(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_mock_response(prompt_tokens=50, completion_tokens=20)
        cb.log_success_event(kwargs, response, time.time(), time.time())
        ep = list(collector._episode_index.values())[0]
        assert ep.summary.token_usage is not None
        assert ep.summary.token_usage.prompt_tokens == 50

    def test_captures_timing(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_mock_response()
        start = time.time()
        cb.log_success_event(kwargs, response, start, start + 0.25)
        ep = list(collector._episode_index.values())[0]
        assert ep.summary.timing is not None
        assert ep.summary.timing.total_ms >= 200

    def test_captures_logprobs(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        lp1 = MagicMock()
        lp1.token = "Hello"
        lp1.logprob = -0.3
        lp1.token_id = None
        lp1.top_logprobs = None
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_mock_response(content="Hello", logprobs_content=[lp1])
        cb.log_success_event(kwargs, response, time.time(), time.time())
        ep = list(collector._episode_index.values())[0]
        assistant_msg = [m for m in ep.messages if m.role == "assistant"][0]
        assert assistant_msg.logprobs is not None
        assert assistant_msg.logprobs[0].token == "Hello"
        assert assistant_msg.logprobs[0].logprob == -0.3

    def test_captures_tool_calls(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        tc = MagicMock()
        tc.id = "tc-1"
        tc.function.name = "search"
        tc.function.arguments = '{"q": "x"}'
        kwargs = {"messages": [{"role": "user", "content": "search x"}]}
        response = _make_mock_response(content="", tool_calls=[tc])
        cb.log_success_event(kwargs, response, time.time(), time.time())
        ep = list(collector._episode_index.values())[0]
        assistant_msg = [m for m in ep.messages if m.role == "assistant"][0]
        assert assistant_msg.tool_calls is not None
        assert assistant_msg.tool_calls[0].name == "search"

    def test_triggers_batch(self) -> None:
        batches: list[list] = []
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=lambda eps: batches.append(eps),
        )
        cb = LfxCallback(collector=collector)
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_mock_response(content="hello there friend")
        cb.log_success_event(kwargs, response, time.time(), time.time())
        cb.log_success_event(kwargs, response, time.time(), time.time())
        assert len(batches) == 1

    def test_session_id_from_metadata(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {
            "messages": [{"role": "user", "content": "hi"}],
            "metadata": {"session_id": "sess-abc"},
        }
        response = _make_mock_response()
        cb.log_success_event(kwargs, response, time.time(), time.time())
        ep = list(collector._episode_index.values())[0]
        assert ep.session_id == "sess-abc"

    def test_datetime_timing(self) -> None:
        """litellm passes datetime objects for start/end time."""
        from datetime import datetime, timedelta

        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_mock_response()
        start = datetime(2026, 3, 11, 12, 0, 0)
        end = start + timedelta(milliseconds=250)
        cb.log_success_event(kwargs, response, start, end)
        ep = list(collector._episode_index.values())[0]
        assert ep.summary.timing is not None
        assert abs(ep.summary.timing.total_ms - 250.0) < 1.0

    def test_vision_message_content(self) -> None:
        """List-type content (vision/multimodal) must not crash session_id hash."""
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }
            ],
        }
        response = _make_mock_response()
        cb.log_success_event(kwargs, response, time.time(), time.time())
        assert collector.metrics["episodes_collected"] == 1

    def test_input_tool_calls_parsed(self) -> None:
        """Tool calls on prior assistant messages in input should be parsed."""
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {
            "messages": [
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
            ],
        }
        response = _make_mock_response(content="Here is x")
        cb.log_success_event(kwargs, response, time.time(), time.time())
        ep = list(collector._episode_index.values())[0]
        asst_msgs = [m for m in ep.messages if m.role == "assistant"]
        assert asst_msgs[0].tool_calls is not None
        assert asst_msgs[0].tool_calls[0].name == "search"

    def test_error_handling_does_not_propagate(self) -> None:
        """Errors in _process should be logged, not propagated."""
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        # Malformed response — choices is empty
        response = MagicMock()
        response.choices = []
        cb.log_success_event(
            {"messages": [{"role": "user", "content": "hi"}]},
            response, time.time(), time.time(),
        )
        assert collector.metrics["episodes_collected"] == 0

    def test_none_content_not_stringified(self) -> None:
        """Messages with content=None (e.g. tool-call-only) must become '' not 'None'."""
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        kwargs = {
            "messages": [
                {"role": "user", "content": "call search"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "tc-1", "type": "function",
                     "function": {"name": "search", "arguments": '{"q":"x"}'}},
                ]},
                {"role": "tool", "content": "found x", "tool_call_id": "tc-1"},
            ],
        }
        response = _make_mock_response(content="Here is x")
        cb.log_success_event(kwargs, response, time.time(), time.time())
        ep = list(collector._episode_index.values())[0]
        asst_input = [m for m in ep.messages if m.role == "assistant"][0]
        assert asst_input.content == ""
        assert asst_input.content != "None"

    def test_log_failure_is_noop(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector)
        cb.log_failure_event(
            kwargs={"messages": []},
            response_obj=None,
            start_time=time.time(),
            end_time=time.time(),
            exception=ValueError("test"),
        )
        assert collector.metrics["episodes_collected"] == 0


class TestLfxCallbackTracer:
    def test_accepts_tracer_kwarg(self) -> None:
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector, tracer=None)
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_mock_response()
        cb.log_success_event(kwargs, response, time.time(), time.time())
        assert collector.metrics["episodes_collected"] == 1

    def test_emits_span_with_tracer(self) -> None:
        pytest = __import__("pytest")
        pytest.importorskip("opentelemetry")

        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        mem = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(mem))
        tracer = provider.get_tracer("test")

        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        cb = LfxCallback(collector=collector, tracer=tracer)
        kwargs = {"messages": [{"role": "user", "content": "hi"}]}
        response = _make_mock_response(content="hello")
        cb.log_success_event(kwargs, response, time.time(), time.time())

        spans = mem.get_finished_spans()
        assert len(spans) == 1
        assert spans[0].attributes.get("openinference.span.kind") == "LLM"
        assert collector.metrics["episodes_collected"] == 1
