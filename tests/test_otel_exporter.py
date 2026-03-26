"""Tests for clawloop.exporters.otel — OTelExporter."""

from __future__ import annotations

import json
import sys
import time
from typing import Any
from unittest.mock import patch

import pytest

opentelemetry = pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from clawloop.core.episode import (
    Episode,
    EpisodeSummary,
    Message,
    StepMeta,
    Timing,
    TokenUsage,
    ToolCall,
)
from clawloop.core.reward import RewardSignal
from clawloop.exporters.otel import OTelExporter, _ms_to_ns, _to_ns


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

_BASE_TS = 1_700_000_000.0  # a fixed unix timestamp for deterministic tests


def _make_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Return a fresh (provider, in-memory exporter) pair."""
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    return provider, exp


def _make_otel_episode(
    *,
    n_steps: int = 2,
    reward: float = 0.8,
    with_tools: bool = False,
    with_timestamps: bool = True,
) -> Episode:
    """Build a minimal but realistic Episode for OTel export tests.

    Message layout per step (with_tools=True):
      - user message
      - assistant message (with a tool_call when with_tools)
      - tool result message (when with_tools)

    Message layout per step (with_tools=False):
      - user message
      - assistant message
    """
    messages: list[Message] = []
    step_boundaries: list[int] = []
    steps: list[StepMeta] = []

    # System prompt (before any step boundary)
    messages.append(
        Message(
            role="system",
            content="You are a helpful assistant.",
            timestamp=_BASE_TS if with_timestamps else None,
        )
    )

    per_step_ms: list[float] = []

    for t in range(n_steps):
        step_boundaries.append(len(messages))

        ts_user = (_BASE_TS + t * 10.0) if with_timestamps else None
        ts_asst = (_BASE_TS + t * 10.0 + 1.0) if with_timestamps else None

        messages.append(
            Message(
                role="user",
                content=f"Step {t} user input",
                timestamp=ts_user,
            )
        )

        tool_calls: list[ToolCall] | None = None
        if with_tools:
            tool_calls = [
                ToolCall(
                    id=f"tc-{t}-0",
                    name="search",
                    arguments=json.dumps({"q": f"query {t}"}),
                    result=f"search result {t}",
                    success=True,
                    latency_ms=50.0,
                )
            ]

        messages.append(
            Message(
                role="assistant",
                content=f"Step {t} response",
                model="gpt-4o",
                token_count=20 + t,
                timestamp=ts_asst,
                tool_calls=tool_calls,
            )
        )

        if with_tools:
            # Tool result message — linked by tool_call_id
            messages.append(
                Message(
                    role="tool",
                    content=f"search result {t}",
                    name="search",
                    tool_call_id=f"tc-{t}-0",
                    timestamp=ts_asst,
                )
            )

        is_terminal = t == n_steps - 1
        step_timing = 500.0
        per_step_ms.append(step_timing)
        steps.append(
            StepMeta(
                t=t,
                reward=reward if is_terminal else 0.0,
                done=is_terminal,
                timing_ms=step_timing,
            )
        )

    # Build signals
    mapped = reward * 2.0 - 1.0  # [0,1] → [-1,1]
    signals = {
        "outcome": RewardSignal(name="outcome", value=mapped, confidence=1.0),
        "execution": RewardSignal(name="execution", value=0.5, confidence=0.9),
    }
    summary = EpisodeSummary(
        signals=signals,
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=40, total_tokens=140),
        timing=Timing(total_ms=float(n_steps * 500), per_step_ms=per_step_ms),
        filtered=False,
    )

    return Episode(
        id="ep-test-001",
        state_id="state-abc",
        task_id="task-xyz",
        bench="test-bench",
        messages=messages,
        step_boundaries=step_boundaries,
        steps=steps,
        summary=summary,
        session_id="sess-123",
        model="gpt-4o",
        created_at=_BASE_TS if with_timestamps else None,
        metadata={"harness_version": "v2.1"},
    )


def _run_export(episode: Episode) -> list[Any]:
    """Export episode with a fresh in-memory provider; return sorted spans."""
    provider, exp = _make_provider()
    tracer = provider.get_tracer("test")
    exporter = OTelExporter(tracer=tracer)
    exporter.export_one(episode)
    return exp.get_finished_spans()


# ---------------------------------------------------------------------------
# TestHelpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_to_ns_zero(self) -> None:
        assert _to_ns(0.0) == 0

    def test_to_ns_one_second(self) -> None:
        assert _to_ns(1.0) == 1_000_000_000

    def test_to_ns_fractional(self) -> None:
        assert _to_ns(1.5) == 1_500_000_000

    def test_to_ns_large(self) -> None:
        assert _to_ns(_BASE_TS) == int(_BASE_TS * 1_000_000_000)

    def test_ms_to_ns_zero(self) -> None:
        assert _ms_to_ns(0.0) == 0

    def test_ms_to_ns_one_ms(self) -> None:
        assert _ms_to_ns(1.0) == 1_000_000

    def test_ms_to_ns_fractional(self) -> None:
        assert _ms_to_ns(0.5) == 500_000


# ---------------------------------------------------------------------------
# TestSpanTree
# ---------------------------------------------------------------------------


class TestSpanTree:
    def test_root_agent_span_exists(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        spans = _run_export(ep)
        names = [s.name for s in spans]
        assert any(n.startswith("episode:") for n in names)

    def test_root_span_has_no_parent(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        spans = _run_export(ep)
        roots = [s for s in spans if s.parent is None]
        assert len(roots) == 1

    def test_llm_spans_are_children_of_agent(self) -> None:
        ep = _make_otel_episode(n_steps=2)
        spans = _run_export(ep)
        root = next(s for s in spans if s.parent is None)
        llm_spans = [s for s in spans if s.name.startswith("llm:")]
        assert len(llm_spans) == 2
        for llm in llm_spans:
            assert llm.parent.span_id == root.context.span_id

    def test_tool_spans_are_grandchildren_of_agent(self) -> None:
        ep = _make_otel_episode(n_steps=2, with_tools=True)
        spans = _run_export(ep)
        root = next(s for s in spans if s.parent is None)
        llm_spans = {s.context.span_id: s for s in spans if s.name.startswith("llm:")}
        tool_spans = [s for s in spans if s.name.startswith("tool:")]

        # Each tool span's parent must be an LLM span
        for ts in tool_spans:
            assert ts.parent.span_id in llm_spans
            # LLM span's parent must be root
            llm = llm_spans[ts.parent.span_id]
            assert llm.parent.span_id == root.context.span_id

    def test_three_level_tree_structure(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        spans = _run_export(ep)
        root = next(s for s in spans if s.parent is None)
        llm_spans = [s for s in spans if s.name.startswith("llm:")]
        tool_spans = [s for s in spans if s.name.startswith("tool:")]
        assert len(llm_spans) == 1
        assert len(tool_spans) == 1
        assert llm_spans[0].parent.span_id == root.context.span_id
        assert tool_spans[0].parent.span_id == llm_spans[0].context.span_id

    def test_no_tool_spans_without_tools(self) -> None:
        ep = _make_otel_episode(n_steps=2, with_tools=False)
        spans = _run_export(ep)
        tool_spans = [s for s in spans if s.name.startswith("tool:")]
        assert len(tool_spans) == 0

    def test_span_count_no_tools(self) -> None:
        # 1 root + n_steps LLM spans
        ep = _make_otel_episode(n_steps=3, with_tools=False)
        spans = _run_export(ep)
        assert len(spans) == 1 + 3

    def test_span_count_with_tools(self) -> None:
        # 1 root + n_steps LLM + n_steps TOOL (one tool call per step)
        ep = _make_otel_episode(n_steps=2, with_tools=True)
        spans = _run_export(ep)
        assert len(spans) == 1 + 2 + 2


# ---------------------------------------------------------------------------
# TestClawLoopAttributes
# ---------------------------------------------------------------------------


class TestClawLoopAttributes:
    def _root(self, ep: Episode) -> Any:
        spans = _run_export(ep)
        return next(s for s in spans if s.parent is None)

    def test_episode_id(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["clawloop.episode.id"] == ep.id

    def test_state_id(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["clawloop.state.id"] == "state-abc"

    def test_task_id(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["clawloop.task.id"] == "task-xyz"

    def test_session_id(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["clawloop.session.id"] == "sess-123"

    def test_bench(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["clawloop.bench"] == "test-bench"

    def test_reward_effective(self) -> None:
        ep = _make_otel_episode(reward=0.8)
        root = self._root(ep)
        # effective_reward is in [-1, 1]; 0.8 → mapped = 0.6
        assert abs(root.attributes["clawloop.reward.effective"] - 0.6) < 1e-6

    def test_reward_normalized(self) -> None:
        ep = _make_otel_episode(reward=0.8)
        root = self._root(ep)
        assert abs(root.attributes["clawloop.reward.normalized"] - 0.8) < 1e-6

    def test_reward_effective_range_negative(self) -> None:
        ep = _make_otel_episode(reward=0.0)
        root = self._root(ep)
        # 0.0 → mapped = -1.0
        assert abs(root.attributes["clawloop.reward.effective"] - (-1.0)) < 1e-6

    def test_reward_normalized_range_full(self) -> None:
        ep = _make_otel_episode(reward=1.0)
        root = self._root(ep)
        assert abs(root.attributes["clawloop.reward.normalized"] - 1.0) < 1e-6

    def test_signals_json(self) -> None:
        ep = _make_otel_episode(reward=0.8)
        root = self._root(ep)
        raw = root.attributes["clawloop.reward.signals"]
        signals = json.loads(raw)
        assert "outcome" in signals
        assert "execution" in signals
        assert "value" in signals["outcome"]
        assert "confidence" in signals["outcome"]

    def test_per_signal_attrs(self) -> None:
        ep = _make_otel_episode(reward=0.8)
        root = self._root(ep)
        assert "clawloop.reward.outcome.value" in root.attributes
        assert "clawloop.reward.outcome.confidence" in root.attributes
        assert "clawloop.reward.execution.value" in root.attributes
        assert "clawloop.reward.execution.confidence" in root.attributes
        assert root.attributes["clawloop.reward.outcome.confidence"] == 1.0

    def test_harness_version(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["clawloop.harness.version"] == "v2.1"

    def test_harness_version_missing(self) -> None:
        ep = _make_otel_episode()
        ep.metadata = {}
        root = self._root(ep)
        assert "clawloop.harness.version" not in root.attributes

    def test_filtered_false(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["clawloop.filtered"] is False

    def test_filtered_true(self) -> None:
        ep = _make_otel_episode()
        ep.summary.filtered = True
        root = self._root(ep)
        assert root.attributes["clawloop.filtered"] is True

    def test_evaluation_event(self) -> None:
        ep = _make_otel_episode(reward=0.8)
        root = self._root(ep)
        events = root.events
        assert len(events) == 1
        assert events[0].name == "gen_ai.evaluation.result"
        # evaluation score uses effective_reward() ([-1,1] canonical range)
        # reward=0.8 → outcome signal value = 0.6, effective_reward() = 0.6
        assert abs(events[0].attributes["gen_ai.evaluation.score"] - 0.6) < 1e-6

    def test_input_tokens_on_root(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["gen_ai.usage.input_tokens"] == 100

    def test_no_output_tokens_on_root(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert "gen_ai.usage.output_tokens" not in root.attributes

    def test_conversation_id(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["gen_ai.conversation.id"] == "sess-123"

    def test_openinference_span_kind_agent(self) -> None:
        ep = _make_otel_episode()
        root = self._root(ep)
        assert root.attributes["openinference.span.kind"] == "AGENT"


# ---------------------------------------------------------------------------
# TestGenAiAttributes
# ---------------------------------------------------------------------------


class TestGenAiAttributes:
    def _llm_spans(self, ep: Episode) -> list[Any]:
        spans = _run_export(ep)
        return [s for s in spans if s.name.startswith("llm:")]

    def test_model_attribute(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        llm_spans = self._llm_spans(ep)
        assert len(llm_spans) == 1
        assert llm_spans[0].attributes["gen_ai.request.model"] == "gpt-4o"

    def test_input_messages_json_string(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        llm_spans = self._llm_spans(ep)
        raw = llm_spans[0].attributes["gen_ai.input.messages"]
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        # system message + user message should be in input
        roles = [m["role"] for m in parsed]
        assert "system" in roles
        assert "user" in roles

    def test_output_messages_json_string(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        llm_spans = self._llm_spans(ep)
        raw = llm_spans[0].attributes["gen_ai.output.messages"]
        parsed = json.loads(raw)
        assert isinstance(parsed, list)
        assert parsed[0]["role"] == "assistant"

    def test_output_tokens_on_llm(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        llm_spans = self._llm_spans(ep)
        assert "gen_ai.usage.output_tokens" in llm_spans[0].attributes
        assert llm_spans[0].attributes["gen_ai.usage.output_tokens"] == 20

    def test_openinference_span_kind_llm(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        llm_spans = self._llm_spans(ep)
        assert llm_spans[0].attributes["openinference.span.kind"] == "LLM"

    def test_llm_model_name_attribute(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        llm_spans = self._llm_spans(ep)
        assert "llm.model_name" in llm_spans[0].attributes
        assert llm_spans[0].attributes["llm.model_name"] == "gpt-4o"

    def test_llm_input_messages_oi_attr(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        llm_spans = self._llm_spans(ep)
        assert "llm.input_messages" in llm_spans[0].attributes

    def test_llm_output_messages_oi_attr(self) -> None:
        ep = _make_otel_episode(n_steps=1)
        llm_spans = self._llm_spans(ep)
        assert "llm.output_messages" in llm_spans[0].attributes


# ---------------------------------------------------------------------------
# TestToolResultLinking
# ---------------------------------------------------------------------------


class TestToolResultLinking:
    def _tool_spans(self, ep: Episode) -> list[Any]:
        spans = _run_export(ep)
        return [s for s in spans if s.name.startswith("tool:")]

    def test_tool_output_from_message(self) -> None:
        """tool.output should come from the linked tool-result message."""
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        tool_spans = self._tool_spans(ep)
        assert len(tool_spans) == 1
        assert tool_spans[0].attributes["tool.output"] == "search result 0"

    def test_tool_name_attribute(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        tool_spans = self._tool_spans(ep)
        assert tool_spans[0].attributes["tool.name"] == "search"

    def test_tool_parameters_not_double_encoded(self) -> None:
        """arguments is already a JSON string — must NOT be double-encoded."""
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        tool_spans = self._tool_spans(ep)
        raw = tool_spans[0].attributes["tool.parameters"]
        # raw should be valid JSON (not '\"{\\\\"q\\\\":...}'
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)
        assert "q" in parsed

    def test_fallback_to_result_field(self) -> None:
        """When no matching tool-result message, fall back to tc.result."""
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        # Remove tool-result messages so fallback triggers
        ep.messages = [m for m in ep.messages if m.role != "tool"]
        tool_spans = self._tool_spans(ep)
        assert tool_spans[0].attributes["tool.output"] == "search result 0"

    def test_openinference_span_kind_tool(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        tool_spans = self._tool_spans(ep)
        assert tool_spans[0].attributes["openinference.span.kind"] == "TOOL"


# ---------------------------------------------------------------------------
# TestToolErrors
# ---------------------------------------------------------------------------


class TestToolErrors:
    def test_failed_tool_error_status(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        # Mark the tool call as failed
        asst_msg = next(m for m in ep.messages if m.role == "assistant")
        asst_msg.tool_calls[0].success = False
        asst_msg.tool_calls[0].error = "connection timeout"

        spans = _run_export(ep)
        tool_spans = [s for s in spans if s.name.startswith("tool:")]
        assert len(tool_spans) == 1
        assert tool_spans[0].status.status_code == StatusCode.ERROR

    def test_failed_tool_exception_event(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        asst_msg = next(m for m in ep.messages if m.role == "assistant")
        asst_msg.tool_calls[0].success = False
        asst_msg.tool_calls[0].error = "connection timeout"

        spans = _run_export(ep)
        tool_spans = [s for s in spans if s.name.startswith("tool:")]
        events = tool_spans[0].events
        assert len(events) >= 1
        event_names = [e.name for e in events]
        assert "exception" in event_names

    def test_failed_tool_exception_message(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        asst_msg = next(m for m in ep.messages if m.role == "assistant")
        asst_msg.tool_calls[0].success = False
        asst_msg.tool_calls[0].error = "connection timeout"

        spans = _run_export(ep)
        tool_spans = [s for s in spans if s.name.startswith("tool:")]
        exc_events = [e for e in tool_spans[0].events if e.name == "exception"]
        assert exc_events[0].attributes["exception.message"] == "connection timeout"

    def test_successful_tool_no_error_status(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        spans = _run_export(ep)
        tool_spans = [s for s in spans if s.name.startswith("tool:")]
        assert tool_spans[0].status.status_code != StatusCode.ERROR

    def test_failed_tool_no_error_message_uses_fallback(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        asst_msg = next(m for m in ep.messages if m.role == "assistant")
        asst_msg.tool_calls[0].success = False
        asst_msg.tool_calls[0].error = None  # no error message provided

        spans = _run_export(ep)
        tool_spans = [s for s in spans if s.name.startswith("tool:")]
        exc_events = [e for e in tool_spans[0].events if e.name == "exception"]
        assert exc_events[0].attributes["exception.message"] == "tool call failed"


# ---------------------------------------------------------------------------
# TestTimestamps
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_agent_span_start_time(self) -> None:
        ep = _make_otel_episode(with_timestamps=True)
        spans = _run_export(ep)
        root = next(s for s in spans if s.parent is None)
        expected = _to_ns(_BASE_TS)
        assert root.start_time == expected

    def test_agent_span_end_time(self) -> None:
        ep = _make_otel_episode(n_steps=2, with_timestamps=True)
        spans = _run_export(ep)
        root = next(s for s in spans if s.parent is None)
        # 2 steps * 500 ms = 1000 ms
        expected_end = _to_ns(_BASE_TS) + _ms_to_ns(1000.0)
        assert root.end_time == expected_end

    def test_llm_span_start_time_from_message_timestamp(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_timestamps=True)
        spans = _run_export(ep)
        llm_spans = [s for s in spans if s.name.startswith("llm:")]
        asst_msg = next(m for m in ep.messages if m.role == "assistant")
        assert llm_spans[0].start_time == _to_ns(asst_msg.timestamp)

    def test_graceful_without_timestamps(self) -> None:
        ep = _make_otel_episode(with_timestamps=False)
        # Should not raise
        spans = _run_export(ep)
        root = next(s for s in spans if s.parent is None)
        # start time should be a positive integer (current time)
        assert root.start_time > 0

    def test_tool_duration_from_latency_ms(self) -> None:
        ep = _make_otel_episode(n_steps=1, with_tools=True)
        spans = _run_export(ep)
        tool_spans = [s for s in spans if s.name.startswith("tool:")]
        ts = tool_spans[0]
        # latency_ms = 50.0 → 50_000_000 ns duration
        assert ts.end_time - ts.start_time == _ms_to_ns(50.0)


# ---------------------------------------------------------------------------
# TestStepIndex
# ---------------------------------------------------------------------------


class TestStepIndex:
    def test_step_index_on_llm_spans(self) -> None:
        ep = _make_otel_episode(n_steps=3)
        spans = _run_export(ep)
        llm_spans = sorted(
            [s for s in spans if s.name.startswith("llm:")],
            key=lambda s: s.attributes["clawloop.step.index"],
        )
        assert len(llm_spans) == 3
        for i, ls in enumerate(llm_spans):
            assert ls.attributes["clawloop.step.index"] == i

    def test_step_reward_on_llm_spans(self) -> None:
        ep = _make_otel_episode(n_steps=2, reward=0.9)
        spans = _run_export(ep)
        llm_spans = sorted(
            [s for s in spans if s.name.startswith("llm:")],
            key=lambda s: s.attributes["clawloop.step.index"],
        )
        # First step reward = 0.0 (non-terminal)
        assert llm_spans[0].attributes["clawloop.step.reward"] == 0.0
        # Last step reward = 0.9 (terminal)
        assert abs(llm_spans[1].attributes["clawloop.step.reward"] - 0.9) < 1e-6

    def test_step_done_flag(self) -> None:
        ep = _make_otel_episode(n_steps=2)
        spans = _run_export(ep)
        llm_spans = sorted(
            [s for s in spans if s.name.startswith("llm:")],
            key=lambda s: s.attributes["clawloop.step.index"],
        )
        assert llm_spans[0].attributes["clawloop.step.done"] is False
        assert llm_spans[1].attributes["clawloop.step.done"] is True

    def test_step_index_on_tool_spans(self) -> None:
        ep = _make_otel_episode(n_steps=2, with_tools=True)
        spans = _run_export(ep)
        tool_spans = sorted(
            [s for s in spans if s.name.startswith("tool:")],
            key=lambda s: s.attributes["clawloop.step.index"],
        )
        assert len(tool_spans) == 2
        assert tool_spans[0].attributes["clawloop.step.index"] == 0
        assert tool_spans[1].attributes["clawloop.step.index"] == 1


# ---------------------------------------------------------------------------
# TestTracerInjection
# ---------------------------------------------------------------------------


class TestTracerInjection:
    def test_injected_tracer_is_used(self) -> None:
        provider, exp = _make_provider()
        tracer = provider.get_tracer("injected")
        exporter = OTelExporter(tracer=tracer)

        ep = _make_otel_episode(n_steps=1)
        exporter.export_one(ep)

        spans = exp.get_finished_spans()
        assert len(spans) > 0

    def test_global_provider_respected(self) -> None:
        """When a real global TracerProvider is set, OTelExporter should use it."""
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        provider, exp = _make_provider()
        original = trace.get_tracer_provider()
        try:
            trace.set_tracer_provider(provider)
            exporter = OTelExporter()  # no tracer injected
            ep = _make_otel_episode(n_steps=1)
            exporter.export_one(ep)
            spans = exp.get_finished_spans()
            assert len(spans) > 0
        finally:
            trace.set_tracer_provider(original)

    def test_no_provider_ownership_when_tracer_injected(self) -> None:
        """When tracer is injected, exporter must not own the provider."""
        provider, _ = _make_provider()
        tracer = provider.get_tracer("test")
        exporter = OTelExporter(tracer=tracer)
        assert exporter._provider is None

    def test_flush_noop_without_own_provider(self) -> None:
        provider, _ = _make_provider()
        tracer = provider.get_tracer("test")
        exporter = OTelExporter(tracer=tracer)
        # Should not raise
        exporter.flush()

    def test_flush_calls_force_flush_on_owned_provider(self) -> None:
        """When OTelExporter owns the provider, flush() must call force_flush."""
        from unittest.mock import MagicMock, patch as _patch

        provider, _ = _make_provider()
        tracer = provider.get_tracer("test")
        exporter = OTelExporter(tracer=tracer)

        # Simulate owned provider
        mock_provider = MagicMock()
        exporter._provider = mock_provider
        exporter.flush(timeout_millis=3000)
        mock_provider.force_flush.assert_called_once_with(timeout_millis=3000)


# ---------------------------------------------------------------------------
# TestStepIndexMultipleAssistants
# ---------------------------------------------------------------------------


class TestStepIndexMultipleAssistants:
    """Step index resolution when a single step has multiple assistant messages."""

    def test_two_assistants_in_same_step(self) -> None:
        """Two assistant messages within one step should both resolve to step 0."""
        messages = [
            Message(role="system", content="sys", timestamp=_BASE_TS),
            Message(role="user", content="q1", timestamp=_BASE_TS + 1),
            Message(role="assistant", content="thinking...", model="gpt-4o",
                    token_count=5, timestamp=_BASE_TS + 2),
            Message(role="assistant", content="done", model="gpt-4o",
                    token_count=10, timestamp=_BASE_TS + 3),
        ]
        ep = Episode(
            id="ep-multi-asst",
            state_id="s1",
            task_id="t1",
            bench="test",
            messages=messages,
            step_boundaries=[1],  # step 0 starts at msg index 1
            steps=[StepMeta(t=0, reward=1.0, done=True, timing_ms=500)],
            summary=EpisodeSummary(
                signals={"outcome": RewardSignal(name="outcome", value=0.8, confidence=1.0)},
                token_usage=TokenUsage(prompt_tokens=10, completion_tokens=15, total_tokens=25),
                timing=Timing(total_ms=500.0, per_step_ms=[500.0]),
                filtered=False,
            ),
            session_id="sess",
            model="gpt-4o",
            created_at=_BASE_TS,
        )

        spans = _run_export(ep)
        llm_spans = [s for s in spans if s.name.startswith("llm:")]
        assert len(llm_spans) == 2
        # Both should be step 0
        for ls in llm_spans:
            assert ls.attributes["clawloop.step.index"] == 0


# ---------------------------------------------------------------------------
# TestExportBatch
# ---------------------------------------------------------------------------


class TestExportBatch:
    def test_export_multiple_episodes(self) -> None:
        provider, exp = _make_provider()
        tracer = provider.get_tracer("test")
        exporter = OTelExporter(tracer=tracer)

        ep1 = _make_otel_episode(n_steps=1)
        ep2 = _make_otel_episode(n_steps=2)
        ep2.id = "ep-test-002"

        exporter.export([ep1, ep2])
        spans = exp.get_finished_spans()

        # ep1: 1+1=2 spans; ep2: 1+2=3 spans → total 5
        assert len(spans) == 5

    def test_export_empty_list(self) -> None:
        provider, exp = _make_provider()
        tracer = provider.get_tracer("test")
        exporter = OTelExporter(tracer=tracer)
        exporter.export([])
        spans = exp.get_finished_spans()
        assert len(spans) == 0

    def test_export_one_returns_none(self) -> None:
        provider, exp = _make_provider()
        tracer = provider.get_tracer("test")
        exporter = OTelExporter(tracer=tracer)
        ep = _make_otel_episode(n_steps=1)
        result = exporter.export_one(ep)
        assert result is None


# ---------------------------------------------------------------------------
# TestImportError
# ---------------------------------------------------------------------------


class TestImportError:
    def test_raises_without_otel(self) -> None:
        """OTelExporter.__init__ must raise ImportError when otel is missing."""
        # We simulate a missing opentelemetry by temporarily hiding it
        original_modules = {}
        otel_modules = [k for k in sys.modules if k.startswith("opentelemetry")]
        for mod in otel_modules:
            original_modules[mod] = sys.modules.pop(mod)

        try:
            with patch.dict("sys.modules", {"opentelemetry": None}):
                with pytest.raises(ImportError, match="opentelemetry"):
                    OTelExporter()
        finally:
            sys.modules.update(original_modules)


# ---------------------------------------------------------------------------
# TestOpenInferenceFallback
# ---------------------------------------------------------------------------


class TestOpenInferenceFallback:
    def test_string_literals_when_oi_missing(self) -> None:
        """When openinference is not installed, string literals are used."""
        # Import the module directly to check constants
        import importlib

        import clawloop.exporters.otel as otel_mod

        original_span_kind = otel_mod._SPAN_KIND_ATTR
        original_kind_agent = otel_mod._KIND_AGENT

        # Simulate import failure
        with patch.dict(
            "sys.modules",
            {"openinference.semconv.trace": None, "openinference": None, "openinference.semconv": None},
        ):
            # Force reimport to trigger fallback path
            try:
                import importlib

                importlib.reload(otel_mod)
                # After reload with missing module, should use string fallbacks
                assert otel_mod._SPAN_KIND_ATTR == "openinference.span.kind"
                assert otel_mod._KIND_AGENT == "AGENT"
                assert otel_mod._KIND_LLM == "LLM"
                assert otel_mod._KIND_TOOL == "TOOL"
                assert otel_mod._TOOL_NAME == "tool.name"
                assert otel_mod._TOOL_PARAMETERS == "tool.parameters"
            finally:
                # Restore module to working state
                importlib.reload(otel_mod)

    def test_fallback_values_match_spec(self) -> None:
        """Even if openinference IS installed, fallback strings must match the spec."""
        from clawloop.exporters.otel import (
            _KIND_AGENT,
            _KIND_LLM,
            _KIND_TOOL,
            _SPAN_KIND_ATTR,
            _TOOL_NAME,
            _TOOL_PARAMETERS,
        )

        # These are the expected string values regardless of which path was taken
        assert _SPAN_KIND_ATTR == "openinference.span.kind"
        assert _KIND_AGENT == "AGENT"
        assert _KIND_LLM == "LLM"
        assert _KIND_TOOL == "TOOL"
        assert _TOOL_NAME == "tool.name"
        assert _TOOL_PARAMETERS == "tool.parameters"
