"""OTelExporter — Episode → OpenTelemetry / OpenInference span conversion.

Produces a three-level span tree per episode:
  AGENT root  →  one LLM span per assistant message
                   → one TOOL span per tool call within that message

Attributes follow the OpenInference semantic conventions where a matching
constant exists; custom lfx.* attributes cover everything else.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from lfx.core.episode import Episode
from lfx.exporters.base import TraceExporter

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenInference semconv — import with string-literal fallback
# ---------------------------------------------------------------------------

try:
    from openinference.semconv.trace import (  # type: ignore[import-untyped]
        OpenInferenceSpanKindValues,
        SpanAttributes as OISpanAttributes,
    )

    _SPAN_KIND_ATTR: str = OISpanAttributes.OPENINFERENCE_SPAN_KIND
    _KIND_AGENT: str = OpenInferenceSpanKindValues.AGENT.value
    _KIND_LLM: str = OpenInferenceSpanKindValues.LLM.value
    _KIND_TOOL: str = OpenInferenceSpanKindValues.TOOL.value
    _LLM_MODEL_NAME: str = OISpanAttributes.LLM_MODEL_NAME
    _LLM_INPUT_MESSAGES: str = OISpanAttributes.LLM_INPUT_MESSAGES
    _LLM_OUTPUT_MESSAGES: str = OISpanAttributes.LLM_OUTPUT_MESSAGES
    _LLM_TOKEN_COUNT_COMPLETION: str = OISpanAttributes.LLM_TOKEN_COUNT_COMPLETION
    _LLM_TOKEN_COUNT_PROMPT: str = OISpanAttributes.LLM_TOKEN_COUNT_PROMPT
    _TOOL_NAME: str = OISpanAttributes.TOOL_NAME
    _TOOL_PARAMETERS: str = OISpanAttributes.TOOL_PARAMETERS
except ImportError:
    _log.warning(
        "openinference-semantic-conventions not installed; "
        "falling back to string literals for span attributes"
    )
    _SPAN_KIND_ATTR = "openinference.span.kind"
    _KIND_AGENT = "AGENT"
    _KIND_LLM = "LLM"
    _KIND_TOOL = "TOOL"
    _LLM_MODEL_NAME = "llm.model_name"
    _LLM_INPUT_MESSAGES = "llm.input_messages"
    _LLM_OUTPUT_MESSAGES = "llm.output_messages"
    _LLM_TOKEN_COUNT_COMPLETION = "llm.token_count.completion"
    _LLM_TOKEN_COUNT_PROMPT = "llm.token_count.prompt"
    _TOOL_NAME = "tool.name"
    _TOOL_PARAMETERS = "tool.parameters"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_ns(unix_seconds: float) -> int:
    """Convert a Unix timestamp (seconds, float) to nanoseconds (int)."""
    return int(unix_seconds * 1_000_000_000)


def _ms_to_ns(ms: float) -> int:
    """Convert milliseconds to nanoseconds (int)."""
    return int(ms * 1_000_000)


# ---------------------------------------------------------------------------
# OTelExporter
# ---------------------------------------------------------------------------


class OTelExporter(TraceExporter):
    """Export :class:`~lfx.core.episode.Episode` objects as OTel/OpenInference spans.

    Parameters
    ----------
    tracer:
        An already-configured :class:`opentelemetry.trace.Tracer`.  When
        provided the exporter uses it directly and does **not** create or own
        a :class:`~opentelemetry.sdk.trace.TracerProvider`.
    endpoint:
        OTLP/HTTP collector endpoint (only used when we create our own provider).
    service_name:
        ``service.name`` resource attribute (only used when we create our own
        provider).
    resource_attributes:
        Extra key/value pairs to merge into the OTel Resource.
    """

    def __init__(
        self,
        *,
        tracer: Any = None,
        endpoint: str = "http://localhost:4318",
        service_name: str = "lfx",
        resource_attributes: dict[str, Any] | None = None,
    ) -> None:
        # Lazy import — OTel is an optional dependency.
        try:
            import opentelemetry  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "opentelemetry-api and opentelemetry-sdk are required for OTelExporter. "
                "Install with: pip install 'lfx[otel]'"
            ) from exc

        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.trace import ProxyTracerProvider

        self._provider: TracerProvider | None = None

        if tracer is not None:
            # Caller owns the tracer/provider — we just use it.
            self._tracer = tracer
            return

        # Check whether the application has already configured a global provider.
        global_provider = trace.get_tracer_provider()
        if not isinstance(global_provider, ProxyTracerProvider):
            # Real provider already set — piggy-back on it.
            self._tracer = global_provider.get_tracer("lfx.exporters.otel")
            return

        # No provider available — create our own with an OTLP HTTP exporter.
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        res_attrs: dict[str, Any] = {"service.name": service_name}
        if resource_attributes:
            res_attrs.update(resource_attributes)

        resource = Resource.create(res_attrs)
        provider = TracerProvider(resource=resource)
        otlp_exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
        provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

        self._provider = provider
        self._tracer = provider.get_tracer("lfx.exporters.otel")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def flush(self, timeout_millis: int = 5000) -> None:
        """Flush any buffered spans.  Only meaningful when we own the provider."""
        if self._provider is not None:
            self._provider.force_flush(timeout_millis=timeout_millis)

    def export(self, episodes: list[Episode]) -> None:
        for ep in episodes:
            self._export_episode(ep)

    def export_one(self, episode: Episode) -> None:
        self._export_episode(episode)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_step(msg_index: int, step_boundaries: list[int]) -> int:
        """Return the step index that owns the message at *msg_index*.

        Scans ``step_boundaries`` to find the last boundary that is <=
        *msg_index*.  Returns 0 when *msg_index* precedes all boundaries.
        """
        step = 0
        for i, boundary in enumerate(step_boundaries):
            if boundary <= msg_index:
                step = i
            else:
                break
        return step

    def _input_messages_for(
        self,
        msg_index: int,
        episode: Episode,
    ) -> list[dict[str, Any]]:
        """Return the full conversation history up to (but not including) *msg_index*.

        Uses the complete history because each LLM call in an agentic loop
        sees the entire prior trajectory (system prompt, reasoning, tool
        calls, tool results, prior assistant turns).
        """
        return [m.to_openai_dict() for m in episode.messages[:msg_index]]

    def _export_episode(self, ep: Episode) -> None:
        from opentelemetry import trace
        from opentelemetry.trace import SpanKind, StatusCode

        summary = ep.summary

        # ----------------------------------------------------------------
        # Compute root span timing
        # ----------------------------------------------------------------
        start_ns = _to_ns(ep.created_at) if ep.created_at else _to_ns(time.time())
        if summary.timing and summary.timing.total_ms:
            end_ns = start_ns + _ms_to_ns(summary.timing.total_ms)
        else:
            end_ns = None

        # ----------------------------------------------------------------
        # Build root AGENT span
        # ----------------------------------------------------------------
        root_span = self._tracer.start_span(
            name=f"episode:{ep.id}",
            kind=SpanKind.SERVER,
            start_time=start_ns,
        )

        # Core OI attribute
        root_span.set_attribute(_SPAN_KIND_ATTR, _KIND_AGENT)

        # Episode identity
        root_span.set_attribute("lfx.episode.id", ep.id)
        root_span.set_attribute("lfx.state.id", ep.state_id)
        root_span.set_attribute("lfx.task.id", ep.task_id)
        if ep.session_id:
            root_span.set_attribute("lfx.session.id", ep.session_id)
            root_span.set_attribute("gen_ai.conversation.id", ep.session_id)
        if ep.bench:
            root_span.set_attribute("lfx.bench", ep.bench)

        # Rewards
        root_span.set_attribute("lfx.reward.effective", summary.effective_reward())
        root_span.set_attribute("lfx.reward.normalized", summary.normalized_reward())
        root_span.set_attribute(
            "lfx.reward.signals",
            json.dumps(
                {
                    name: {"value": sig.value, "confidence": sig.confidence}
                    for name, sig in summary.signals.items()
                }
            ),
        )
        root_span.set_attribute("lfx.filtered", summary.filtered)

        # Per-signal attributes
        for name, sig in summary.signals.items():
            root_span.set_attribute(f"lfx.reward.{name}.value", sig.value)
            root_span.set_attribute(f"lfx.reward.{name}.confidence", sig.confidence)

        # Token usage
        if summary.token_usage:
            root_span.set_attribute(
                "gen_ai.usage.input_tokens", summary.token_usage.prompt_tokens
            )

        # Metadata
        harness_version = ep.metadata.get("harness_version")
        if harness_version is not None:
            root_span.set_attribute("lfx.harness.version", str(harness_version))

        # Evaluation event
        root_span.add_event(
            "gen_ai.evaluation.result",
            {"gen_ai.evaluation.score": summary.effective_reward()},
        )

        # ----------------------------------------------------------------
        # Build LLM + TOOL child spans
        # ----------------------------------------------------------------
        root_ctx = trace.set_span_in_context(root_span)
        per_step_ms = summary.timing.per_step_ms if summary.timing else []

        # Build a lookup: tool_call_id -> result content from tool messages
        tool_result_map: dict[str, str] = {}
        for msg in ep.messages:
            if msg.role == "tool" and msg.tool_call_id is not None:
                tool_result_map[msg.tool_call_id] = msg.content

        # Walk messages to find assistant turns
        llm_cursor_ns = start_ns  # running clock for LLM span starts

        for msg_idx, msg in enumerate(ep.messages):
            if msg.role != "assistant":
                continue

            step_idx = self._resolve_step(msg_idx, ep.step_boundaries)

            # Determine LLM span timing
            llm_start_ns: int
            llm_end_ns: int | None = None

            if msg.timestamp is not None:
                llm_start_ns = _to_ns(msg.timestamp)
            else:
                llm_start_ns = llm_cursor_ns

            if step_idx < len(per_step_ms) and per_step_ms[step_idx]:
                step_dur_ns = _ms_to_ns(per_step_ms[step_idx])
                llm_end_ns = llm_start_ns + step_dur_ns
                llm_cursor_ns = llm_end_ns
            else:
                llm_cursor_ns = llm_start_ns

            # LLM span
            llm_span = self._tracer.start_span(
                name=f"llm:step{step_idx}",
                context=root_ctx,
                kind=SpanKind.CLIENT,
                start_time=llm_start_ns,
            )
            llm_span.set_attribute(_SPAN_KIND_ATTR, _KIND_LLM)

            model = msg.model or ep.model
            if model:
                llm_span.set_attribute(_LLM_MODEL_NAME, model)
                llm_span.set_attribute("gen_ai.request.model", model)

            # Input messages: everything seen before this assistant turn
            input_msgs = self._input_messages_for(msg_idx, ep)
            llm_span.set_attribute(_LLM_INPUT_MESSAGES, json.dumps(input_msgs))
            llm_span.set_attribute("gen_ai.input.messages", json.dumps(input_msgs))

            # Output = the assistant message itself
            output_msgs = [msg.to_openai_dict()]
            llm_span.set_attribute(_LLM_OUTPUT_MESSAGES, json.dumps(output_msgs))
            llm_span.set_attribute("gen_ai.output.messages", json.dumps(output_msgs))

            # Token count
            if msg.token_count is not None:
                llm_span.set_attribute(_LLM_TOKEN_COUNT_COMPLETION, msg.token_count)
                llm_span.set_attribute("gen_ai.usage.output_tokens", msg.token_count)

            # Step metadata
            llm_span.set_attribute("lfx.step.index", step_idx)
            if step_idx < len(ep.steps):
                step_meta = ep.steps[step_idx]
                llm_span.set_attribute("lfx.step.reward", step_meta.reward)
                llm_span.set_attribute("lfx.step.done", step_meta.done)

            # TOOL grandchildren
            llm_ctx = trace.set_span_in_context(llm_span)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_start_ns = llm_start_ns  # start at same time as LLM by default
                    tc_end_ns: int | None = None
                    if tc.latency_ms is not None:
                        tc_end_ns = tc_start_ns + _ms_to_ns(tc.latency_ms)

                    tool_span = self._tracer.start_span(
                        name=f"tool:{tc.name}",
                        context=llm_ctx,
                        kind=SpanKind.INTERNAL,
                        start_time=tc_start_ns,
                    )
                    tool_span.set_attribute(_SPAN_KIND_ATTR, _KIND_TOOL)
                    tool_span.set_attribute(_TOOL_NAME, tc.name)
                    tool_span.set_attribute("tool.name", tc.name)
                    tool_span.set_attribute(_TOOL_PARAMETERS, tc.arguments)
                    tool_span.set_attribute("tool.parameters", tc.arguments)

                    # Tool output: prefer linked tool-result message, else .result
                    tool_output = tool_result_map.get(tc.id)
                    if tool_output is None:
                        tool_output = tc.result
                    if tool_output is not None:
                        tool_span.set_attribute("tool.output", tool_output)

                    tool_span.set_attribute("lfx.step.index", step_idx)

                    # Error handling
                    if tc.success is False:
                        from opentelemetry.trace import Status

                        tool_span.set_status(Status(StatusCode.ERROR))
                        error_msg = tc.error or "tool call failed"
                        tool_span.add_event(
                            "exception",
                            {
                                "exception.type": "ToolCallError",
                                "exception.message": error_msg,
                            },
                        )

                    tool_span.end(end_time=tc_end_ns)

            llm_span.end(end_time=llm_end_ns)

        # End root span
        root_span.end(end_time=end_ns)
