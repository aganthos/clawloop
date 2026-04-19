"""clawloop.wrap() — SDK wrapper that intercepts LLM calls for learning."""

from __future__ import annotations

import json as _json
import logging
import time
from typing import Any, Literal, get_args
from uuid import uuid4

from clawloop.collector import EpisodeCollector
from clawloop.completion import CompletionResult
from clawloop.core.episode import Message
from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.parse import _safe_session_hash, parse_tool_calls, resolve_oi_span_kind

log = logging.getLogger(__name__)

TraceLevel = Literal["minimal", "standard", "full"]
_VALID_TRACE_LEVELS: frozenset[str] = frozenset(get_args(TraceLevel))


class WrappedClient:
    """Drop-in LLMClient replacement that intercepts calls for learning."""

    def __init__(
        self,
        client: Any,
        collector: EpisodeCollector,
        *,
        tracer: Any = None,
        intensity: AdaptiveIntensity | None = None,
        cloud_url: str | None = None,
        cloud_api_key: str | None = None,
        trace_level: TraceLevel = "minimal",
    ) -> None:
        if trace_level not in _VALID_TRACE_LEVELS:
            raise ValueError(
                f"trace_level must be one of {sorted(_VALID_TRACE_LEVELS)}, got {trace_level!r}"
            )
        if cloud_url is not None and not cloud_url.strip():
            raise ValueError("cloud_url must be non-empty when provided")
        if cloud_api_key is not None and not cloud_api_key.strip():
            raise ValueError("cloud_api_key must be non-empty when provided")
        if cloud_url and not cloud_api_key:
            raise ValueError("cloud_api_key is required when cloud_url is set")
        self._client = client
        self._collector = collector
        self._tracer = tracer
        self._intensity = intensity
        self._cloud_url = cloud_url
        self._cloud_api_key = cloud_api_key
        self._trace_level = trace_level

        self._llm_kind_attr: str | None = None
        self._llm_kind_value: str | None = None
        if tracer:
            self._llm_kind_attr, self._llm_kind_value = resolve_oi_span_kind()

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> CompletionResult:
        # Record user activity for intensity gating
        if self._intensity is not None:
            self._intensity.record_user_activity()

        # Extract clawloop-specific kwargs before forwarding to the client
        task_id = kwargs.pop("task_id", uuid4().hex)

        _span = None
        if self._tracer:
            try:
                _span = self._tracer.start_span(
                    f"chat {kwargs.get('model', 'unknown')}",
                    attributes={
                        self._llm_kind_attr: self._llm_kind_value,
                        "gen_ai.operation.name": "chat",
                        "gen_ai.input.messages": _json.dumps(messages),
                    },
                )
            except Exception:
                log.warning("Failed to create OTel span", exc_info=True)
                _span = None

        start = time.monotonic()
        try:
            response = self._client.complete(messages, **kwargs)
        except Exception:
            if _span:
                try:
                    from opentelemetry.trace import Status, StatusCode

                    _span.set_status(Status(StatusCode.ERROR, "LLM call failed"))
                except Exception:
                    log.warning("Failed to set error status on OTel span", exc_info=True)
                finally:
                    _span.end()
            raise
        elapsed_ms = (time.monotonic() - start) * 1000

        # Normalize response to CompletionResult
        if isinstance(response, CompletionResult):
            result = response
            # Use client-reported latency if available, else our measurement
            if result.latency_ms is None:
                result = CompletionResult(
                    text=result.text,
                    model=result.model,
                    tool_calls=result.tool_calls,
                    usage=result.usage,
                    logprobs=result.logprobs,
                    latency_ms=elapsed_ms,
                    raw_response=result.raw_response,
                )
        else:
            result = CompletionResult(text=str(response), latency_ms=elapsed_ms)

        if _span:
            try:
                _span.set_attribute(
                    "gen_ai.output.messages",
                    _json.dumps([{"role": "assistant", "content": result.text}]),
                )
                _span.set_attribute(
                    "gen_ai.request.model",
                    result.model or "",
                )
                if result.usage:
                    _span.set_attribute(
                        "gen_ai.usage.output_tokens",
                        result.usage.completion_tokens,
                    )
            except Exception:
                log.warning("Failed to set attributes on OTel span", exc_info=True)
            finally:
                _span.end()

        # Build rich Message objects from input messages
        ep_messages: list[Message] = []
        for m in messages:
            ep_messages.append(
                Message(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    name=m.get("name"),
                    tool_calls=parse_tool_calls(m.get("tool_calls")),
                    tool_call_id=m.get("tool_call_id"),
                )
            )

        # Assistant response message with rich metadata
        ep_messages.append(
            Message(
                role="assistant",
                content=result.text,
                model=result.model,
                tool_calls=result.tool_calls,
                logprobs=result.logprobs,
                token_count=(result.usage.completion_tokens if result.usage else None),
                timestamp=time.time(),
            )
        )

        # Stable session_id from first user message
        session_id = ""
        for m in messages:
            if m.get("role") == "user":
                session_id = _safe_session_hash(m.get("content", ""))
                break

        self._collector.ingest(
            ep_messages,
            task_id=task_id,
            session_id=session_id,
            usage=result.usage,
            timing_ms=result.latency_ms,
            model=result.model,
        )

        return result


def wrap(
    client: Any,
    collector: EpisodeCollector,
    *,
    tracer: Any = None,
    intensity: AdaptiveIntensity | None = None,
    cloud_url: str | None = None,
    cloud_api_key: str | None = None,
    trace_level: TraceLevel = "minimal",
) -> WrappedClient:
    """Wrap an LLMClient with live-mode episode collection.

    Usage::

        # Local mode (default):
        wrapped = clawloop.wrap(my_client, collector=collector)
        result = wrapped.complete(messages)  # works exactly like before

        # With OTel tracing:
        wrapped = clawloop.wrap(my_client, collector=collector, tracer=my_tracer)

        # Cloud mode — send traces to a ClawLoop cloud endpoint.
        # cloud_api_key is required when cloud_url is set.
        # trace_level controls verbosity (stored now, used by future transport).
        wrapped = clawloop.wrap(
            my_client,
            collector=collector,
            cloud_url="https://api.clawloop.dev",
            cloud_api_key="cl-key-...",
            trace_level="standard",  # minimal | standard | full
        )
    """
    return WrappedClient(
        client,
        collector,
        tracer=tracer,
        intensity=intensity,
        cloud_url=cloud_url,
        cloud_api_key=cloud_api_key,
        trace_level=trace_level,
    )
