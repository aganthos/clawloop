# Migration Plan: OTel + OpenInference Integration

## Status: REVISION 6 — addressing Codex review round 5

## Goal

Make lfx a good citizen of the OTel/OpenInference ecosystem so that any agent using lfx automatically gets traces flowing to Langfuse, Datadog, Phoenix, Grafana, or any OTLP-compatible backend — while keeping Episodes as the internal learning format.

## Principles

1. **OTel/OTLP is the transport** — we emit standard OTLP, not vendor-specific APIs
2. **OpenInference is the semantic layer** — we use its span kinds (LLM, TOOL, AGENT) and attribute conventions, extended with lfx-specific fields
3. **Episodes stay internal** — they're richer than OTel spans (step_boundaries, composable rewards, state_id). OTel export is a view, not a replacement
4. **Optional dependency** — `opentelemetry-api`/`opentelemetry-sdk` are extras, not required. lfx works without OTel installed
5. **No breaking changes** — everything is additive
6. **No provider conflicts** — respect existing OTel configurations; allow tracer/provider injection

## Current State

- `lfx/exporters/base.py` — `TraceExporter` protocol (export Episodes to external formats)
- `lfx/exporters/skyrl.py` — SkyRLExporter (Episodes → GRPO training data)
- `lfx/exporters/router_tuples.py` — RouterTupleExporter (Episodes → routing samples)
- `lfx/callbacks/litellm_cb.py` — LfxCallback (litellm hooks → EpisodeCollector)
- `lfx/wrapper.py` — WrappedClient (intercepts LLM calls → EpisodeCollector)
- No OTel, OpenInference, Langfuse, or tracing code exists

## Phase 1: OTel Exporter (Episode → OTLP spans)

**New file:** `lfx/exporters/otel.py`

Converts completed Episodes into OpenInference-compatible OTel spans and exports via OTLP.

### Span Mapping

Three-level tree: AGENT root → LLM children per assistant message → TOOL grandchildren per tool call. No intermediate "step" spans — step metadata is carried as attributes on LLM/TOOL spans. TOOL spans are children of the LLM span that initiated them, preserving the natural call→result relationship.

```
Episode (lfx)              →  OTel Span Tree
─────────────────────────      ─────────────────────────
Episode                    →  Root AGENT span
  session_id               →  gen_ai.conversation.id
  state_id                 →  lfx.state.id (custom attr)
  task_id                  →  lfx.task.id (custom attr)
  bench                    →  lfx.bench (custom attr)
  summary.effective_reward →  lfx.reward.effective (float, [-1, 1])
  summary.normalized_reward→  lfx.reward.normalized (float, [0, 1])
  summary.signals          →  lfx.reward.signals (JSON string)
                              + per-signal: lfx.reward.<name>.value, lfx.reward.<name>.confidence
                           →  ONE gen_ai.evaluation.result event (effective reward only)
  summary.timing.total_ms  →  span duration (start_time + total_ms = end_time)
  created_at               →  span start_time
  metadata.get("harness_version") →  lfx.harness.version (if present)

  assistant message[i]     →  LLM span (child of AGENT, openinference.span.kind=LLM)
    input messages          →  gen_ai.input.messages (JSON string, not dict)
    output content          →  gen_ai.output.messages (JSON string, not dict)
    token_count             →  gen_ai.usage.output_tokens (completion tokens only)
    summary.token_usage     →  gen_ai.usage.input_tokens (aggregate, on root AGENT span only)
    model                   →  gen_ai.request.model
    timestamp               →  span start_time (if available)
    step index (from        →  lfx.step.index, lfx.step.reward, lfx.step.done
     step_boundaries+steps)

    tool_call[j]           →  TOOL span (child of LLM, openinference.span.kind=TOOL)
      ToolCall.name         →  tool.name
      ToolCall.arguments    →  tool.parameters (pass through, already JSON string — no double-encoding)
      ToolCall.result       →  tool.output (JSON string)
        (matched via ToolCall.id ↔ Message.tool_call_id)
      ToolCall.success      →  otel status code (OK/ERROR)
      ToolCall.latency_ms   →  span duration
      ToolCall.error        →  exception event (if present)
      step index            →  lfx.step.index
```

**Step index resolution:** Each assistant message maps to a step via `step_boundaries`. The exporter finds the step `t` where `step_boundaries[t] <= message_index < step_boundaries[t+1]` (or end of messages for the last step). If multiple assistant messages fall within the same step, they all get `lfx.step.index = t`.

**Tool result linking:** Each `ToolCall` on an assistant message has an `id`. The corresponding tool-role `Message` has `tool_call_id` matching that `id`. The exporter joins these to populate `tool.output` on the TOOL span from the tool-role message's `content`, and sets `ToolCall.result`/`success`/`error` from the matched data.

**Attribute encoding:** OTel attributes only support primitives and arrays of primitives. All structured data (`gen_ai.input.messages`, `gen_ai.output.messages`, `tool.parameters`, `lfx.reward.signals`) are JSON-encoded strings, never raw dicts.

**Timestamps and durations (OTel API details):**

OTel Python expects `start_time` and `end_time` in **epoch nanoseconds** (int). The exporter converts as follows:

```python
import time

def _to_ns(unix_seconds: float) -> int:
    """Convert unix timestamp (seconds) to OTel nanoseconds."""
    return int(unix_seconds * 1_000_000_000)

def _ms_to_ns(ms: float) -> int:
    """Convert milliseconds to nanoseconds."""
    return int(ms * 1_000_000)
```

- **AGENT root span:** Uses `tracer.start_span()` (not context manager) with explicit `start_time=_to_ns(ep.created_at)`. Manually calls `span.end(end_time=_to_ns(ep.created_at) + _ms_to_ns(summary.timing.total_ms))`. If `created_at` is None, uses current time.
- **LLM spans:** `start_time=_to_ns(msg.timestamp)` if `msg.timestamp` is available. Duration from `steps[t].timing_ms` → `span.end(end_time=start_time + _ms_to_ns(timing_ms))`. If no timestamp, omit `start_time` (OTel defaults to "now").
- **TOOL spans:** Duration from `ToolCall.latency_ms` → `span.end(end_time=start_time + _ms_to_ns(latency_ms))`.
- **Missing timing:** If no timing data exists, omit explicit start/end and let OTel use wall clock (spans will show export time, which is imprecise but not fabricated). Log a debug warning.

### lfx Custom Attributes (lfx.* namespace)

| Attribute | Type | On Span | Description |
|---|---|---|---|
| `lfx.episode.id` | string | AGENT | Episode identifier |
| `lfx.state.id` | string | AGENT | Content-addressed config hash |
| `lfx.task.id` | string | AGENT | Task identifier |
| `lfx.session.id` | string | AGENT | Conversation grouping |
| `lfx.bench` | string | AGENT | Domain/benchmark name |
| `lfx.reward.effective` | float | AGENT | Priority-resolved reward [-1, 1] |
| `lfx.reward.normalized` | float | AGENT | Normalized reward [0, 1] |
| `lfx.reward.signals` | string | AGENT | JSON-serialized signals dict |
| `lfx.reward.<name>.value` | float | AGENT | Per-signal reward value [-1, 1] |
| `lfx.reward.<name>.confidence` | float | AGENT | Per-signal confidence [0, 1] |
| `lfx.harness.version` | string | AGENT | From `Episode.metadata["harness_version"]` if present |
| `lfx.step.index` | int | LLM/TOOL | Step index within episode |
| `lfx.step.reward` | float | LLM | Per-step reward |
| `lfx.step.done` | bool | LLM | Terminal step flag |
| `lfx.filtered` | bool | AGENT | Whether episode was format-gated |

### Implementation

```python
# lfx/exporters/otel.py

class OTelExporter(TraceExporter):
    """Export Episodes as OpenInference-compatible OTel spans via OTLP."""

    def __init__(
        self,
        *,
        tracer: "Tracer | None" = None,
        endpoint: str = "http://localhost:4318",  # OTLP HTTP default
        service_name: str = "lfx",
        resource_attributes: dict[str, str] | None = None,
    ) -> None:
        # Lazy import — otel is optional
        from opentelemetry import trace

        if tracer is not None:
            # Use caller-provided tracer (respects their TracerProvider)
            self._tracer = tracer
        else:
            # Check for existing global provider first
            global_provider = trace.get_tracer_provider()
            if hasattr(global_provider, "get_tracer") and not isinstance(
                global_provider, trace.ProxyTracerProvider
            ):
                # App already configured OTel — use their provider
                self._tracer = global_provider.get_tracer("lfx", version="0.1.0")
            else:
                # No existing provider — create one with OTLP export
                from opentelemetry.sdk.trace import TracerProvider
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.resources import Resource

                resource = Resource.create({
                    "service.name": service_name,
                    **(resource_attributes or {}),
                })
                provider = TracerProvider(resource=resource)
                provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
                )
                self._tracer = provider.get_tracer("lfx", version="0.1.0")

    def export(self, episodes: list[Episode]) -> None:
        for ep in episodes:
            self._export_episode(ep)

    def export_one(self, episode: Episode) -> None:
        self._export_episode(episode)

    def _export_episode(self, ep: Episode) -> None:
        import json
        import logging
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode

        log = logging.getLogger(__name__)

        # OpenInference constants with string-literal fallback
        try:
            from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
            SPAN_KIND_ATTR = SpanAttributes.OPENINFERENCE_SPAN_KIND
            AGENT_KIND = OpenInferenceSpanKindValues.AGENT.value
            LLM_KIND = OpenInferenceSpanKindValues.LLM.value
            TOOL_KIND = OpenInferenceSpanKindValues.TOOL.value
        except ImportError:
            log.warning("openinference-semantic-conventions not installed, using string literals")
            SPAN_KIND_ATTR = "openinference.span.kind"
            AGENT_KIND, LLM_KIND, TOOL_KIND = "AGENT", "LLM", "TOOL"

        start_ns = _to_ns(ep.created_at) if ep.created_at else None
        end_ns = None
        if start_ns and ep.summary.timing and ep.summary.timing.total_ms:
            end_ns = start_ns + _ms_to_ns(ep.summary.timing.total_ms)

        # Root AGENT span — explicit start/end, no context manager auto-end
        root = self._tracer.start_span(
            f"episode {ep.id[:8]}",
            attributes={
                SPAN_KIND_ATTR: AGENT_KIND,
                "lfx.episode.id": ep.id,
                "lfx.state.id": ep.state_id,
                "lfx.task.id": ep.task_id,
                "lfx.session.id": ep.session_id,
                "lfx.bench": ep.bench,
                "lfx.reward.effective": ep.summary.effective_reward(),
                "lfx.reward.normalized": ep.summary.normalized_reward(),
                "lfx.reward.signals": json.dumps(
                    {k: {"value": s.value, "confidence": s.confidence}
                     for k, s in ep.summary.signals.items()}
                ),
                "lfx.filtered": ep.summary.filtered,
            },
            **({"start_time": start_ns} if start_ns else {}),
        )

        # Aggregate input token usage on root span only
        # (output tokens live on individual LLM spans via Message.token_count)
        if ep.summary.token_usage:
            root.set_attribute("gen_ai.usage.input_tokens", ep.summary.token_usage.prompt)

        # harness_version from metadata if present
        if hv := ep.metadata.get("harness_version"):
            root.set_attribute("lfx.harness.version", hv)

        # Per-signal attributes
        for name, sig in ep.summary.signals.items():
            root.set_attribute(f"lfx.reward.{name}.value", sig.value)
            root.set_attribute(f"lfx.reward.{name}.confidence", sig.confidence)

        # One gen_ai.evaluation.result event for effective reward
        root.add_event("gen_ai.evaluation.result", attributes={
            "gen_ai.evaluation.score": ep.summary.effective_reward(),
        })

        # Build tool result lookup: tool_call_id → Message.content
        tool_results = {
            m.tool_call_id: m.content
            for m in ep.messages if m.role == "tool" and m.tool_call_id
        }

        # Reconstruct input messages per assistant turn
        # Full conversation history up to (but not including) the assistant
        # message at index i.  Uses the complete history because each LLM
        # call in an agentic loop sees the entire prior trajectory (system
        # prompt, reasoning, tool calls, tool results, prior assistant turns).
        def _input_messages_for(msg_index: int) -> list[dict]:
            return [
                {"role": m.role, "content": m.content}
                for m in ep.messages[:msg_index]
            ]

        # Create LLM spans per assistant message, TOOL spans as children of LLM
        with trace.use_span(root, end_on_exit=False):
            for i, msg in enumerate(ep.messages):
                if msg.role != "assistant":
                    continue

                step_idx = self._resolve_step(i, ep.step_boundaries)
                step_meta = ep.steps[step_idx] if step_idx < len(ep.steps) else None
                llm_start = _to_ns(msg.timestamp) if msg.timestamp else None

                llm_span = self._tracer.start_span(
                    f"chat {msg.model or ep.model or 'unknown'}",
                    attributes={
                        SPAN_KIND_ATTR: LLM_KIND,
                        "gen_ai.request.model": msg.model or ep.model or "",
                        "gen_ai.input.messages": json.dumps(_input_messages_for(i)),
                        "gen_ai.output.messages": json.dumps([{"role": "assistant", "content": msg.content}]),
                        "lfx.step.index": step_idx,
                    },
                    **({"start_time": llm_start} if llm_start else {}),
                )
                if msg.token_count:
                    llm_span.set_attribute("gen_ai.usage.output_tokens", msg.token_count)
                if step_meta:
                    llm_span.set_attribute("lfx.step.reward", step_meta.reward)
                    llm_span.set_attribute("lfx.step.done", step_meta.done)

                # TOOL spans as children of this LLM span
                if msg.tool_calls:
                    with trace.use_span(llm_span, end_on_exit=False):
                        for tc in msg.tool_calls:
                            tool_span = self._tracer.start_span(
                                f"tool {tc.name}",
                                attributes={
                                    SPAN_KIND_ATTR: TOOL_KIND,
                                    "tool.name": tc.name,
                                    "tool.parameters": tc.arguments,  # already JSON string
                                    "lfx.step.index": step_idx,
                                },
                                # Only set start_time when we have it
                                **({"start_time": llm_start} if llm_start else {}),
                            )
                            # Link tool result via tool_call_id
                            if tc.id and tc.id in tool_results:
                                tool_span.set_attribute("tool.output", tool_results[tc.id])
                            elif tc.result:
                                tool_span.set_attribute("tool.output", tc.result)

                            if tc.success is False:
                                tool_span.set_status(Status(StatusCode.ERROR, tc.error or "tool call failed"))
                            if tc.error:
                                tool_span.add_event("exception", {"exception.message": tc.error})

                            # Only set end_time when start_time was also set
                            tool_end = None
                            if llm_start and tc.latency_ms:
                                tool_end = llm_start + _ms_to_ns(tc.latency_ms)
                            tool_span.end(**({"end_time": tool_end} if tool_end else {}))

                llm_end = None
                if llm_start and step_meta and step_meta.timing_ms:
                    llm_end = llm_start + _ms_to_ns(step_meta.timing_ms)
                llm_span.end(**({"end_time": llm_end} if llm_end else {}))

        root.end(**({"end_time": end_ns} if end_ns else {}))
```

### Dependencies

```toml
# pyproject.toml — optional extra
[project.optional-dependencies]
otel = [
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "opentelemetry-exporter-otlp-proto-http>=1.20",
    "openinference-semantic-conventions>=0.1.6",
]
```

### OpenInference Constants

Import span kind constants from `openinference-semantic-conventions` in both the exporter and Phase 2 wrapper. Imports are lazy (inside functions/methods) so the dependency remains optional:

```python
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues

# Usage:
span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.LLM.value)
```

If `openinference-semantic-conventions` is not installed (e.g., user installed `opentelemetry-*` but not the semconv package), fall back to string literals with a logged warning.

### Tests

- `tests/test_otel_exporter.py`
  - InMemorySpanExporter to capture spans, verify span tree: AGENT root → LLM children → TOOL grandchildren
  - Verify OpenInference span kind attributes on each span type
  - Verify lfx.* custom attributes (both reward ranges: effective and normalized)
  - Verify gen_ai.* attributes are JSON strings (not dicts)
  - Verify tool result linking (ToolCall.id ↔ Message.tool_call_id populates tool.output)
  - Verify ToolCall.arguments passed directly (no double JSON encoding)
  - Verify timestamps in nanoseconds derived from Episode.created_at and timing data
  - Verify explicit span.end() with correct end_time (not auto-ended by context manager)
  - Verify token_count maps to output_tokens only (not input_tokens); input_tokens from summary.token_usage
  - Verify lfx.harness.version sourced from Episode.metadata["harness_version"]
  - Verify step index resolution when multiple assistant messages share a step
  - Verify graceful ImportError when otel not installed
  - Verify tracer injection (caller-provided tracer used instead of creating new provider)
  - Verify existing global TracerProvider is respected

## Phase 2: Inline Span Emission (wrap-time tracing)

**Modified files:** `lfx/wrapper.py`, `lfx/callbacks/litellm_cb.py`

Optionally emit OTel spans at call time (not just post-hoc export). This gives real-time tracing in Langfuse/Datadog dashboards.

### Design

Tracer is passed explicitly to `wrap()` and `LfxCallback` — no global state. A future `lfx.configure(tracer=...)` can be added as additive convenience (sets a default used when no explicit tracer is provided).

```python
# lfx/wrapper.py

class WrappedClient:
    def __init__(self, client, collector, *, tracer=None):
        self._tracer = tracer  # Optional OTel tracer

    def complete(self, messages, **kwargs):
        if self._tracer:
            with self._tracer.start_as_current_span(
                f"chat {kwargs.get('model', 'unknown')}",
                attributes={
                    "openinference.span.kind": "LLM",
                    "gen_ai.operation.name": "chat",
                    "gen_ai.input.messages": json.dumps(messages),  # JSON string
                    ...
                }
            ) as span:
                response = self._client.complete(messages, **kwargs)
                span.set_attribute("gen_ai.output.messages", json.dumps([...]))  # JSON string
                span.set_attribute("gen_ai.usage.output_tokens", ...)
                # Also ingest into collector as before
        else:
            response = self._client.complete(messages, **kwargs)
            # Ingest into collector as before
```

### wrap() API change

```python
# Before:
wrapped = lfx.wrap(client, collector=collector)

# After (backward compatible):
wrapped = lfx.wrap(client, collector=collector)  # no tracing
wrapped = lfx.wrap(client, collector=collector, tracer=my_tracer)  # with tracing
```

### Same for LfxCallback

```python
# lfx/callbacks/litellm_cb.py
class LfxCallback:
    def __init__(self, collector, *, tracer=None):
        self._tracer = tracer
```

## Phase 3: OTel Ingestion (future, not in this PR)

Accept OTLP traces from external sources, convert to Episodes. Documented as lossy:

- No step_boundaries (OTel doesn't have this concept)
- No composable reward signals (OTel evaluation events are simpler)
- No state_id (OTel doesn't track agent config)

This is a v0.3+ feature. Not designed here.

## File Changes Summary

### New files
- `lfx/exporters/otel.py` — OTelExporter (Episode → OTLP spans)
- `tests/test_otel_exporter.py` — exporter tests

### Modified files
- `pyproject.toml` — add `[otel]` optional dependency group
- `lfx/wrapper.py` — optional `tracer` param in WrappedClient + wrap()
- `lfx/callbacks/litellm_cb.py` — optional `tracer` param in LfxCallback

### NOT modified
- `lfx/core/episode.py` — Episode format unchanged
- `lfx/collector.py` — collection unchanged
- `lfx/learner.py` — learning loop unchanged
- All existing exporters — untouched

## Migration Risks

1. **OTel SDK weight** — `opentelemetry-sdk` adds ~5MB of dependencies. Mitigated: optional extra, not required.

2. **OpenInference attribute stability** — OpenInference attributes may change. Mitigated: we depend on `openinference-semantic-conventions` and import constants; updates are a version bump.

3. **Span volume** — A 10-step episode produces ~20 spans (1 agent + 10 LLM + ~10 tool). At scale this could be noisy. Mitigated: exporter is opt-in, users configure sampling at the OTel Collector level.

4. **Phase 2 latency** — Inline span emission adds overhead to every LLM call. Mitigated: spans are batched by BatchSpanProcessor (async, non-blocking). Measured overhead expected <1ms.

5. **Dual-write consistency** — Phase 2 emits OTel spans AND creates Episodes. If one fails, the other still succeeds. This is acceptable — observability is best-effort, learning is primary.

6. **TracerProvider conflicts** — Apps may already configure OTel globally. Mitigated: OTelExporter accepts an injected `tracer` or falls back to the existing global provider before creating its own.

## Verification

```bash
# Phase 1: Exporter works
pytest tests/test_otel_exporter.py -v

# Phase 2: Inline tracing works
pytest tests/test_wrapper.py tests/test_litellm_callback.py -v

# All existing tests still pass
pytest tests/ -v  # 316+ tests
```

## Resolved Questions

### From Codex Review Round 1
1. **Location:** `lfx/exporters/otel.py` — consistent with existing exporters. Spin out `lfx-otel` later if needed.
2. **Span mapping:** Three-level: AGENT root → LLM children → TOOL grandchildren. No step spans. Step metadata as attributes.
3. **Evaluation events:** One `gen_ai.evaluation.result` for effective reward. Raw signals in `lfx.reward.signals` JSON + per-signal `lfx.reward.<name>.*` attributes.
4. **Tracer injection:** Explicit `tracer=` param on `wrap()` and `LfxCallback`. Global `lfx.configure()` deferred as additive.
5. **OpenInference constants:** Depend on `openinference-semantic-conventions` in the `otel` extra. Lazy imports with string-literal fallback.

### From Codex Review Round 2
6. **OTel timestamp API:** All times in epoch nanoseconds. Use `tracer.start_span()` + explicit `span.end(end_time=...)` instead of context managers that auto-end at wall clock.
7. **Span hierarchy:** AGENT → LLM → TOOL (three levels). TOOL spans are children of the LLM span that initiated them, not siblings.
8. **Token usage:** `Message.token_count` = output tokens only (set on assistant messages in wrapper). `Episode.summary.token_usage` provides aggregate input/output. Map accordingly.
9. **lfx.harness.version:** Sourced from `Episode.metadata["harness_version"]` if present. Not a field on Episode itself.
10. **ToolCall.arguments:** Already a JSON string in Episode schema. Pass directly to `tool.parameters` attribute — no `json.dumps()` to avoid double-encoding.

### From Codex Review Round 3
11. **TOOL span start_time:** Pass `start_time=` to TOOL `start_span()` when available, so `end_time` is consistent. Never set `end_time` without `start_time`.
12. **trace import in _export_episode:** `from opentelemetry import trace` inside `_export_episode()` (lazy import, avoids NameError).
13. **set_status signature:** Use `Status(StatusCode.ERROR, description)` object, not positional args.
14. **gen_ai.input.messages:** Reconstruct per-turn input from messages[step_start:msg_index]. JSON string.
15. **OpenInference constants:** Import from `openinference.semconv.trace` with string-literal fallback + log warning. Used consistently in exporter code.

### From Codex Review Round 4
16. **Input message reconstruction includes all roles:** `_input_messages_for()` includes all message roles (system, user, assistant, tool) so multi-turn steps preserve earlier assistant outputs as context.
17. **Logger defined explicitly:** `log = logging.getLogger(__name__)` defined at top of `_export_episode()` before use in fallback warning.

### From Codex Review Round 5
18. **Token usage split:** Root AGENT span carries only `gen_ai.usage.input_tokens` (aggregate from `summary.token_usage.prompt`). Output tokens live exclusively on individual LLM spans via `Message.token_count`. No duplicate output token reporting.
