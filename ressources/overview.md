# lfx — Technical Design Document

**Continual improvement layer for LLM-powered agents.**

Status: v0.1 — Harness layer complete, Router and Weights in progress.
Feedback requested on: architecture, trajectory format, integration strategy, open questions.

---

## Problem

LLM agents generate rich execution traces — conversations, tool calls, successes, failures. These traces contain signal about what works and what doesn't. Today they're discarded. Manual prompt engineering doesn't scale: it's slow, subjective, and can't incorporate feedback from thousands of production interactions.

lfx closes this loop automatically: observe → extract reward signals → reflect → update → validate → deploy.

---

## Core Abstraction

```
Agent = Model + Harness
Harness = System Prompt + Tools + Memory (Playbook)
```

The **Model** generates responses. The **Harness** is everything around it — the system prompt, tool definitions, and a structured memory (playbook) of learned strategies. lfx improves the harness automatically from execution traces, and optionally improves model selection (Router) and model weights (Weights).

---

## Architecture

### Learning Layers

Three independent layers, each with its own timescale and cost profile:

| Layer | Controls | Mechanism | Latency | Cost |
|-------|----------|-----------|---------|------|
| **Harness** | System prompt, playbook entries, tool configs | GEPA prompt evolution + ACE reflector | Seconds | Zero (prompt-only) |
| **Router** | Model selection per query | Bayesian bandit on query features | Minutes | Minimal (routing logic) |
| **Weights** | Model parameters | LoRA/GRPO via SkyRL | Hours | GPU |

All layers implement the **Layer Protocol**:

```python
class Layer(Protocol):
    def forward_backward(self, data: Datum) -> Future[FBResult]  # accumulate
    def optim_step(self) -> Future[OptimResult]                  # apply (atomic)
    def sample(self, ctx: SampleContext) -> Future[SampleResult]  # produce output
    def save_state(self, name: str) -> Future[SaveResult]
    def load_state(self, state_dict: dict) -> Future[LoadResult]
    def to_dict(self) -> dict[str, Any]                          # for hashing/logging
```

**Transactional semantics**: All layers run `forward_backward` first. If any fails, all abort. Only when all succeed do they `optim_step`. If any `optim_step` fails, all roll back via `deepcopy` snapshots. No mixed states.

### Data Flow

```
┌─────────────────────────────────────────────────┐
│                    lfx.wrap()                    │
│   (intercepts LLM calls, captures messages,     │
│    tool calls, token usage, model ID, timing)   │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │ EpisodeCollector │  thread-safe (lock + OrderedDict LRU)
              │  + RewardPipeline│  formatting gate filters low-quality
              └────────┬────────┘
                       │ on_batch(episodes)
              ┌────────▼────────┐
              │  AsyncLearner   │  background thread, queue-based
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          │     forward_backward    │  all layers, accumulate only
          └────────────┼────────────┘
                       │ all ok
          ┌────────────┼────────────┐
          │       optim_step        │  all layers, snapshot-rollback
          └────────────┼────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ Harness  │   │  Router  │   │ Weights  │
   │ prompt   │   │ model    │   │ LoRA     │
   │ tools    │   │ selection│   │ GRPO     │
   │ playbook │   └──────────┘   └──────────┘
   └─────────┘

        │              │              │
        ▼              ▼              ▼
   ┌──────────────────────────────────────┐
   │            Exporters                 │
   │  LangfuseExporter  │  SkyRLExporter │
   │  (observe/debug)   │  (GRPO train)  │
   └──────────────────────────────────────┘
```

### Key Properties

- **Non-blocking**: Learning runs in a background thread. Agent call latency is unaffected.
- **Atomic rollback**: `optim_step()` snapshots via `deepcopy` before mutating. Failure restores the snapshot.
- **Content-addressed state**: `StateID` = SHA-256 of `harness.to_dict() + router.to_dict() + weights.to_dict()`. Deterministic — any configuration is reproducible from its hash.
- **Regression safety**: `gate_for_deploy()` compares reward distributions before promoting a new state to production.

---

## Trajectory Format (Episode)

The Episode is the central data structure. One episode = one complete agent interaction.

```
Episode
├── id                  str         uuid4 hex
├── state_id            str         SHA-256 hash of layer config at time of execution
├── task_id             str         stable identifier (env-provided or content hash)
├── session_id          str         groups episodes from same conversation (optional)
├── bench               str         domain/benchmark name
├── model               str|None    primary model used
├── created_at          float|None  unix timestamp
├── metadata            dict        extensible key-value bag
│
├── messages[]          list[Message]   full conversation, OpenAI chat format
│   ├── role            "system" | "user" | "assistant" | "tool"
│   ├── content         str
│   ├── tool_calls[]    list[ToolCall] | None
│   │   ├── id          str         tool call ID
│   │   ├── name        str         tool name
│   │   ├── arguments   str         JSON string
│   │   ├── result      str|None    tool output
│   │   ├── success     bool|None
│   │   ├── latency_ms  float|None
│   │   └── error       str|None
│   ├── tool_call_id    str|None    (for role="tool" responses)
│   ├── name            str|None    tool name (for role="tool")
│   ├── timestamp       float|None
│   ├── token_count     int|None
│   └── model           str|None    which model generated this message
│
├── step_boundaries[]   list[int]   indices into messages where each agent turn starts
├── steps[]             list[StepMeta]
│   ├── t               int         step index
│   ├── reward          float       0.0 for intermediate, actual for terminal
│   ├── done            bool
│   ├── timing_ms       float
│   └── info            dict
│
└── summary             EpisodeSummary
    ├── signals{}       dict[str, RewardSignal]
    │   └── RewardSignal(name, value[-1,1], confidence[0,1])
    ├── filtered        bool    whether episode was gated by formatting filter
    ├── score_breakdown dict    per-dimension scores (optional)
    ├── token_usage     TokenUsage(prompt, completion, total)
    └── timing          Timing(total_ms, per_step_ms[])
```

**Design decisions:**

- **Messages stored once.** `step_boundaries` indexes into the flat message list to delineate multi-turn agent steps. No content duplication. Trade-off: step access requires slicing by index rather than direct iteration.
- **OpenAI chat format.** Messages serialize to `{"role": ..., "content": ..., "tool_calls": [...]}` via `to_openai_messages()`. Compatible with any OpenAI-compatible API.
- **Rich tool metadata.** Each `ToolCall` captures arguments, result, success flag, latency, and error. This powers the `ExecutionExtractor` which derives reward signals from tool outcomes without needing ground truth.
- **Composable reward signals.** `summary.signals` holds multiple named signals with values in [-1, 1] and confidence in [0, 1]. `effective_reward()` resolves via priority: user > outcome > execution (high conf) > judge. No ambiguity about which signal drives learning.
- **session_id vs task_id.** `task_id` identifies the task (stable across retries). `session_id` groups episodes from the same conversation. In live mode, `task_id` is a uuid per request; in eval mode, it's a deterministic hash of bench + question + context.

---

## Reward System

### Signal Extraction

```python
RewardPipeline.with_defaults()  # → [ExecutionExtractor, UserFeedbackExtractor]
```

| Extractor | Signal name | Source | Default |
|-----------|------------|--------|---------|
| `ExecutionExtractor` | `execution` | Tool call success/failure patterns, error keywords, HTTP status codes | Yes |
| `UserFeedbackExtractor` | `user` | `collector.submit_feedback(episode_id, score)` | Yes |
| `OutcomeExtractor` | `outcome` | Task environment ground truth (e.g., eval score) | No (needs env) |
| Judge | `judge` | LLM-as-judge evaluating the trajectory | No (costs tokens) |

### Priority Resolution

```
effective_reward() priority:
  1. user       (always wins)
  2. outcome    (ground truth)
  3. execution  (only if confidence >= 0.7)
  4. judge      (fallback)
  → 0.0 if no qualifying signal
```

Judge is **lazy**: `needs_judge()` returns False if any higher-priority signal exists. The pipeline auto-skips judge extractors when unnecessary.

### Canonical Range

All signals use [-1, 1] internally. `normalized_reward()` maps to [0, 1] for backward compatibility. The `total_reward` property provides a legacy bridge: setting it stores an `outcome` signal; reading it returns `normalized_reward()`.

---

## Harness Layer (Detail)

The Harness combines three mechanisms:

### 1. GEPA Prompt Evolution
Maintains a **Pareto front** of non-dominated prompt candidates per benchmark. An LLM reads execution traces as "Actionable Side Information" (ASI), diagnoses failures, and proposes targeted prompt mutations. Per-task scores prevent premature convergence — a candidate survives if it's best on *any* task, even if its average is suboptimal.

### 2. ACE Playbook Memory
A structured, itemized playbook where each entry carries `helpful`/`harmful` counters. Updates are always **incremental deltas** (add/update/remove) via a Reflector→Curator pipeline. Each entry has tags for categorization and scores for pruning.

The Reflector analyzes episode traces (JSON-serialized, sanitized against null bytes and injection) and produces `Insight` objects. The Curator (`apply_insights()`) validates insights (action validity, tag whitelist, injection pattern detection, length limits, type checks) before applying as playbook deltas.

### 3. Tool Config Tuning
Tool schemas and descriptions can evolve using the same GEPA loop — optimizing descriptions so the agent selects the right tools more reliably.

### Stagnation Handling
`AdaptiveIntensity` tracks reward history. When stagnating, `ParadigmBreakthrough` generates creative strategy shifts that bypass local optima. All breakthrough insights pass through the same validation pipeline.

---

## Integration Strategy

### Principle: lfx captures, platforms display

lfx owns trace capture because it needs structured multi-turn trajectories (Episodes) with step boundaries, tool metadata, and composable reward signals. Observability platforms see flat spans — not the same thing.

### Ingestion (traces in)

| Method | Code change | Data richness | Best for |
|--------|------------|---------------|----------|
| `lfx.wrap(client)` | 1 line | Full (steps, tools, timing, model) | Any Python agent |
| litellm callback | Config | Good (request/response/tokens) | Teams on litellm already |
| n8n webhook/node | Node config | Moderate (execution results) | Workflow automation |
| Langfuse adapter | Adapter | **Lossy** (flat spans → episodes, no step boundaries) | Teams who won't add wrap() |

`lfx.wrap()` is the primary path. The litellm callback is a high-leverage secondary — litellm is widely adopted and its callback system lets lfx register without code changes to the agent. Langfuse ingestion is a fallback for adoption friction, but the data is structurally lossy.

### Export (data out)

| Exporter | Destination | Purpose |
|----------|------------|---------|
| `LangfuseExporter` | Langfuse | Visualization, debugging, dashboards |
| `SkyRLExporter` | SkyRL/OpenClaw | GRPO training data for Weights layer |
| Custom | Any | Implement exporter protocol |

### Why not langfuse as intermediary?

Langfuse captures individual LLM calls as spans/generations. lfx needs multi-turn trajectories with step boundaries, tool call results (success/failure/latency), and composable reward signals. Converting langfuse spans back into Episodes loses structure and requires heuristic reconstruction of turn boundaries. Direct capture avoids this entirely.

---

## Security Considerations

### Prompt Injection Surface

The Reflector reads episode traces (which may contain adversarial user input) and produces insights that become playbook entries (which become part of future system prompts). This is an indirect prompt injection vector.

**Mitigations implemented:**
- Reflector uses **JSON-structured traces** (not text interpolation) — harder to inject across field boundaries
- **Null byte stripping** on all string fields before serialization (`_sanitize_str`)
- `_validate_insights()` runs on **all insight sources** (Reflector and ParadigmBreakthrough):
  - Action whitelist (`add`/`update`/`remove` only)
  - `target_entry_id` required for update/remove
  - Tag character whitelist (alphanumeric + hyphens + underscores)
  - Type checks on content, tags, source_episode_ids
  - Injection pattern regex on content **and** tags
  - Content length cap (2000 chars)
- Agent input sanitization: null byte stripping + 8k char length clamp
- `isinstance(item, dict)` guard in Reflector response parsing

**Not yet implemented:**
- Content policy filter (semantic analysis of insight content)
- Human-in-the-loop approval for playbook mutations
- Rate limiting on insight volume per batch

### State Integrity

`StateID` uses `_safe_default()` which logs a warning when encountering non-serializable objects instead of silently falling back to `str()`. Validators are serialized as names (not raw objects) in `to_dict()` for deterministic hashing.

---

## File Structure

```
lfx/
├── core/
│   ├── episode.py      Episode, Message, ToolCall, EpisodeSummary, StepMeta
│   ├── reward.py       RewardSignal, RewardExtractor protocol, RewardPipeline
│   ├── state.py        StateID (content-addressed SHA-256)
│   ├── reflector.py    LLM-based trace analysis → Insight objects
│   ├── env.py          TaskEnvironment, Sample, EvalResult protocols
│   ├── types.py        Datum, Future, Layer protocol, result types
│   ├── loop.py         AgentState, learning_loop orchestrator
│   ├── layer.py        Layer protocol definition
│   ├── gate.py         gate_for_deploy() regression safety
│   ├── intensity.py    AdaptiveIntensity (when to reflect)
│   └── paradigm.py     ParadigmBreakthrough (stagnation escape)
├── layers/
│   ├── harness.py      Harness, Playbook, PlaybookEntry, ParetoFront, Insight
│   ├── router.py       Router (Bayesian bandit model selection)
│   └── weights.py      Weights (LoRA/GRPO via SkyRL)
├── extractors/
│   ├── execution.py    ExecutionExtractor (tool call heuristics)
│   ├── outcome.py      OutcomeExtractor (env ground truth)
│   ├── user_feedback.py UserFeedbackExtractor (passthrough)
│   └── formatting.py   FormattingFilter (quality gate)
├── exporters/
│   └── skyrl.py        SkyRLExporter (GRPO training format)
├── collector.py        EpisodeCollector (live mode, thread-safe)
├── learner.py          AsyncLearner (background worker)
├── wrapper.py          lfx.wrap() SDK wrapper
├── agent.py            LfXAgent high-level API
└── llm.py              LLMClient protocol, LiteLLMClient, MockLLMClient
```

---

## Open Questions

1. **Langfuse exporter granularity** — Export entire episodes as single traces, or individual steps as nested spans? Episodes are more natural for lfx, but nested spans integrate better with langfuse's UI.

2. **litellm callback limitations** — litellm callbacks see individual completions, not multi-turn trajectories. How do we reconstruct step boundaries? Options: (a) session-based grouping with timeout, (b) require callers to pass a session_id, (c) treat each call as a single-step episode.

3. **Playbook convergence** — The playbook grows monotonically (pruning removes low-score entries, but new entries accumulate). Should there be a hard cap? A compression step that merges similar entries?

4. **Cross-agent playbook transfer** — When multiple agents share a domain, should playbook entries transfer between them? What's the right abstraction for "this insight is general" vs. "this insight is agent-specific"?

5. **Weights layer integration boundary** — Where exactly does lfx stop and SkyRL/OpenClaw begin? Currently `SkyRLExporter` converts episodes to training format. Should lfx also manage the training loop, or just produce training data?

6. **Human-in-the-loop** — For high-stakes deployments, playbook mutations should be reviewable. What's the right UX? Git-style diffs? Dashboard approval queue?

---

## How to Run Tests

```bash
pytest tests/ -v                                    # all 316 tests
pytest tests/ -k "sanitiz or inject or default" -v  # security + new behavior
```
