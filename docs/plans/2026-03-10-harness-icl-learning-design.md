# Harness In-Context Learning Design

**Date:** 2026-03-10
**Status:** Approved
**Next PR:** Fine-tuning / Weights layer (separate design)

## Goal

Make the Harness layer actually learn from experience. The system prompt
improves automatically over iterations via a Reflector-Curator pipeline
(ACE-style playbook deltas) with SkyDiscover-inspired adaptive intensity
and paradigm breakthrough. Demonstrated on MATH/AIME and tau2-bench.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User's World                          │
│                                                          │
│  Option A: LfXAgent(task_model, reflector_model)        │
│            runs tasks, records episodes automatically    │
│                                                          │
│  Option B: User's own agent -> episode traces -> ingest  │
└─────────────────┬───────────────────────────────────────┘
                  │ Episodes
                  ▼
┌─────────────────────────────────────────────────────────┐
│              LfX Learning Engine                         │
│                                                          │
│  ┌───────────┐   ┌────────────┐   ┌──────────┐         │
│  │ Reflector │──>│  Curator   │──>│ Playbook │         │
│  │(configurable) │(deterministic)  │  (JSON)  │         │
│  └───────────┘   └────────────┘   └──────────┘         │
│       │                                │                 │
│       v                                v                 │
│  ┌────────────┐              ┌────────────────┐         │
│  │ Paradigm   │              │ System Prompt  │         │
│  │ Breakthru  │              │   Injection    │         │
│  │(on stall)  │              │(next episode)  │         │
│  └────────────┘              └────────────────┘         │
│                                                          │
│  Billing boundary:                                       │
│    Task LLM = customer's API key + model                 │
│    Reflector/Paradigm = LfX-managed (or configurable)    │
└─────────────────────────────────────────────────────────┘
```

## Key Design Decisions

### 1. ACE core + SkyDiscover spices

- **Playbook with delta edits (ACE):** Core knowledge store. Incremental
  growth avoids context collapse and brevity bias.
- **Generator-Reflector-Curator pipeline (ACE):** Episode traces ->
  Insight objects -> playbook deltas.
- **Sibling context (SkyDiscover):** When reflecting, show what previous
  mutations of the same playbook entry produced (helped/hurt/unchanged).
- **Paradigm Breakthrough (SkyDiscover):** When improvement stagnates,
  ask a strong LLM for entirely new strategic directions.
- **Adaptive reflection intensity (SkyDiscover):** Reflect more when
  stagnating, less when improving. Saves LLM calls.
- **GEPA Pareto front: DEFERRED** to a later PR.

### 2. Split-brain billing

Customer's API key for task execution. LfX-managed key (or configurable)
for reflection and paradigm calls. The learning is the value-add.

### 3. LLM client abstraction

Uses LiteLLM under the hood for 100+ provider support. Two client
instances: task_client and reflector_client. Each configurable with
different model, API key, temperature, etc.

### 4. MATH/AIME for math demo, tau2 for agent demo

MATH/AIME: deterministic correctness check, low baseline on cheap
models (Haiku ~30%), room for improvement.

tau2-bench: multiplicative reward from dimension breakdown, local
Python API (no A2A server needed).

## Components

### LLMClient (lfx/llm.py)

```python
class LLMClient(Protocol):
    def complete(self, messages: list[dict], **kwargs) -> str: ...

class LiteLLMClient:
    def __init__(self, model: str, api_key: str | None = None, **kwargs): ...
    def complete(self, messages, **kwargs) -> str: ...
```

### TaskEnvironment (lfx/core/env.py)

```python
class TaskEnvironment(Protocol):
    def get_tasks(self) -> list[Sample]: ...
    def evaluate(self, sample: Sample, response: str) -> EvalResult: ...

@dataclass
class Sample:
    question: str
    context: str = ""
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class EvalResult:
    score: float              # 0.0 - 1.0
    feedback: str = ""        # Actionable Side Information
    metrics: dict[str, float] = field(default_factory=dict)
```

### Reflector (lfx/core/reflector.py)

LLM call that reads episode traces and produces Insight objects.

Input: current playbook + batch of episodes with traces + sibling
context (previous mutations of same entries).

Output: list[Insight] — add/update/remove operations on playbook.

LLM: configurable, default claude-sonnet-4-6, temperature 0.7.

### ParadigmBreakthrough (lfx/core/paradigm.py)

Triggered when reward improvement < threshold for K iterations.

Input: current playbook + reward history + previously tried paradigms.

Output: 1-3 new strategic playbook entries tagged [paradigm].

LLM: strong model (Sonnet/Opus), one call only on stagnation.

### AdaptiveIntensity (lfx/core/intensity.py)

Tracks improvement signal G per benchmark.
- G high (improving) -> reflect every Nth batch, smaller batches
- G low (stagnating) -> reflect every batch, trigger paradigm

### MathEnvironment (lfx/envs/math.py)

MATH/AIME subset (~50-100 problems). Deterministic scoring:
score = 1.0 if parsed_answer == ground_truth else 0.0.
Feedback includes expected vs actual answer.

### Tau2Environment (lfx/envs/tau2.py)

Wraps tau2-bench LocalAgent. Maps SimulationRun -> Episode.
score = product(dimension_scores). Feedback = reward_breakdown.

### Harness modifications (lfx/layers/harness.py)

- forward_backward: calls Reflector LLM, accumulates Insights
- optim_step: applies via Curator (mostly done already)
- New: store sibling context for SkyDiscover-style reflection

### Loop modifications (lfx/core/loop.py)

- Adaptive reflection intensity check
- Paradigm breakthrough trigger
- Episode construction from Sample + EvalResult

### LfXAgent convenience wrapper (lfx/agent.py)

```python
agent = LfXAgent(
    task_model="haiku",
    reflector_model="sonnet",
)
results = agent.learn(samples, env, iterations=5)
# or
agent.ingest(episodes)  # from external traces
prompt = agent.get_system_prompt()
```

### Demo script (examples/demo_math.py)

Runnable end-to-end demo: `python examples/demo_math.py`

## Cost Model

Per iteration (batch_size=5, MATH/AIME):

| Call              | Model  | Tokens (approx)       | Cost    |
|-------------------|--------|-----------------------|---------|
| 5x task execution | Haiku  | ~2k in + ~500 out ea  | ~$0.005 |
| 1x reflection     | Sonnet | ~8k in + ~1k out      | ~$0.03  |
| Paradigm (rare)   | Sonnet | ~4k in + ~1k out      | ~$0.02  |

~$0.035 per iteration. 10-iteration demo costs ~$0.35.

## Out of Scope (this PR)

- Router layer learning (accumulates signals but no real routing)
- Weights / GRPO fine-tuning (next PR)
- GEPA Pareto front prompt evolution (future PR)
- Langfuse / Phoenix export (future PR)
- LiteLLM proxy mode (future PR)
- LfXCompletion drop-in wrapper (future PR)

## References

- ACE: Agentic Context Engineering (arXiv 2510.04618, ICLR 2026)
- GEPA: Reflective Prompt Evolution (arXiv 2507.19457, ICLR 2026)
- SkyDiscover: github.com/skydiscover-ai/skydiscover
- Kayba ACE: github.com/kayba-ai/agentic-context-engine
- DataWizz: datawizz.ai (integration pattern reference)
