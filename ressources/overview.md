# lfx — Learning from Experience

**A universal continual improvement layer for AI agents.**

lfx makes any LLM-powered agent get better over time. One line of code to integrate. Three independent learning mechanisms. No vendor lock-in.

---

## What It Does

Every AI agent generates traces — conversations, tool calls, successes, failures. Today those traces are thrown away. lfx captures them and uses them to improve the agent automatically.

```
Agent ←→ LLM
         ↑
        lfx        ← captures traces, extracts reward signals,
                      improves prompts/routing/weights
```

The agent doesn't need to change how it works. lfx wraps the LLM client, observes what happens, and feeds improvements back in.

---

## Three Learning Layers

lfx has three independent improvement mechanisms, each operating at a different timescale:

| Layer | What it does | Speed | Cost |
|-------|-------------|-------|------|
| **Harness** | Evolves system prompts and playbook rules from experience | Seconds | Free (prompt changes only) |
| **Router** | Learns which model to route each query to | Minutes | Minimal |
| **Weights** | Fine-tunes model weights via LoRA/GRPO | Hours | GPU cost |

Each layer follows the same **Layer Protocol**:
1. `forward_backward(datum)` — analyze a batch of episodes
2. `optim_step()` — apply the update (with atomic snapshot-rollback on failure)

You can use any combination. Most users start with Harness only (zero cost, immediate results) and add layers as needed.

---

## Composable Reward Signals

lfx doesn't rely on a single reward number. It extracts multiple signals and combines them with a priority system:

```
user feedback  >  outcome (env score)  >  execution heuristics  >  LLM judge
```

- **User feedback**: Explicit thumbs up/down from end users (always wins when present)
- **Outcome**: Ground-truth score from a task environment (benchmarks, evals)
- **Execution**: Heuristic analysis of tool call success/failure patterns
- **Judge**: LLM-as-judge fallback, only invoked when no better signal exists

This means:
- In **benchmarks**: outcome signals drive learning automatically
- In **production**: execution heuristics work immediately, user feedback refines over time
- **Judge** is lazy — never called when unnecessary, saving cost

---

## Usage

### Level 0: Wrap any LLM client (live mode)

```python
import lfx

# Your existing LLM client (litellm, openai, custom, anything)
client = your_llm_client()

# Add lfx in one line
collector = lfx.EpisodeCollector(
    pipeline=lfx.RewardPipeline([]),  # execution heuristics auto-included
    batch_size=16,
    on_batch=learner.on_batch,
)
wrapped = lfx.wrap(client, collector=collector)

# Use exactly as before — lfx observes and learns in the background
response = wrapped.complete(messages)
```

User feedback flows in via a simple API:
```python
collector.submit_feedback(episode_id, score=1.0)   # thumbs up
collector.submit_feedback(episode_id, score=-1.0)  # thumbs down
```

### Level 1: Plug-and-play learning loop

```python
agent = lfx.LfXAgent(
    task_client=my_llm,
    reflector_client=my_llm,
    base_system_prompt="You are a helpful assistant.",
)

results = agent.learn(env=my_task_env, iterations=20)
# System prompt now includes learned playbook rules
print(agent.get_system_prompt())
```

### Level 2: Bring your own traces

```python
# Collect episodes however you want, then feed them in
agent.ingest(my_episodes)
```

### Level 3: Direct loop access

```python
from lfx.core.loop import learning_loop
# Full control over every parameter
```

---

## Integration Points

lfx is designed to work with anything:

- **litellm** — wrap any model provider (OpenAI, Anthropic, Gemini, local)
- **n8n** — webhook integration for workflow automation agents
- **OpenClaw / SkyRL** — export episodes to GRPO trainers via `SkyRLExporter`
- **Any LLM client** — `lfx.wrap()` works with any object that has `.complete()`

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│                    lfx.wrap()                    │
│         (intercepts LLM calls for learning)     │
└──────────────────────┬──────────────────────────┘
                       │
              ┌────────▼────────┐
              │ EpisodeCollector │  thread-safe, LRU cache
              │  + RewardPipeline│  formatting gate
              └────────┬────────┘
                       │ on_batch()
              ┌────────▼────────┐
              │  AsyncLearner   │  background worker thread
              │  (queue-based)  │  overflow: drop/block
              └────────┬────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   ┌─────────┐   ┌──────────┐   ┌──────────┐
   │ Harness  │   │  Router  │   │ Weights  │
   │ (prompt) │   │ (model)  │   │ (LoRA)   │
   └─────────┘   └──────────┘   └──────────┘

   All layers: forward_backward → optim_step
   Atomic snapshot-rollback on failure
   Content-addressed StateID for reproducibility
```

Key design properties:
- **Non-blocking**: Learning runs in a background thread. Agent latency is unaffected.
- **Atomic rollback**: Every `optim_step()` snapshots state first. If it fails, state is restored.
- **Content-addressed state**: Each configuration gets a deterministic StateID (SHA-256). You can reproduce any agent state from its ID.
- **Regression safety**: `gate_for_deploy()` compares reward distributions before promoting a new state.

---

## Competitive Positioning

### vs. MetaClaw (API proxy + LoRA fine-tuning)

| | lfx | MetaClaw |
|---|---|---|
| Integration | SDK wrapper (1 line) | API proxy (infra change) |
| Learning layers | 3 independent (prompt, routing, weights) | 2 conflated (skills + LoRA) |
| Rollback | Atomic per-layer snapshot | Manual |
| Regression safety | Built-in gate_for_deploy | None |
| Reward system | Composable signals with priority | PRM only (majority vote) |
| Provider lock-in | None (any LLM) | Tinker cloud for training |
| Stagnation handling | Paradigm breakthrough | None |

### vs. building it yourself

Prompt engineering by hand works until it doesn't scale. lfx automates the feedback loop: observe → extract signals → reflect → update → validate → deploy. The Harness layer alone replaces manual prompt iteration with data-driven prompt evolution.

---

## Monetization

### Open Core Model

**Free (open source):**
- Harness layer (prompt/playbook evolution)
- EpisodeCollector + live mode pipeline
- Composable reward signals
- All extractors (execution, outcome, user feedback, formatting)
- AsyncLearner background processing
- Full API: `lfx.wrap()`, `LfXAgent`, direct loop

**Paid tiers:**

| Tier | What you get | Price |
|------|-------------|-------|
| **Pro** | Router layer (intelligent model selection), managed judge LLM, dashboard & analytics | $49/mo |
| **Team** | Multi-agent coordination, shared playbooks across team, A/B testing, gate_for_deploy automation | $199/mo |
| **Enterprise** | Weights layer (managed LoRA training + hot-swap), SLA, on-prem, custom extractors | Custom |

### Revenue Drivers

1. **Compute margin on judge calls** — lfx is the judge LLM provider for Pro+ tiers. Lazy invocation keeps costs down while marking up the calls.
2. **Training infrastructure** — Weights layer requires GPUs. Managed training-as-a-service at margin.
3. **Router optimization savings** — Router layer saves users money by routing simple queries to cheaper models. lfx captures a fraction of the savings.

### GTM Strategy

1. **Developer adoption** — `pip install lfx` + one-line wrap. Free Harness layer shows value immediately.
2. **Usage-based upgrade** — Once agents process enough traffic, Router and Weights layers unlock measurable cost/quality improvements.
3. **Framework partnerships** — Pre-built integrations with litellm, n8n, LangChain, CrewAI make lfx the default learning layer.

---

## Quick Start

```bash
pip install lfx
```

```python
import lfx

# Wrap your LLM client
wrapped = lfx.wrap(your_client, collector=lfx.EpisodeCollector(
    pipeline=lfx.RewardPipeline([]),
    batch_size=16,
))

# That's it. Your agent now learns from experience.
response = wrapped.complete([{"role": "user", "content": "Hello"}])
```
