# Concepts

This page explains ClawLoop's core types and how they fit together.

## The Learning Loop

```
Environment → Episodes → Layers → Improved Agent → Environment → ...
```

An agent interacts with an environment. ClawLoop collects **episodes** —
structured traces of messages, tool calls, and rewards. Learning **layers**
process these episodes and update the agent. Repeat.

## Episodes

### Episode

One complete agent trajectory: a sequence of messages with step boundaries
and reward signals.

```python
episode.messages           # list[Message] — full conversation in OpenAI format
episode.steps              # list[StepMeta] — per-turn metadata (reward, timing)
episode.summary            # EpisodeSummary — aggregate metrics
episode.terminal_reward()  # float — final reward
```

### EpisodeSummary

Aggregate metrics for a completed episode. Stores named reward signals
with priority-based resolution: user > outcome > execution > judge.

```python
summary.effective_reward()   # float in [-1, 1] — priority-resolved
summary.normalized_reward()  # float in [0, 1] — for compatibility
summary.needs_judge()        # bool — should an LLM judge score this?
summary.signals              # dict[str, RewardSignal]
```

### Datum

The input bundle passed to each learning layer — a batch of episodes plus
loss function configuration.

```python
datum = Datum(episodes=[ep1, ep2, ...], loss_fn="default")
layer.forward_backward(datum)
```

## Layers

All three layers implement the **Layer Protocol** — a two-phase contract:

1. **`forward_backward(data)`** — accumulate updates without mutating state
2. **`optim_step()`** — apply updates atomically; rollback on failure

### Harness

Prompt optimization and memory. Combines three mechanisms:

- **Reflector** — LLM reads episode traces, extracts reusable strategies as
  `Insight` objects (not task-specific answers, but general patterns)
- **Playbook Curator** — retrieve-classify-revise pipeline that integrates
  insights, merges duplicates, resolves conflicts, and prunes bad entries
- **GEPA** — Pareto-front prompt evolution via mutation and crossover

```python
harness.system_prompt("math")  # returns prompt with injected playbook entries
harness.playbook               # current learned strategies
```

### Router

Trainable model routing. Maps queries to the cheapest capable model using
a multi-dimension complexity scorer.

```python
router.route(features)    # returns model_id for this query
router.classify(features) # returns tier: LIGHT, MEDIUM, HEAVY, REASONING
```

### Weights

Model fine-tuning via SkyRL. Tracks base model, LoRA adapters, and GRPO
training configuration.

```python
weights.active_adapter  # current LoRA adapter reference
weights.grpo_config     # GRPOConfig with learning rate, KL coeff, etc.
```

## State

### AgentState

Bundle of all three layers. Provides a content-addressed fingerprint for
reproducibility.

```python
agent_state = AgentState()
agent_state.harness   # Harness layer
agent_state.router    # Router layer
agent_state.weights   # Weights layer
agent_state.state_id() # StateID — SHA-256 hash of full config
```

### StateID

Content-addressed fingerprint (SHA-256) across all layers. Two agents with
identical configurations produce the same `StateID`.

```python
state_id.combined_hash  # single hash for the full configuration
state_id.harness_hash   # hash of harness layer alone
```

## Evolution

### Evolver

Pluggable interface for harness optimization backends. The community edition
ships `LocalEvolver` (Reflector + GEPA + Paradigm). Enterprise backends
provide broader search via evolutionary algorithms.

```python
result = evolver.evolve(episodes, harness_state, context)
result.insights     # new playbook entries
result.candidates   # prompt candidates for GEPA fronts
```

### Paradigm Breakthrough

Stagnation escape mechanism. When rewards plateau, asks a strong LLM
for fundamentally new strategic directions rather than incremental refinements.
