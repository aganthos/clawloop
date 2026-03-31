# Adding Environments

ClawLoop environments are pluggable via the `EnvAdapter` interface.

## The Adapter Interface

```python
from clawloop.adapters.base import EnvAdapter
from clawloop.core.episode import Episode
from clawloop.core.loop import AgentState

class MyAdapter(EnvAdapter):
    def setup(self, config: dict) -> None:
        """Initialize from config (model, paths, credentials)."""
        ...

    def run_episode(self, task: Any, agent_state: AgentState) -> Episode:
        """Run one agent trajectory and return a structured Episode."""
        ...

    def list_tasks(self, split: str = "test") -> list:
        """Return available task IDs."""
        ...
```

## Building an Episode

Your `run_episode` must return an `Episode` with messages, steps, and reward
signals:

```python
from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.reward import RewardSignal

episode = Episode(
    id=str(uuid4()),
    state_id=agent_state.state_id().combined_hash,
    task_id=task_id,
    bench="my_bench",
    messages=[
        Message(role="system", content=system_prompt),
        Message(role="user", content=task_prompt),
        Message(role="assistant", content=agent_response),
    ],
    step_boundaries=[1],  # agent turn starts at message index 1
    steps=[StepMeta(t=0, reward=score, done=True, timing_ms=0.0)],
    summary=EpisodeSummary(
        signals={"outcome": RewardSignal(name="outcome", value=score, confidence=1.0)},
    ),
)
```

**Existing adapters to learn from:**

- [`clawloop/envs/math.py`](https://github.com/aganthos/clawloop/blob/main/clawloop/envs/math.py) — minimal (~80 lines), good starting point
- [`clawloop/envs/harbor.py`](https://github.com/aganthos/clawloop/blob/main/clawloop/envs/harbor.py) — sandboxed agent tasks via Docker
- [`clawloop/adapters/car.py`](https://github.com/aganthos/clawloop/blob/main/clawloop/adapters/car.py) — external process orchestration (agentbeats-run)
- [`clawloop/adapters/entropic.py`](https://github.com/aganthos/clawloop/blob/main/clawloop/adapters/entropic.py) — CRMArena A2A benchmark

## Registering Your Adapter

Add a builder function to the training entrypoint:

```python
# clawloop/train.py
def _build_my_env(config, llm_clients):
    adapter = MyAdapter()
    adapter.setup(config)
    tasks = adapter.list_tasks()
    return adapter, tasks

ENV_BUILDERS["my_env"] = _build_my_env
```

Then run:

```bash
python examples/train_runner.py my_config.json
```

## Reward Signals

Episodes carry named reward signals with a priority system:

| Priority | Source | When to use |
|----------|--------|-------------|
| 1 (highest) | `user` | Explicit human feedback |
| 2 | `outcome` | Verifiable correctness (math, code tests) |
| 3 | `execution` | Tool call success, format compliance |
| 4 (lowest) | `judge` | LLM-as-judge scoring |

`EpisodeSummary.effective_reward()` resolves to the highest-priority signal
available. If only low-confidence execution signals exist,
`summary.needs_judge()` returns `True` — useful for triggering LLM judge
evaluation only when needed.
