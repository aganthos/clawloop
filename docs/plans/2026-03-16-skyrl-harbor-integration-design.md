# SkyRL + Harbor Integration for LfX

**Date**: 2026-03-16
**Status**: Draft
**Branch**: TBD (skyrl-harbor-integration)

## Goal

Replace the Weights layer stub with real SkyRL integration, wire in Harbor env format for episode collection, and create a unified Tinker-compatible training interface.

The training script selects a **mode** that controls what the Weights layer does:

- `mode=weight`: Weights layer delegates to SkyRL backend (real LoRA/GRPO training)
- `mode=harness_learning`: Weights layer is a stub (no weight training)

**In both modes, all three layers (Harness, Router, Weights) run every iteration.** The Harness layer always evolves prompts/playbooks. The mode only determines whether the Weights layer does real training or stays inert.

For this PR: one mode per run. Unified mode (real weight training + harness learning as two active Tinker backends in a single coordinated run) comes later.

## Architecture Overview

```
Training Script (lfx/train.py)
    │
    ├── mode=harness_learning → HarnessLearningBackend (wraps Harness layer)
    │                           └── GEPA prompt evolution, playbook, reflector
    │
    └── mode=weight           → SkyRLWeightsBackend (delegates to SkyRL AbstractBackend)
                                └── All SkyRL capabilities: PPO/CISPO/CE/IS losses,
                                    GRPO/GAE/RLOO/REINFORCE++ advantage estimators,
                                    FSDP2/Megatron strategies, multi-adapter LoRA,
                                    fully async training, off-policy correction

Episodes sourced from:
    ├── HarborTaskEnvironment  → Harbor trials (sandboxed agent + verifier)
    └── EpisodeCollector       → Live agent traffic (existing, unchanged)
```

## File Layout

```
lfx/
├── backends/
│   ├── __init__.py           # Exports LfXBackend, SkyRLWeightsBackend, HarnessLearningBackend
│   ├── base.py               # LfXBackend protocol
│   ├── harness_learning.py   # HarnessLearningBackend
│   └── skyrl.py              # SkyRLWeightsBackend
│
├── envs/
│   ├── __init__.py
│   └── harbor.py             # HarborTaskEnvironment + HarborAdapter
│
├── utils/
│   └── async_bridge.py       # run_async() helper for sync/async contexts
│
├── layers/
│   └── weights.py            # Updated — delegates to backend when available
│
├── exporters/
│   └── skyrl.py              # Unchanged — Episode → GeneratorOutput
│
├── core/
│   └── loop.py               # Unchanged — already orchestrates layers
│
└── train.py                  # Unified training entry point
```

## 1. LfXBackend Protocol

Identical to the existing Layer protocol. Backends ARE layers.

```python
class LfXBackend(Protocol):
    def forward_backward(self, data: Datum) -> Future[FBResult]: ...
    def optim_step(self) -> Future[OptimResult]: ...
    def sample(self, ctx: SampleContext) -> Future[SampleResult]: ...
    def save_state(self, name: str) -> Future[SaveResult]: ...
    def load_state(self, state: dict) -> Future[LoadResult]: ...
    def clear_pending_state(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
```

The `load_state` dict schema is exactly what `to_dict()` returns. This is the existing convention for all LfX layers.

## 2. SkyRLWeightsBackend

Wraps SkyRL's `AbstractBackend`. All SkyRL config passes through as dicts — LfX does not define dataclasses for SkyRL's configuration knobs.

### Config

```python
@dataclass
class SkyRLWeightsConfig:
    base_model: str              # e.g. "Qwen/Qwen3-8B"
    backend_type: str            # "jax" or "skyrl_train"
    backend_config: dict         # Passed straight to SkyRL backend
    lora_config: dict            # rank, alpha, target_modules, init_method, etc.
    training_config: dict        # loss_fn, advantage_estimator, adam_params, all SkyRL knobs
    tokenizer_name: str          # HF tokenizer for Episode → GeneratorOutput
```

### Init Lifecycle

1. Validate imports: `skyrl.tinker.types`, `skyrl.backends` (fail with `BackendError(code="import_error")`)
2. Load tokenizer, validate `apply_chat_template()` works (fail with `BackendError(code="tokenizer_mismatch")`)
3. Instantiate SkyRL backend (`JaxBackend` or `SkyRLTrainBackend`) with `backend_config`
4. Call `backend.create_model(model_id, lora_config)` — creates LoRA adapter + optimizer
5. If `backend_config.enable_http_endpoint`: store `inference_url` from backend. If not set, `inference_url` is `None` — Harbor agents must have their own `api_base` configured in `trial_config`, or they use a local model (non-Harbor usage).
6. Ready for `forward_backward` / `optim_step` cycles

Only chat models are supported (must have `apply_chat_template`). Non-chat models are out of scope.

### Data Flow

```
Episode (LfX native)
  │
  ▼  SkyRLExporter.export()
GeneratorOutput (SkyRL public type)
  │  prompt_token_ids, response_ids, loss_masks,
  │  sparse rewards, trajectory_ids, rollout_logprobs
  │
  ▼  _to_forward_backward_input()
ForwardBackwardInput (SkyRL public Tinker type)
  │  model_input chunks, loss_fn_inputs
  │  loss_fn: str, loss_fn_config: dict
  │
  ▼  self._backend.forward_backward(prepared_batch)
ForwardBackwardOutput (SkyRL public type)
  │  loss_fn_outputs, metrics
  │
  ▼  _to_fb_result()
FBResult (LfX type)
```

Only public SkyRL types cross the boundary. The `_to_forward_backward_input()` translation is the glue code — small, tested, protected by submodule pin.

### Advantage Computation

LfX computes GRPO advantages before calling the SkyRL backend. SkyRL's
`AbstractBackend.forward_backward` takes `PreparedModelPassBatch` which
requires pre-computed `all_advantages`. LfX performs group-mean subtraction
across rollouts sharing the same `task_id` (via `trajectory_ids`):
`advantage_i = reward_i - mean(rewards in group)`.

This is simple GRPO. For GAE, RLOO, or REINFORCE++, LfX can delegate to
SkyRL's `compute_advantages_and_returns` utility (future enhancement).

### forward_backward(data: Datum)

1. Receive `Datum` with list of Episodes
2. Use `SkyRLExporter` to convert Episodes → `GeneratorOutput`
3. Translate `GeneratorOutput` → `PreparedModelPassBatch` (real SkyRL Pydantic type):
   - `all_input_ids`: prompt_token_ids + response_ids (concatenated per sequence)
   - `all_targets`: response_ids
   - `all_token_weights`: loss_masks cast to float
   - `all_sampling_logprobs`: from episode rollout_logprobs (zeros if unavailable)
   - `all_advantages`: GRPO group-mean advantages broadcast to response token length
   - `all_model_ids`, `all_loss_fns`, `all_loss_fn_configs`: from training_config
   - `request_batch_slices`: one slice per sequence
4. Call `backend.forward_backward(prepared_batch)`
5. Return `FBResult(status="ok", metrics={loss_fn_outputs, metrics})`
6. On error: return `FBResult(status="error", metrics={"error": BackendError(...)})`

### optim_step()

1. Build `OptimStepInput` from `training_config.adam_params` (lr, betas, eps, weight_decay)
2. Call `backend.optim_step(model_id, optim_input)`
3. Return `OptimResult(status="ok", updates_applied=1, metrics={grad_norm, lr, step_count})`
4. On error: return `OptimResult(status="error", updates_applied=0, metrics={...})`

### clear_pending_state()

No-op. SkyRL's backend manages its own gradient buffers internally.

### to_dict()

```python
def to_dict(self) -> dict:
    return {
        "model_ref": self.config.base_model,
        "backend_type": self.config.backend_type,
        "backend_config": self.config.backend_config,
        "lora_config": self.config.lora_config,
        "training_config": self.config.training_config,
        "adapter_refs": self._adapter_refs,
    }
```

All config included for StateID hashing (content-addressed reproducibility).

### Checkpoints

- `save_state(name)` → `backend.save_checkpoint(path, model_id)`, returns `SaveResult`
- `load_state(state)` → validates required keys from `to_dict()` schema, calls `backend.load_checkpoint(adapter_refs[-1], model_id)` if adapters exist, returns `LoadResult`

`adapter_refs` is append-only (chronological by construction), so `[-1]` is always the latest.

## 3. BackendError

Structured error type for all backend failures:

```python
@dataclass(frozen=True)
class BackendError:
    code: str
    message: str
    recoverable: bool

    @classmethod
    def from_exception(cls, e: Exception) -> "BackendError":
        # Maps known SkyRL exceptions to stable codes
        ...
```

Error codes:
- `gpu_oom` — GPU out of memory
- `tokenizer_mismatch` — tokenizer incompatible with model
- `backend_unreachable` — SkyRL backend not responding
- `invalid_config` — bad config values
- `training_diverged` — NaN loss or exploding gradients
- `schema_incompatible` — SkyRL Tinker type mismatch (version drift)
- `import_error` — SkyRL module not importable
- `unknown` — unmapped exception

Surfaces in `FBResult.metrics["error"]` / `OptimResult.metrics["error"]`. The learning loop checks `result.status == "error"` and triggers cross-layer rollback (existing mechanism).

## 4. HarnessLearningBackend

Thin wrapper around the existing Harness layer. Pure delegation.

```python
class HarnessLearningBackend:
    def __init__(self, harness: Harness, config: HarnessLearningConfig | None = None):
        self._harness = harness
        self._config = config or HarnessLearningConfig()

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        return self._harness.forward_backward(data)

    def optim_step(self) -> Future[OptimResult]:
        return self._harness.optim_step()

    # ... all methods delegate to self._harness
```

`HarnessLearningConfig` is currently unused — placeholder for unified mode where the training script configures harness params (reflector scheduling, paradigm thresholds) alongside weight params in one config. Contains `reflector_enabled`, `intensity_config`, `paradigm_enabled` with defaults.

The Harness layer's internal mechanisms (GEPA prompt evolution, playbook ACE reflector, tool config tuning, adaptive intensity, paradigm breakthrough) are unchanged.

## 5. HarborTaskEnvironment

Runs Harbor trials and produces LfX Episodes. Implements `TaskEnvironment`. Harbor is an optional dependency (`pip install lfx[harbor]`).

### Init

```python
class HarborTaskEnvironment:
    def __init__(
        self,
        task_dir: Path,
        trial_config: dict,
        reward_transform: Callable[[float], float] | None = None,
        train_on_truncated: bool = True,
    ):
        from harbor.models.trial.config import TrialConfig
        from harbor.trial import Trial
        self._Trial = Trial
        self._TrialConfig = TrialConfig

        self._task_dir = task_dir
        self._trial_config = trial_config
        self._reward_transform = reward_transform
        self._train_on_truncated = train_on_truncated

        # Validate required config keys
        if "agent" not in trial_config:
            raise ValueError("trial_config must contain 'agent' key")
        trial_config.setdefault("task", {})
        trial_config["agent"].setdefault("kwargs", {})

    @property
    def task_id(self) -> str:
        return self._task_dir.name
```

### run_episode

```python
async def run_episode(self, agent_state: AgentState) -> Episode:
    config = deepcopy(self._trial_config)
    config["task"]["path"] = str(self._task_dir)
    config["agent"]["kwargs"]["session_id"] = uuid4().hex

    # Inject current model endpoint from agent_state
    if agent_state.inference_url:
        config["agent"]["kwargs"]["api_base"] = agent_state.inference_url

    # Inject current system prompt from harness
    if agent_state.harness:
        sample_result = agent_state.harness.sample(
            SampleContext(bench=self._task_dir.name)
        )
        config["agent"]["kwargs"]["system_prompt_override"] = sample_result.result().output

    trial = self._Trial(self._TrialConfig(**config))

    try:
        results = await trial.run()
    except ContextLengthExceededError:
        if self._train_on_truncated:
            return self._build_episode(agent_state, reward=0.0, metadata={"truncated": True})
        else:
            return self._build_episode(agent_state, filtered=True, metadata={"truncated": True})
    except AgentTimeoutError:
        return self._build_episode(agent_state, filtered=True, metadata={"timeout": True})
    except Exception as e:
        return self._build_episode(agent_state, filtered=True, metadata={"error": type(e).__name__})

    raw_reward = results.verifier_result.rewards.get("reward", 0.0)
    try:
        reward = self._reward_transform(raw_reward) if self._reward_transform else raw_reward
    except Exception:
        reward = raw_reward
        metadata["reward_transform_error"] = True

    chat_history = results.agent_result.metadata.get("all_messages", [])
    score_breakdown = results.verifier_result.rewards

    return self._build_episode(
        agent_state,
        chat_history=chat_history,
        reward=reward,
        score_breakdown=score_breakdown,
        metadata={"raw_reward": raw_reward, "transformed_reward": reward},
    )
```

### Episode Construction

```python
def _build_episode(self, agent_state, chat_history=None, reward=0.0,
                   filtered=False, score_breakdown=None, metadata=None) -> Episode:
    messages = [Message.from_openai_dict(m) for m in (chat_history or [])]
    step_boundaries = _compute_step_boundaries(messages)
    steps = _build_steps(messages, step_boundaries, reward)

    summary = EpisodeSummary(filtered=filtered, score_breakdown=score_breakdown)
    if not filtered:
        summary.signals["outcome"] = RewardSignal(
            name="outcome", value=float(reward), confidence=1.0
        )

    return Episode(
        id=uuid4().hex,
        state_id=agent_state.state_id if agent_state else None,
        task_id=self.task_id,
        messages=messages,
        step_boundaries=step_boundaries,
        steps=steps,
        summary=summary,
        metadata=metadata or {},
    )
```

Rewards are stored as-is. `RewardSignal`'s `__post_init__` clamps to [-1, 1] — this is the global LfX boundary, same for all reward sources. For non-standard scales (e.g., 0-100), use `reward_transform`.

Filtered episodes (timeout, failure) are excluded from training by the existing learning loop — same mechanism as `FormattingFilter`.

`state_id` is set from `agent_state` for off-policy staleness tracking.

### Error Handling

| Error | Trainable? | Reward | Configurable? |
|-------|-----------|--------|--------------|
| Success | Yes | From verifier | — |
| ContextLengthExceeded | Default yes | 0.0 | `train_on_truncated` |
| AgentTimeout | No (filtered) | — | — |
| Other exceptions | No (filtered) | — | — |

Harbor retries internally (up to 2x) before raising exceptions.

### No ATIF Parsing (This PR)

Uses `all_messages` (OpenAI chat format) directly. ATIF's token-level bookkeeping (`completion_token_ids`, `logprobs`) is future work — would bypass re-tokenization but requires deeper integration.

## 6. HarborAdapter

Sync wrapper around `HarborTaskEnvironment` list. Implements `AdapterLike`.

```python
class HarborAdapter:
    def __init__(self, envs: list[HarborTaskEnvironment]):
        self._envs = {env.task_id: env for env in envs}

    def run_episode(self, task: str, agent_state: AgentState) -> Episode:
        return run_async(self._envs[task].run_episode(agent_state))

    def run_batch(self, tasks: list[str], agent_state: AgentState,
                  n_per_task: int = 1) -> list[Episode]:
        async def _gather():
            coros = [self._envs[t].run_episode(agent_state)
                     for t in tasks for _ in range(n_per_task)]
            return await asyncio.gather(*coros)
        return run_async(_gather())
```

### Async Bridge

```python
# lfx/utils/async_bridge.py
import asyncio
from concurrent.futures import ThreadPoolExecutor

_EXECUTOR = ThreadPoolExecutor(max_workers=1)

def run_async(coro):
    """Run async coroutine from sync code safely.

    No event loop running: asyncio.run() (fast path).
    Event loop running (Jupyter, async orchestration): thread pool with own loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        future = _EXECUTOR.submit(asyncio.run, coro)
        return future.result()
```

## 7. Async RL Support

The design is **compatible with** SkyRL's fully async training, but this PR only implements the synchronous path:

**This PR (synchronous)**: LfX `learning_loop()` — collect batch → `forward_backward` → `optim_step` → repeat. Rollouts within a batch run concurrently via `asyncio.gather()`, but training is synchronous (batch completes before next batch starts).

**Future (async orchestration, NOT this PR)**: SkyRL's `FullyAsyncTrainer` replaces LfX's learning loop as the orchestrator. `HarborTaskEnvironment` feeds a concurrent episode queue. Training steps run in parallel with generation. This is listed in "What's NOT in this PR."

**What this PR provides for the async future**:

1. **Concurrent rollouts**: `run_episode()` is async. Many trials run in parallel. Concurrency bounded by sandbox provider.
2. **Inference endpoint is external**: `trial_config.agent.kwargs.api_base` points to SkyRL's vLLM server. Weight updates happen at the vLLM level (LoRA hot-swap) — transparent to Harbor.
3. **Staleness tracking**: Episodes carry `state_id`. SkyRL's off-policy correction can use this.

**Assumption**: The learning loop is sequential (one batch at a time). The `run_async()` helper uses a single-thread executor. Re-entrant or parallel loop calls are not supported.

## 8. Updated Weights Layer

```python
@dataclass
class Weights:
    model_ref: str = ""
    adapter_refs: list[str] = field(default_factory=list)
    grpo_config: GRPOConfig = field(default_factory=GRPOConfig)
    training_history: list[dict] = field(default_factory=list)
    _pending: _WeightsPending = field(default_factory=_WeightsPending)
    _backend: LfXBackend | None = None  # Optional real backend

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        if self._backend:
            return self._backend.forward_backward(data)
        return self._stub_forward_backward(data)

    def optim_step(self) -> Future[OptimResult]:
        if self._backend:
            return self._backend.optim_step()
        return self._stub_optim_step()

    # sample, save_state, load_state, clear_pending_state, to_dict
    # all delegate to backend when available, stub otherwise
```

Backward compatible. No backend = stub mode (existing tests pass unchanged).

## 9. Training Script

```python
# lfx/train.py
def train(config: TrainConfig):
    # 1. Always build harness and router
    harness = Harness(
        system_prompts={b: config.system_prompt for b in config.benches},
        reflector=Reflector(llm=reflector_llm) if config.harness.reflector_enabled else None,
    )
    router = Router(tier_models=config.router.tier_models)

    # 2. Build environments
    if config.env_type == "harbor":
        envs = [HarborTaskEnvironment(d, config.harbor.trial_config,
                    reward_transform=config.harbor.reward_transform,
                    train_on_truncated=config.harbor.train_on_truncated)
                for d in config.harbor.task_dirs]

    # 3. Build backend based on mode
    if config.mode == "weight":
        backend = SkyRLWeightsBackend(config.skyrl)
        weights = Weights(model_ref=config.skyrl.base_model, _backend=backend)
    elif config.mode == "harness_learning":
        weights = Weights()  # Stub

    # 4. Build agent state
    agent_state = AgentState(
        harness=harness, router=router, weights=weights,
        inference_url=backend.inference_url if config.mode == "weight" else None,
    )

    # 5. Run learning loop (unchanged)
    learning_loop(
        adapter=HarborAdapter(envs),
        agent_state=agent_state,
        tasks=[env.task_id for env in envs],
        n_episodes=config.episodes_per_iter,
        n_iterations=config.n_iterations,
    )
```

### YAML Config

```yaml
mode: weight  # or harness_learning
env_type: harbor
harbor:
  task_dirs: ["/data/harbor/code-contests/"]
  trial_config:
    agent:
      name: terminus-2
      override_timeout_sec: 1200
      kwargs:
        max_turns: 32
        store_all_messages: true
        temperature: 1.0
    environment:
      type: daytona
skyrl:
  base_model: Qwen/Qwen3-8B
  backend_type: jax
  backend_config:
    tensor_parallel_size: 4
  lora_config:
    rank: 32
    alpha: 16.0
  training_config:
    loss_fn: ppo
    advantage_estimator: grpo
    adam_params:
      learning_rate: 1.0e-6
      weight_decay: 0.01
```

`env_type: "custom"` is out of scope. Users with custom environments use `learning_loop()` directly.

In weight mode, Harness still runs each iteration (prompt evolution alongside weight training). Cross-layer rollback handles failure isolation.

Config validation: `TrainConfig` uses Pydantic with field validators. Missing required fields fail at load time.

## 10. AgentState Changes

New field on `AgentState`:

```python
@dataclass
class AgentState:
    harness: Harness
    router: Router
    weights: Weights
    inference_url: str | None = None  # NEW: vLLM endpoint for Harbor agents
```

Set by `SkyRLWeightsBackend` at init. Read by `HarborTaskEnvironment` in `run_episode()` to configure the agent's API base. When SkyRL hot-swaps LoRA after `optim_step`, the URL stays the same — only the weights behind it change.

## 11. Submodule Update Strategy

- **Pin to tagged releases**: Only update to SkyRL tagged releases (e.g., v0.3.0).
- **Update process**: `git checkout` tag → compat check → test suite → commit
- **CI compat gate**: A test that builds a minimal `Episode` fixture (using existing test helpers), runs it through `SkyRLExporter.export()` → `_to_forward_backward_input()`, and validates the output has expected fields. Catches schema drift on the full translation path.
- **CI env isolation**: Ensure CI uses the submodule `skyrl` (via `PYTHONPATH` or `sys.path` manipulation), not any system-installed `skyrl` package.
- **Frequency**: On SkyRL releases we need, or monthly cadence.
- **Breaking changes**: CI catches them. Fix adapter in same PR as submodule bump.
- **No vendoring**: Submodule IS the dependency.
- **`git submodule update --init --recursive`** required in CI for clean checkouts.

## What's NOT in This PR

- Unified mode (both active Tinker backends in one coordinated run)
- Tinker REST API server for LfX (HTTP interface)
- Advantage estimator configuration in LfX (SkyRL owns this)
- New loss functions in LfX (SkyRL owns this)
- Harbor task registry integration
- Dashboard/analytics for training metrics
- ATIF trajectory parsing (token-level bookkeeping — future)
- Non-chat model support
- Async RL orchestration (SkyRL's FullyAsyncTrainer as loop replacement)
- Custom env_type support in train.py (users with custom envs use learning_loop() directly)
