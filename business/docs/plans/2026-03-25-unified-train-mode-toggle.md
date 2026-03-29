# Unified train.py with clean mode presets

## Goal

Make `train.py` support a single `mode` flag that cleanly toggles between
weight training, harness (prompt/playbook) learning, and multi-layer learning.
Same config shape, change one field.

## Three modes

| mode | layers trained | GPU | reflector LLM | description |
|------|---------------|-----|---------------|-------------|
| `weight` | weights | yes | no | SkyRL GRPO/PPO training |
| `harness_learning` | harness, router | no | yes | prompt/playbook optimization |
| `full` | harness, router, weights | yes | yes | multi-layer: failures→harness, successes→weights |

No `active_layers` field. Mode IS the layers. If we need finer control later,
we add `active_layers` as a separate feature.

## Inference provider rule (per env)

Simple, no conditionals:

- **Harbor**: uses `agent_state.inference_url` (SkyRL endpoint) in weight/full
  mode. In harness_learning mode, Harbor's agent uses its own configured model.
  Validation: weight/full + harbor must produce a non-None inference_url, else
  hard error at config time.
- **Math**: always uses `llm_clients["task"]`. Math env runs an LLM client to
  generate responses, never a vLLM endpoint. In weight/full mode, the LLM
  generates episodes and SkyRL trains on them.

No generic "env supports inference_url" rule. Each env's inference path is
explicit.

## Config schema

```python
class LLMClientConfig(BaseModel):
    model: str
    api_base: str = ""
    api_key: SecretStr = ""  # never serialized in plain text
    temperature: float = 0.7
    max_tokens: int = 2000

class TrainConfig(BaseModel):
    mode: Literal["weight", "harness_learning", "full"]
    env_type: Literal["harbor", "math"] = "harbor"

    llm_clients: dict[str, LLMClientConfig] = {}
    skyrl: dict[str, Any] | None = None
    harbor: HarborConfig | None = None

    system_prompt: str = "You are a helpful assistant."
    benches: list[str] = ["default"]
    episodes_per_iter: int = 10
    n_iterations: int = 100
```

## Validation (complete, upfront)

```python
MODE_LAYERS = {
    "weight": ["weights"],
    "harness_learning": ["harness", "router"],
    "full": ["harness", "router", "weights"],
}

def validate_config(config: TrainConfig):
    layers = MODE_LAYERS[config.mode]

    if "weights" in layers and not config.skyrl:
        raise ValueError("mode requires skyrl config for weight training")

    if "harness" in layers and "reflector" not in config.llm_clients:
        raise ValueError("mode requires 'reflector' in llm_clients")

    if config.env_type == "harbor":
        if not config.harbor or not config.harbor.task_dirs:
            raise ValueError("harbor env requires harbor.task_dirs")

    if config.env_type == "math":
        if "task" not in config.llm_clients:
            raise ValueError("math env requires 'task' in llm_clients")
```

## Wiring in train()

```python
def train(config: TrainConfig):
    validate_config(config)
    layers = MODE_LAYERS[config.mode]

    # 1. Harness — always created, reflector only if harness is active
    harness = Harness(system_prompts=prompts)
    if "harness" in layers:
        reflector_cfg = config.llm_clients["reflector"]
        reflector_client = LiteLLMClient(
            model=reflector_cfg.model,
            api_key=reflector_cfg.api_key.get_secret_value(),
            api_base=reflector_cfg.api_base,
        )
        harness._reflector = Reflector(client=reflector_client)

    # 2. Weights backend
    if "weights" in layers:
        backend = SkyRLWeightsBackend(SkyRLWeightsConfig(**config.skyrl))
        weights = Weights(model_ref=..., _backend=backend)
    else:
        weights = Weights()

    # 3. Environment adapter
    if config.env_type == "harbor":
        adapter = HarborAdapter(build_harbor_envs(config))
    elif config.env_type == "math":
        task_cfg = config.llm_clients["task"]
        task_client = LiteLLMClient(
            model=task_cfg.model,
            api_key=task_cfg.api_key.get_secret_value(),
            api_base=task_cfg.api_base,
        )
        adapter = MathAdapter(MathEnvironment(), task_client)

    # 4. Agent state
    agent_state = AgentState(
        harness=harness, router=Router(), weights=weights,
        inference_url=getattr(backend, "inference_url", None) if "weights" in layers else None,
    )

    # 5. Learning loop
    return learning_loop(
        adapter=adapter, agent_state=agent_state,
        tasks=tasks, active_layers=layers,
        intensity=AdaptiveIntensity() if "harness" in layers else None,
        n_episodes=config.episodes_per_iter,
        n_iterations=config.n_iterations,
    )
```

## LLM bottleneck mitigation

Already handled by existing infrastructure:

- **AdaptiveIntensity** gates reflector calls — only fires on reward plateau
- **ParadigmBreakthrough** only fires on stagnation (even rarer)
- In `full` mode, harness reflection adds ~2-5s when it fires vs 10-30s for
  weight training. Not the bottleneck.

## API key security

- `Pydantic SecretStr` for api_key — `repr()` and `json()` show `**********`
- `ExperimentLog` logs rewards/metrics only, not config objects
- `to_dict()` on layers does not include LLM client config

## MathAdapter (new, ~40 lines)

```python
class MathAdapter:
    """Wraps MathEnvironment + LLM client as AdapterLike."""

    def __init__(self, env: MathEnvironment, client):
        self._env = env
        self._client = client

    def run_episode(self, task, agent_state) -> Episode:
        sample = self._env.get_sample(task)
        prompt = agent_state.harness.sample(
            SampleContext(bench="math")
        ).result().output
        response = self._client.complete([
            {"role": "system", "content": prompt or "Solve step by step."},
            {"role": "user", "content": sample.input},
        ])
        result = self._env.evaluate(sample, response)
        return _build_episode(sample, response, result, agent_state)
```

Note: `agent_state.harness` is always created (even in weight-only mode) so
`harness.sample()` is always safe. It returns the base system_prompt when no
playbook entries exist.

## Existing patterns leveraged

The learning loop already implements:

1. **Support-query separation**: DISABLED. The split (failures→harness,
   successes→weights) is incompatible with GRPO which needs all episodes
   for advantage variance. See roadmap Task 2.1 for rework plan.
2. **Generation-flush**: when `playbook_generation` advances, stale episodes
   flushed from weights buffer. Prevents training on pre-adaptation behavior.
3. **Cross-layer rollback**: if any layer's optim_step fails, all layers roll
   back to pre-optim snapshot.

## Files to modify

- `lfx/train.py` — LLMClientConfig, mode Literal, validation, wiring
- `lfx/envs/math.py` — add MathAdapter class (~40 lines)

## Files NOT modified

- `lfx/core/loop.py` — active_layers, support-query, generation-flush all work
- `lfx/backends/skyrl.py` — just validated on GPU
- `lfx/layers/harness.py` — _reflector field already exists

## Scope guard

This plan does NOT add:

- Custom env support (YAGNI — only harbor and math)
- Async/concurrent LLM calls (profile first)
- `active_layers` config field (mode presets are sufficient)
- Changes to the Layer protocol or Episode types

## Example configs

### Harness-only (math, no GPU)
```json
{
    "mode": "harness_learning",
    "env_type": "math",
    "system_prompt": "You are a math solver. Use \\boxed{} notation.",
    "llm_clients": {
        "reflector": {
            "model": "openai/claude-sonnet-4-5-20250929",
            "api_base": "http://127.0.0.1:8317/v1",
            "api_key": "key"
        },
        "task": {
            "model": "openai/claude-haiku-4-5-20251001",
            "api_base": "http://127.0.0.1:8317/v1",
            "api_key": "key"
        }
    },
    "n_iterations": 10,
    "episodes_per_iter": 5
}
```

### Weight-only (Harbor + SkyRL, GPU)
```json
{
    "mode": "weight",
    "env_type": "harbor",
    "harbor": {
        "task_dirs": ["/data/tasks/task1", "/data/tasks/task2"]
    },
    "skyrl": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "backend_type": "skyrl_train",
        "backend_config": {
            "strategy": "fsdp2",
            "trainer.placement.colocate_all": true
        },
        "lora_config": { "rank": 32 }
    },
    "n_iterations": 100,
    "episodes_per_iter": 10
}
```

### Full multi-layer (GPU + reflector)
```json
{
    "mode": "full",
    "env_type": "harbor",
    "harbor": { "task_dirs": ["..."] },
    "skyrl": {
        "base_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "backend_type": "skyrl_train",
        "backend_config": { "strategy": "fsdp2" },
        "lora_config": { "rank": 32 }
    },
    "llm_clients": {
        "reflector": {
            "model": "openai/claude-sonnet-4-5-20250929",
            "api_base": "http://127.0.0.1:8317/v1",
            "api_key": "key"
        }
    },
    "n_iterations": 100,
    "episodes_per_iter": 10
}
```
