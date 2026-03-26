"""Unified training entry point.

Two modes:
  weight           — SkyRL GRPO/PPO weight training (GPU)
  harness_learning — prompt/playbook optimization via reflector LLM (no GPU)

Environments are pluggable via ENV_BUILDERS registry. Each builder is a
function (config, llm_clients) -> (adapter, tasks) that constructs an
AdapterLike and a task list for the learning loop.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, SecretStr


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODE_LAYERS: dict[str, list[str]] = {
    "weight": ["weights"],
    "harness_learning": ["harness", "router"],
    "full": ["harness", "router", "weights"],
}


class LLMClientConfig(BaseModel):
    """LLM client configuration for reflector or task inference."""

    model: str
    api_base: str = ""
    api_key: SecretStr = SecretStr("")
    temperature: float = 0.7
    max_tokens: int = 2000

    model_config = {"arbitrary_types_allowed": True}


class HarborConfig(BaseModel):
    """Harbor environment configuration."""

    task_dirs: list[str]
    trial_config: dict[str, Any] = {}
    train_on_truncated: bool = True

    model_config = {"arbitrary_types_allowed": True}


class TrainConfig(BaseModel):
    """Unified training configuration."""

    mode: Literal["weight", "harness_learning", "full"]
    env_type: str = "harbor"

    llm_clients: dict[str, LLMClientConfig] = {}
    skyrl: dict[str, Any] | None = None
    harbor: HarborConfig | None = None
    env_config: dict[str, Any] | None = None

    system_prompt: str = "You are a helpful assistant."
    benches: list[str] = ["default"]
    episodes_per_iter: int = 10
    n_iterations: int = 100

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# LLM client helper
# ---------------------------------------------------------------------------

def _make_llm_client(cfg: LLMClientConfig):
    """Build a LiteLLMClient from config."""
    from lfx.llm import LiteLLMClient

    key = cfg.api_key.get_secret_value() or None
    return LiteLLMClient(
        model=cfg.model,
        api_key=key,
        api_base=cfg.api_base or None,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )


# ---------------------------------------------------------------------------
# Environment builders — each returns (adapter, tasks)
# ---------------------------------------------------------------------------

def _build_harbor(config: TrainConfig, llm_clients: dict[str, LLMClientConfig]):
    from lfx.envs.harbor import HarborAdapter, HarborTaskEnvironment

    envs = [
        HarborTaskEnvironment(
            task_dir=Path(d),
            trial_config=config.harbor.trial_config,
            train_on_truncated=config.harbor.train_on_truncated,
        )
        for d in config.harbor.task_dirs
    ]
    return HarborAdapter(envs), [env.task_id for env in envs]


def _build_math(config: TrainConfig, llm_clients: dict[str, LLMClientConfig]):
    from lfx.envs.math import MathAdapter, MathEnvironment

    task_client = _make_llm_client(llm_clients["task"])
    math_env = MathEnvironment()
    tasks = math_env.get_tasks()
    return MathAdapter(env=math_env, client=task_client), [s.question for s in tasks]


def _build_entropic(config: TrainConfig, llm_clients: dict[str, LLMClientConfig]):
    from lfx.adapters.entropic import EntropicAdapter

    entropic_cfg = dict(config.env_config or {})
    if "task" in llm_clients:
        tc = llm_clients["task"]
        entropic_cfg.setdefault("model", tc.model)
        key = tc.api_key.get_secret_value() if tc.api_key else None
        if key:
            entropic_cfg.setdefault("api_key", key)
        if tc.api_base:
            entropic_cfg.setdefault("api_base", tc.api_base)

    adapter = EntropicAdapter()
    adapter.setup(entropic_cfg)
    # Default to 3 tasks (CRMArena indices 0-2) if not specified in env_config
    n_tasks = entropic_cfg.get("task_limit", len(entropic_cfg.get("task_ids", [0, 1, 2])))
    return adapter, [f"base_{i}" for i in range(n_tasks)]


# ---------------------------------------------------------------------------
# Environment registry — add new envs here
# ---------------------------------------------------------------------------

ENV_BUILDERS: dict[str, Any] = {
    "harbor": _build_harbor,
    "math": _build_math,
    "entropic": _build_entropic,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(config: TrainConfig) -> list[str]:
    """Validate config consistency. Returns the active layers list."""
    if config.mode == "full":
        raise NotImplementedError(
            "mode='full' is disabled. The support-query split needs rework: "
            "GRPO requires all episodes for advantage variance, and the "
            "on-policy boundary after harness updates is unresolved. "
            "Use mode='weight' or mode='harness_learning' separately."
        )

    layers = MODE_LAYERS[config.mode]

    if "weights" in layers and not config.skyrl:
        raise ValueError(f"mode='{config.mode}' requires 'skyrl' config for weight training")

    if "harness" in layers and "reflector" not in config.llm_clients:
        raise ValueError(f"mode='{config.mode}' requires 'reflector' in llm_clients")

    if config.env_type not in ENV_BUILDERS:
        raise ValueError(
            f"Unknown env_type: {config.env_type!r}. "
            f"Available: {sorted(ENV_BUILDERS.keys())}"
        )

    # Env-specific validation (fail fast before expensive backend init)
    if config.env_type == "harbor":
        if not config.harbor or not config.harbor.task_dirs:
            raise ValueError("harbor env requires harbor.task_dirs")
    if config.env_type == "math":
        if "task" not in config.llm_clients:
            raise ValueError("math env requires 'task' in llm_clients")
    if config.env_type == "entropic":
        if not config.env_config:
            raise ValueError("entropic env requires 'env_config'")

    return layers


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(config: TrainConfig):
    """Unified training entry point.

    Builds layers, environment adapter, and runs the learning loop.
    Environment is selected via env_type and constructed by the matching
    builder from ENV_BUILDERS.
    """
    from lfx.core.intensity import AdaptiveIntensity
    from lfx.core.loop import AgentState, learning_loop
    from lfx.layers.harness import Harness
    from lfx.layers.router import Router
    from lfx.layers.weights import Weights

    layers = validate_config(config)

    # 1. Harness — always created; reflector wired only if harness is active
    prompts = {b: config.system_prompt for b in config.benches}
    # Auto-register env_type as prompt key so harness.sample(bench=env_type) works
    prompts.setdefault(config.env_type, config.system_prompt)
    harness = Harness(system_prompts=prompts)

    if "harness" in layers:
        from lfx.core.reflector import Reflector

        reflector_client = _make_llm_client(config.llm_clients["reflector"])
        harness.reflector = Reflector(client=reflector_client)

    # 2. Weights backend
    backend = None
    if "weights" in layers:
        from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig

        skyrl_cfg = SkyRLWeightsConfig(**config.skyrl)
        backend = SkyRLWeightsBackend(skyrl_cfg)
        weights = Weights(model_ref=skyrl_cfg.base_model, _backend=backend)
    else:
        weights = Weights()

    router = Router()

    # 3. Environment adapter — dispatched via registry
    build_env = ENV_BUILDERS[config.env_type]
    adapter, tasks = build_env(config, config.llm_clients)

    # 4. Agent state
    agent_state = AgentState(
        harness=harness,
        router=router,
        weights=weights,
        inference_url=getattr(backend, "inference_url", None) if backend else None,
    )

    # 5. Learning loop
    return learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=tasks,
        n_episodes=config.episodes_per_iter,
        n_iterations=config.n_iterations,
        active_layers=layers,
        intensity=AdaptiveIntensity() if "harness" in layers else None,
    )
