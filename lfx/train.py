"""Unified training entry point.

Three modes:
  weight           — SkyRL GRPO/PPO weight training (GPU)
  harness_learning — prompt/playbook optimization via reflector LLM (no GPU)
  full             — multi-layer: failures→harness, successes→weights (GPU + LLM)
"""

from __future__ import annotations

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
    env_type: Literal["harbor", "math"] = "harbor"

    llm_clients: dict[str, LLMClientConfig] = {}
    skyrl: dict[str, Any] | None = None
    harbor: HarborConfig | None = None

    system_prompt: str = "You are a helpful assistant."
    benches: list[str] = ["default"]
    episodes_per_iter: int = 10
    n_iterations: int = 100

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_config(config: TrainConfig) -> list[str]:
    """Validate config consistency. Returns the active layers list."""
    layers = MODE_LAYERS[config.mode]

    if "weights" in layers and not config.skyrl:
        raise ValueError(f"mode='{config.mode}' requires 'skyrl' config for weight training")

    if "harness" in layers and "reflector" not in config.llm_clients:
        raise ValueError(f"mode='{config.mode}' requires 'reflector' in llm_clients")

    if config.env_type == "harbor":
        if not config.harbor or not config.harbor.task_dirs:
            raise ValueError("harbor env requires harbor.task_dirs")

    if config.env_type == "math":
        if "task" not in config.llm_clients:
            raise ValueError("math env requires 'task' in llm_clients")

    return layers


# ---------------------------------------------------------------------------
# Train
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


def train(config: TrainConfig):
    """Unified training entry point.

    Builds layers, environments, and backend based on config.mode,
    then runs the learning loop.
    """
    from lfx.core.intensity import AdaptiveIntensity
    from lfx.core.loop import AgentState, learning_loop
    from lfx.layers.harness import Harness
    from lfx.layers.router import Router
    from lfx.layers.weights import Weights

    layers = validate_config(config)

    # 1. Harness — always created; reflector wired only if harness is active
    prompts = {b: config.system_prompt for b in config.benches}
    prompts.setdefault("harbor", config.system_prompt)
    prompts.setdefault("math", config.system_prompt)
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

    # 3. Environment adapter
    if config.env_type == "harbor":
        from lfx.envs.harbor import HarborAdapter, HarborTaskEnvironment

        envs = [
            HarborTaskEnvironment(
                task_dir=Path(d),
                trial_config=config.harbor.trial_config,
                train_on_truncated=config.harbor.train_on_truncated,
            )
            for d in config.harbor.task_dirs
        ]
        adapter = HarborAdapter(envs)
        tasks = [env.task_id for env in envs]

    elif config.env_type == "math":
        from lfx.envs.math import MathAdapter, MathEnvironment

        task_client = _make_llm_client(config.llm_clients["task"])
        math_env = MathEnvironment()
        adapter = MathAdapter(env=math_env, client=task_client)
        tasks = [s.question for s in math_env.get_tasks()]

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
