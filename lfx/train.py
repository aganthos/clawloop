"""Unified training entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, field_validator


class HarborConfig(BaseModel):
    """Harbor environment configuration."""

    task_dirs: list[str]
    trial_config: dict[str, Any] = {}
    train_on_truncated: bool = True

    model_config = {"arbitrary_types_allowed": True}


class TrainConfig(BaseModel):
    """Unified training configuration."""

    mode: str  # "weight" or "harness_learning"
    env_type: str = "harbor"

    harbor: HarborConfig | None = None
    skyrl: dict[str, Any] | None = None
    harness: dict[str, Any] | None = None

    system_prompt: str = "You are a helpful assistant."
    benches: list[str] = ["default"]
    episodes_per_iter: int = 10
    n_iterations: int = 100
    router: dict[str, Any] | None = None

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("weight", "harness_learning"):
            raise ValueError(f"mode must be 'weight' or 'harness_learning', got '{v}'")
        return v


def train(config: TrainConfig):
    """Unified training entry point.

    Builds layers, environments, and backend based on config.mode,
    then runs the learning loop.
    """
    from lfx.core.loop import AgentState, learning_loop
    from lfx.layers.harness import Harness
    from lfx.layers.router import Router
    from lfx.layers.weights import Weights

    if config.mode == "weight" and not config.skyrl:
        raise ValueError("mode='weight' requires 'skyrl' config")

    # 1. Always build harness and router
    harness = Harness(
        system_prompts={b: config.system_prompt for b in config.benches},
    )
    router = Router()

    # 2. Build backend based on mode
    backend = None
    if config.mode == "weight":
        from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig

        skyrl_cfg = SkyRLWeightsConfig(**config.skyrl)
        backend = SkyRLWeightsBackend(skyrl_cfg)
        weights = Weights(model_ref=skyrl_cfg.base_model, _backend=backend)
    else:
        weights = Weights()
    # HarnessLearningBackend available for future unified mode

    # 3. Build environments
    if not (config.env_type == "harbor" and config.harbor and config.harbor.task_dirs):
        raise ValueError("No environments configured")

    from lfx.envs.harbor import HarborAdapter, HarborTaskEnvironment

    envs = [
        HarborTaskEnvironment(
            task_dir=Path(d),
            trial_config=config.harbor.trial_config,
            train_on_truncated=config.harbor.train_on_truncated,
        )
        for d in config.harbor.task_dirs
    ]

    # 4. Build agent state
    agent_state = AgentState(
        harness=harness,
        router=router,
        weights=weights,
        inference_url=getattr(backend, "inference_url", None) if backend else None,
    )

    # 5. Run learning loop
    adapter = HarborAdapter(envs)
    tasks = [env.task_id for env in envs]

    return learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=tasks,
        n_episodes=config.episodes_per_iter,
        n_iterations=config.n_iterations,
    )
