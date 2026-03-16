"""Unified training entry point for LfX.

Supports two modes:
- ``weight``: GRPO fine-tuning via SkyRL backend
- ``harness_learning``: prompt/playbook evolution via HarnessLearningBackend
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, field_validator


class HarborConfig(BaseModel):
    task_dirs: list[str]
    trial_config: dict[str, Any] = {}
    reward_transform: Callable | None = None
    train_on_truncated: bool = True

    model_config = {"arbitrary_types_allowed": True}


class TrainConfig(BaseModel):
    mode: str  # "weight" or "harness_learning"
    env_type: str = "harbor"

    # Sub-configs
    harbor: HarborConfig | None = None
    skyrl: dict[str, Any] | None = None  # SkyRLWeightsConfig fields as dict
    harness: dict[str, Any] | None = None  # HarnessLearningConfig fields as dict

    # Learning loop params
    system_prompt: str = "You are a helpful assistant."
    benches: list[str] = ["default"]
    episodes_per_iter: int = 10
    n_iterations: int = 100

    # Router
    router: dict[str, Any] | None = None

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("weight", "harness_learning"):
            raise ValueError(f"mode must be 'weight' or 'harness_learning', got '{v}'")
        return v

    model_config = {"arbitrary_types_allowed": True}


def train(config: TrainConfig):
    """Unified training entry point."""
    from lfx.layers.harness import Harness
    from lfx.layers.router import Router
    from lfx.layers.weights import Weights
    from lfx.core.loop import AgentState, learning_loop

    # 1. Validate mode-specific config early (before any expensive construction)
    if config.mode == "weight" and not config.skyrl:
        raise ValueError("mode='weight' requires 'skyrl' config")

    # 2. Build harness and router (always active)
    harness = Harness(
        system_prompts={b: config.system_prompt for b in config.benches},
    )
    router = Router()

    # 3. Build environments
    envs = []
    if config.env_type == "harbor" and config.harbor:
        from lfx.envs.harbor import HarborTaskEnvironment
        envs = [
            HarborTaskEnvironment(
                task_dir=Path(d),
                trial_config=config.harbor.trial_config,
                reward_transform=config.harbor.reward_transform,
                train_on_truncated=config.harbor.train_on_truncated,
            )
            for d in config.harbor.task_dirs
        ]

    # 4. Build backend based on mode (skyrl presence already validated above)
    backend = None
    if config.mode == "weight":
        from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig
        skyrl_cfg = SkyRLWeightsConfig(**config.skyrl)
        backend = SkyRLWeightsBackend(skyrl_cfg)
        weights = Weights(model_ref=skyrl_cfg.base_model, _backend=backend)
    else:
        weights = Weights()
    # HarnessLearningBackend available for future unified mode

    # 5. Build agent state
    inference_url = backend.inference_url if backend else None
    agent_state_kwargs: dict[str, Any] = dict(
        harness=harness, router=router, weights=weights,
    )
    # Add inference_url only if AgentState supports it
    import dataclasses
    if any(f.name == "inference_url" for f in dataclasses.fields(AgentState)):
        agent_state_kwargs["inference_url"] = inference_url
    agent_state = AgentState(**agent_state_kwargs)

    # 6. Run learning loop
    if not envs:
        raise ValueError("No environments configured")

    from lfx.envs.harbor import HarborAdapter
    adapter = HarborAdapter(envs)
    tasks = [env.task_id for env in envs]

    return learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=tasks,
        n_episodes=config.episodes_per_iter,
        n_iterations=config.n_iterations,
    )
