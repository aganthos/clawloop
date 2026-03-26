"""Arithmetic environment for SkyRL gym (Tinker-compatible).

Single-turn: model receives "What is X + Y?", must answer with \\boxed{Z}.
Reward: 1.0 correct, 0.5 boxed but wrong, 0.0 no boxed answer.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


class ArithmeticEnv(BaseTextEnv):
    """Tinker-compatible arithmetic environment."""

    def __init__(self, env_config: Dict[str, Any] | None = None, extras: Dict[str, Any] | None = None):
        super().__init__()
        extras = extras or {}
        assert "reward_spec" in extras, "reward_spec required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth required"
        self.ground_truth = str(extras["reward_spec"]["ground_truth"])

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        m = re.search(r"\\boxed\{([^}]+)\}", action)
        answer = m.group(1).strip() if m else None

        if answer is not None and answer == self.ground_truth:
            return BaseTextEnvStepOutput(observations=[], reward=1.0, done=True,
                                         metadata={"answer": answer})
        elif answer is not None:
            return BaseTextEnvStepOutput(observations=[], reward=0.5, done=True,
                                         metadata={"answer": answer})
        else:
            return BaseTextEnvStepOutput(observations=[], reward=0.0, done=True,
                                         metadata={"answer": None})
