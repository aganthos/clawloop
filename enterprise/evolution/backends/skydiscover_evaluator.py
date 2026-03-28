"""ClawLoop evaluator adapter for SkyDiscover.

SkyDiscover expects an evaluator callable:
    evaluate(program_path: str) -> {"combined_score": float, ...}

This module wraps ClawLoop's adapter + reward pipeline to provide that.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

from clawloop.core.episode import Episode

log = logging.getLogger(__name__)


class AdapterLike(Protocol):
    """Minimal protocol for what the evaluator needs from an adapter."""

    def run_episode(self, task: Any, agent_state: Any) -> Episode: ...


class AgentStateFactory(Protocol):
    """Creates an AgentState with a candidate system prompt + playbook."""

    def __call__(self, system_prompt: str, playbook: list[dict[str, Any]]) -> Any: ...


class ClawLoopEvaluator:
    """Evaluates a SkyDiscover "program" (agent config) via ClawLoop episodes.

    For each candidate program file, loads the config, runs episodes through
    the adapter, and returns the mean effective reward as combined_score.
    """

    def __init__(
        self,
        adapter: AdapterLike,
        tasks: list[Any],
        agent_state_factory: AgentStateFactory,
        n_episodes: int = 5,
    ) -> None:
        self._adapter = adapter
        self._tasks = tasks
        self._agent_state_factory = agent_state_factory
        self._n_episodes = n_episodes

    def __call__(self, program_path: str) -> dict[str, Any]:
        """Evaluate a candidate program. Returns {"combined_score": float}."""
        program = json.loads(Path(program_path).read_text())
        system_prompt = program.get("system_prompt", "")
        playbook = program.get("playbook", [])

        agent_state = self._agent_state_factory(system_prompt, playbook)

        # Run up to n_episodes, cycling through tasks
        episodes: list[Episode] = []
        for i in range(self._n_episodes):
            task = self._tasks[i % len(self._tasks)]
            try:
                ep = self._adapter.run_episode(task, agent_state)
                episodes.append(ep)
            except Exception:
                log.warning("Episode %d failed for program %s", i, program_path)

        if not episodes:
            return {"combined_score": -1.0, "n_episodes": 0}

        rewards = [ep.summary.effective_reward() for ep in episodes]
        mean_reward = sum(rewards) / len(rewards)

        return {
            "combined_score": mean_reward,
            "n_episodes": len(episodes),
            "rewards": rewards,
        }
