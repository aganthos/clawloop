"""tau2-bench adapter — Python API via LocalAgent subclass.

Uses the Python API directly (not a CLI wrapper).  Maps ``SimulationRun`` ->
``Episode``.  Reward is the product of all dimensions (sparse, binary-ish);
``reward_info.reward_breakdown`` provides per-dimension signals.

Domains: airline, retail.  Use ``"base"`` split for comparability.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from clawloop.adapters.base import EnvAdapter
from clawloop.core.episode import Episode

if TYPE_CHECKING:
    from clawloop.core.loop import AgentState


class Tau2Adapter(EnvAdapter):
    """Adapter for tau2-bench (stub).

    Intended to subclass ``tau2.agent.base.LocalAgent`` and map
    ``SimulationRun`` objects to LfX ``Episode`` instances.
    """

    def setup(self, config: dict[str, Any]) -> None:
        # TODO: import tau2, instantiate LocalAgent subclass,
        # load domain config (airline/retail)
        self._config = config

    def run_episode(self, task: Any, agent_state: AgentState) -> Episode:
        raise NotImplementedError("tau2-bench adapter not yet implemented")

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"bench": "tau2", "episode_id": episode.id}

    def list_tasks(self, split: str = "base") -> list[Any]:
        raise NotImplementedError("tau2-bench adapter not yet implemented")
