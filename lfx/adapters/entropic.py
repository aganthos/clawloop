"""Entropic CRMArenaPro adapter — A2A Purple agent.

Task-implicit tools, receives task context as TextPart.  7-dimension scoring:
  FUNCTIONAL 30%, DRIFT_ADAPTATION 20%, TOKEN_EFFICIENCY 12%,
  QUERY_EFFICIENCY 12%, ERROR_RECOVERY 8%, TRAJECTORY_EFFICIENCY 10%,
  HALLUCINATION_RATE 8%.

Drift/rot hardcoded to "medium".  2140 tasks.  a2a-sdk >= 0.3.20.
Agent-card path: ``/.well-known/agent.json``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lfx.adapters.base import EnvAdapter
from lfx.core.episode import Episode

if TYPE_CHECKING:
    from lfx.core.loop import AgentState

# Dimension weights for the 7-dim scoring rubric
SCORE_WEIGHTS: dict[str, float] = {
    "functional": 0.30,
    "drift_adaptation": 0.20,
    "token_efficiency": 0.12,
    "query_efficiency": 0.12,
    "error_recovery": 0.08,
    "trajectory_efficiency": 0.10,
    "hallucination_rate": 0.08,
}


class EntropicAdapter(EnvAdapter):
    """Adapter for Entropic CRMArenaPro benchmark (stub)."""

    def setup(self, config: dict[str, Any]) -> None:
        # TODO: connect to A2A server, discover agent card at
        # /.well-known/agent.json, validate a2a-sdk version
        self._config = config

    def run_episode(self, task: Any, agent_state: AgentState) -> Episode:
        raise NotImplementedError("Entropic adapter not yet implemented")

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"bench": "entropic", "episode_id": episode.id}

    def list_tasks(self, split: str = "base") -> list[Any]:
        raise NotImplementedError("Entropic adapter not yet implemented")
