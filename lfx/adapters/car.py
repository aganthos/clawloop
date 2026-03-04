"""CAR-bench AgentBeats adapter — A2A Purple agent.

Explicit tool schemas via DataPart, tool_calls via DataPart, Green executes
tools.  254 tasks, 58 tools, 3 splits.

6 reward metrics:
  r_actions_final, r_actions_intermediate, r_tool_subset,
  r_tool_execution_errors, r_policy_errors, r_user_end_conversation.

a2a-sdk >= 0.3.5.  Agent-card path: ``/.well-known/agent-card.json``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lfx.adapters.base import EnvAdapter
from lfx.core.episode import Episode

if TYPE_CHECKING:
    from lfx.core.loop import AgentState

REWARD_METRICS = [
    "r_actions_final",
    "r_actions_intermediate",
    "r_tool_subset",
    "r_tool_execution_errors",
    "r_policy_errors",
    "r_user_end_conversation",
]


class CARAdapter(EnvAdapter):
    """Adapter for CAR-bench AgentBeats benchmark (stub)."""

    def setup(self, config: dict[str, Any]) -> None:
        # TODO: connect to A2A server, discover agent card at
        # /.well-known/agent-card.json, validate a2a-sdk version
        self._config = config

    def run_episode(self, task: Any, agent_state: AgentState) -> Episode:
        raise NotImplementedError("CAR adapter not yet implemented")

    def get_traces(self, episode: Episode) -> dict[str, Any]:
        return {"bench": "car", "episode_id": episode.id}

    def list_tasks(self, split: str = "base") -> list[Any]:
        raise NotImplementedError("CAR adapter not yet implemented")
