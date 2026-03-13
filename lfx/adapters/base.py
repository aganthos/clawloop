"""EnvAdapter ABC — the interface every benchmark adapter must implement.

An adapter bridges a specific benchmark environment and the LfX learning loop.
It is responsible for:
  1. Setting up the environment (loading tasks, connecting to servers, etc.).
  2. Running one episode and returning an ``Episode`` object.
  3. Providing raw traces for observability export.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from lfx.core.episode import Episode

if TYPE_CHECKING:
    from lfx.core.loop import AgentState


class EnvAdapter(ABC):
    """Abstract base for benchmark environment adapters."""

    @abstractmethod
    def setup(self, config: dict[str, Any]) -> None:
        """One-time initialization (connect, load data, etc.)."""
        ...

    @abstractmethod
    def run_episode(self, task: Any, agent_state: AgentState) -> Episode:
        """Execute a single task and return the resulting Episode."""
        ...

    @abstractmethod
    def get_traces(self, episode: Episode) -> dict[str, Any]:
        """Return raw traces for an episode (for observability export)."""
        ...

    @abstractmethod
    def list_tasks(self, split: str = "base") -> list[Any]:
        """Return available tasks for a given split."""
        ...

    def run_batch(
        self, agent_state: "AgentState", task_ids: list[Any]
    ) -> list[Episode]:
        """Run a batch of tasks. Default falls back to sequential run_episode."""
        return [self.run_episode(task_id, agent_state) for task_id in task_ids]
