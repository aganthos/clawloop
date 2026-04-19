"""ArchiveStore protocol — interface for evolution archive backends."""

from __future__ import annotations

from typing import Protocol

from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)


class ArchiveStore(Protocol):
    """Backend-agnostic interface for persisting evolution archive data."""

    def log_run_start(self, run: RunRecord) -> None: ...

    def log_iteration(self, iteration: IterationRecord) -> None: ...

    def log_episodes(self, episodes: list[EpisodeRecord]) -> None: ...

    def log_variant(self, variant: AgentVariant) -> None: ...

    def log_run_complete(
        self,
        run_id: str,
        best_reward: float,
        improvement_delta: float,
        total_cost_tokens: int = 0,
    ) -> None: ...

    def get_run(self, run_id: str) -> RunRecord | None: ...

    def get_similar_runs(
        self,
        config_hash: str,
        domain_tags: list[str],
        limit: int = 10,
    ) -> list[RunRecord]: ...
