"""NullArchiveStore — no-op implementation for when archiving is disabled."""

from __future__ import annotations

from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)


class NullArchiveStore:
    """Drop-in archive store that silently discards all data."""

    def log_run_start(self, run: RunRecord) -> None:
        pass

    def log_iteration(self, iteration: IterationRecord) -> None:
        pass

    def log_episodes(self, episodes: list[EpisodeRecord]) -> None:
        pass

    def log_variant(self, variant: AgentVariant) -> None:
        pass

    def log_run_complete(
        self,
        run_id: str,
        best_reward: float,
        improvement_delta: float,
        total_cost_tokens: int = 0,
    ) -> None:
        pass

    def get_run(self, run_id: str) -> RunRecord | None:
        return None

    def get_similar_runs(
        self,
        config_hash: str,
        domain_tags: list[str],
        limit: int = 10,
    ) -> list[RunRecord]:
        return []
