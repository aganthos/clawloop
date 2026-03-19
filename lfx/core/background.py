"""Background job system — pluggable tasks with unified scheduling.

Provides a single ``BackgroundScheduler`` that runs ``BackgroundTask``
implementations when their conditions are met. Tasks include playbook
consolidation (from PR1's curator) and episode dreaming (cross-episode
pattern analysis).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from lfx.core.episode import Episode
from lfx.layers.harness import Insight, Playbook

log = logging.getLogger(__name__)


@dataclass
class BackgroundState:
    """Snapshot of system state passed to background tasks."""

    episodes_since_last_run: int
    time_since_last_run: float
    is_user_idle: bool
    playbook: Playbook
    recent_episodes: list[Episode]


@runtime_checkable
class BackgroundTask(Protocol):
    """Protocol for pluggable background tasks."""

    name: str

    def should_run(self, state: BackgroundState) -> bool:
        """Return True if this task should execute now."""
        ...

    def run(self, state: BackgroundState) -> None:
        """Execute the background task."""
        ...


@dataclass
class PlaybookConsolidation:
    """Background task that runs the curator's consolidation pipeline."""

    name: str = "playbook_consolidation"
    episode_threshold: int = 50
    min_interval: float = 300.0  # seconds
    curator: Any = None  # PlaybookCurator

    def should_run(self, state: BackgroundState) -> bool:
        if self.curator is None:
            return False
        return (
            state.episodes_since_last_run >= self.episode_threshold
            and state.time_since_last_run >= self.min_interval
            and state.is_user_idle
        )

    def run(self, state: BackgroundState) -> None:
        if self.curator is None:
            return
        try:
            report = self.curator.consolidate(state.playbook)
            log.info(
                "Consolidation: %d->%d entries (merged=%d, pruned=%d)",
                report.before, report.after, report.merged, report.pruned,
            )
        except Exception:
            log.exception("PlaybookConsolidation failed")


@dataclass
class EpisodeDreamer:
    """Cross-episode meta-pattern analysis.

    Analyzes recent episodes for recurring patterns and produces
    insights tagged with 'meta-pattern'.
    """

    name: str = "episode_dreamer"
    episode_threshold: int = 20
    min_interval: float = 600.0  # seconds
    llm: Any = None  # LLMClient

    def should_run(self, state: BackgroundState) -> bool:
        if self.llm is None:
            return False
        return (
            state.episodes_since_last_run >= self.episode_threshold
            and state.time_since_last_run >= self.min_interval
            and state.is_user_idle
        )

    def run(self, state: BackgroundState) -> list[Insight]:
        """Analyze cross-episode patterns and return insights."""
        if self.llm is None:
            return []

        episode_summaries = []
        for ep in state.recent_episodes[-self.episode_threshold :]:
            reward = ep.summary.effective_reward()
            task = ep.task_id
            msgs = len(ep.messages)
            episode_summaries.append(
                f"- Task={task} reward={reward:.2f} messages={msgs}"
            )

        if not episode_summaries:
            return []

        prompt = [
            {
                "role": "system",
                "content": (
                    "You are analyzing patterns across multiple agent episodes. "
                    "Identify recurring failure modes, successful strategies, or "
                    "behavioral patterns. Return a JSON array of insights, each with "
                    '"action" ("add" or "update"), "content" (the insight text), '
                    'and "tags" (list of strings, must include "meta-pattern").'
                ),
            },
            {
                "role": "user",
                "content": (
                    "## Recent Episodes\n"
                    + "\n".join(episode_summaries)
                    + "\n\n## Current Playbook Entries\n"
                    + "\n".join(
                        f"- {e.content[:100]}"
                        for e in state.playbook.active_entries()[:10]
                    )
                    + "\n\nWhat meta-patterns do you see across these episodes?"
                ),
            },
        ]

        try:
            import json

            response = self.llm.complete(prompt)
            text = response.text if hasattr(response, "text") else str(response)
            parsed = json.loads(text)
            insights = []
            for item in parsed:
                tags = item.get("tags", ["meta-pattern"])
                if "meta-pattern" not in tags:
                    tags.append("meta-pattern")
                insights.append(
                    Insight(
                        action=item.get("action", "add"),
                        content=item.get("content", ""),
                        tags=tags,
                    )
                )
            log.info("EpisodeDreamer produced %d insights", len(insights))
            return insights
        except Exception:
            log.exception("EpisodeDreamer failed")
            return []


class BackgroundScheduler:
    """Unified scheduler for periodic background work.

    Calls ``should_run`` on each registered task and runs those
    that are ready. Guards against overlapping runs.
    """

    def __init__(self, tasks: list[Any] | None = None) -> None:
        self.tasks: list[Any] = tasks or []
        self._last_run_times: dict[str, float] = {}
        self._episodes_since: dict[str, int] = {}
        self._in_progress: set[str] = set()

    def register(self, task: Any) -> None:
        """Add a task to the scheduler."""
        self.tasks.append(task)

    def record_episodes(self, count: int) -> None:
        """Record that episodes have been collected since last run."""
        for task in self.tasks:
            name = task.name
            self._episodes_since[name] = self._episodes_since.get(name, 0) + count

    def tick(self, playbook: Playbook, recent_episodes: list[Episode], is_user_idle: bool) -> None:
        """Check all tasks and run those that are ready."""
        now = time.time()
        for task in self.tasks:
            name = task.name
            if name in self._in_progress:
                continue

            last_run = self._last_run_times.get(name, 0.0)
            state = BackgroundState(
                episodes_since_last_run=self._episodes_since.get(name, 0),
                time_since_last_run=now - last_run if last_run > 0 else float("inf"),
                is_user_idle=is_user_idle,
                playbook=playbook,
                recent_episodes=recent_episodes,
            )

            if task.should_run(state):
                self._in_progress.add(name)
                try:
                    task.run(state)
                    self._last_run_times[name] = now
                    self._episodes_since[name] = 0
                except Exception:
                    log.exception("Background task %s failed", name)
                finally:
                    self._in_progress.discard(name)
