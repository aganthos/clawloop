"""Shared test fixtures for enterprise SkyDiscover tests."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.evolver import EvolverContext, HarnessSnapshot
from clawloop.core.reward import RewardSignal


def make_episode(reward: float = 0.5) -> Episode:
    """Create a minimal Episode with a given outcome reward."""
    return Episode(
        id="ep-test",
        state_id="s-test",
        task_id="t-1",
        bench="test",
        messages=[Message(role="user", content="hello")],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(
            signals={"outcome": RewardSignal(name="outcome", value=reward, confidence=1.0)},
        ),
    )


def make_snapshot(
    system_prompt: str = "You are a helpful agent.",
    playbook_entries: list[dict[str, Any]] | None = None,
) -> HarnessSnapshot:
    """Create a HarnessSnapshot with sensible defaults."""
    entries = playbook_entries or [
        {
            "id": "e1",
            "content": "Always verify inputs.",
            "tags": ["safety"],
            "helpful": 5,
            "harmful": 1,
        },
    ]
    return HarnessSnapshot(
        system_prompts={"default": system_prompt},
        playbook_entries=entries,
        pareto_fronts={},
        playbook_generation=2,
        playbook_version=4,
    )


def make_context() -> EvolverContext:
    return EvolverContext(
        reward_history=[0.3, 0.5, 0.4],
        is_stagnating=False,
        iteration=3,
    )


def make_adapter(rewards: list[float] | None = None) -> MagicMock:
    """Create a mock adapter. If rewards given, sets side_effect."""
    adapter = MagicMock()
    if rewards is not None:
        episodes = [make_episode(r) for r in rewards]
        adapter.run_episode.side_effect = episodes
    return adapter


def make_factory() -> MagicMock:
    """Create a mock AgentStateFactory."""
    return MagicMock(return_value=MagicMock())
