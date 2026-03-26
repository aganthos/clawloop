"""Tests for data routing in the learning loop.

The support-query split is disabled (see loop.py).
All layers now receive all episodes. These tests verify the current behavior.
"""

import pytest

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.loop import AgentState, learning_loop
from lfx.core.types import Datum, FBResult, Future


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(task_id="t1", reward=0.8):
    return Episode(
        id=Episode.new_id(), state_id="deadbeef", task_id=task_id, bench="test",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class _CapturingAdapter:
    """Adapter that returns pre-built episodes in order."""

    def __init__(self, episodes):
        self._episodes = list(episodes)
        self._idx = 0

    def run_episode(self, task, agent_state):
        ep = self._episodes[self._idx % len(self._episodes)]
        self._idx += 1
        return ep


def _patch_layer_fb(layer):
    """Monkey-patch forward_backward on *layer* to record incoming Datums."""
    captured: list[Datum] = []
    original_fb = layer.forward_backward

    def capturing_fb(data: Datum) -> Future[FBResult]:
        captured.append(data)
        return original_fb(data)

    layer.forward_backward = capturing_fb
    return captured


# ---------------------------------------------------------------------------
# Tests — all layers get all episodes (split disabled)
# ---------------------------------------------------------------------------

class TestAllLayersGetAllEpisodes:

    def test_harness_gets_all_episodes(self):
        failure_ep = _make_episode(task_id="f1", reward=0.2)
        success_ep = _make_episode(task_id="s1", reward=0.8)
        adapter = _CapturingAdapter([failure_ep, success_ep])
        state = AgentState()

        harness_data = _patch_layer_fb(state.harness)

        learning_loop(
            adapter=adapter, agent_state=state,
            tasks=["t1", "t2"], n_episodes=2, n_iterations=1,
        )

        assert len(harness_data) == 1
        assert len(harness_data[0].episodes) == 2

    def test_weights_gets_all_episodes(self):
        failure_ep = _make_episode(task_id="f1", reward=0.2)
        success_ep = _make_episode(task_id="s1", reward=0.8)
        adapter = _CapturingAdapter([failure_ep, success_ep])
        state = AgentState()

        weights_data = _patch_layer_fb(state.weights)

        learning_loop(
            adapter=adapter, agent_state=state,
            tasks=["t1", "t2"], n_episodes=2, n_iterations=1,
        )

        assert len(weights_data) == 1
        assert len(weights_data[0].episodes) == 2

    def test_router_gets_all_episodes(self):
        failure_ep = _make_episode(task_id="f1", reward=0.2)
        success_ep = _make_episode(task_id="s1", reward=0.8)
        adapter = _CapturingAdapter([failure_ep, success_ep])
        state = AgentState()

        router_data = _patch_layer_fb(state.router)

        learning_loop(
            adapter=adapter, agent_state=state,
            tasks=["t1", "t2"], n_episodes=2, n_iterations=1,
        )

        assert len(router_data) == 1
        assert len(router_data[0].episodes) == 2
        task_ids = {ep.task_id for ep in router_data[0].episodes}
        assert task_ids == {"f1", "s1"}
