"""Tests for support-query data separation in the learning loop.

The learning loop splits episodes by effective_reward():
  - reward < 0  -> harness only (support set: learn from failures)
  - reward >= 0 -> weights only (query set: optimize from successes)
  - all         -> router (needs both signals)

EpisodeSummary.total_reward setter maps [0,1] to [-1,1]:
  mapped = value * 2 - 1
So total_reward=0.2 -> effective_reward=-0.6 (failure),
   total_reward=0.8 -> effective_reward=0.6  (success).
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
    """Monkey-patch forward_backward on *layer* to record incoming Datums.

    Returns the list that will be populated with each Datum passed to the layer.
    """
    captured: list[Datum] = []
    original_fb = layer.forward_backward

    def capturing_fb(data: Datum) -> Future[FBResult]:
        captured.append(data)
        return original_fb(data)

    layer.forward_backward = capturing_fb
    return captured


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSupportQuerySeparation:

    def test_failures_go_to_harness_only(self):
        """Episodes with reward=0.2 (effective=-0.6, failure) should only
        reach harness.forward_backward, not weights."""
        failure_ep = _make_episode(task_id="f1", reward=0.2)
        adapter = _CapturingAdapter([failure_ep])
        state = AgentState()

        harness_data = _patch_layer_fb(state.harness)
        weights_data = _patch_layer_fb(state.weights)

        learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
        )

        # Harness should have received the failure episode
        assert len(harness_data) == 1
        assert len(harness_data[0].episodes) == 1
        assert harness_data[0].episodes[0].task_id == "f1"

        # Weights should have received zero episodes (empty query set)
        assert len(weights_data) == 1
        assert len(weights_data[0].episodes) == 0

    def test_successes_go_to_weights_only(self):
        """Episodes with reward=0.8 (effective=0.6, success) should only
        reach weights.forward_backward, not harness."""
        success_ep = _make_episode(task_id="s1", reward=0.8)
        adapter = _CapturingAdapter([success_ep])
        state = AgentState()

        harness_data = _patch_layer_fb(state.harness)
        weights_data = _patch_layer_fb(state.weights)

        learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
        )

        # Harness should have received zero episodes (empty support set)
        assert len(harness_data) == 1
        assert len(harness_data[0].episodes) == 0

        # Weights should have received the success episode
        assert len(weights_data) == 1
        assert len(weights_data[0].episodes) == 1
        assert weights_data[0].episodes[0].task_id == "s1"

    def test_router_gets_all_episodes(self):
        """Router always gets all episodes regardless of reward."""
        failure_ep = _make_episode(task_id="f1", reward=0.2)
        success_ep = _make_episode(task_id="s1", reward=0.8)
        adapter = _CapturingAdapter([failure_ep, success_ep])
        state = AgentState()

        router_data = _patch_layer_fb(state.router)

        learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2"],
            n_episodes=2,
            n_iterations=1,
        )

        # Router should have received all episodes
        assert len(router_data) == 1
        assert len(router_data[0].episodes) == 2
        task_ids = {ep.task_id for ep in router_data[0].episodes}
        assert task_ids == {"f1", "s1"}

    def test_mixed_episodes_split_correctly(self):
        """A mix of failures and successes should be split correctly:
        failures -> harness, successes -> weights, all -> router."""
        failures = [_make_episode(task_id=f"fail_{i}", reward=0.2) for i in range(3)]
        successes = [_make_episode(task_id=f"succ_{i}", reward=0.8) for i in range(2)]
        all_eps = failures + successes
        adapter = _CapturingAdapter(all_eps)
        state = AgentState()

        harness_data = _patch_layer_fb(state.harness)
        router_data = _patch_layer_fb(state.router)
        weights_data = _patch_layer_fb(state.weights)

        learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=[f"t{i}" for i in range(5)],
            n_episodes=5,
            n_iterations=1,
        )

        # Harness: only failures
        assert len(harness_data) == 1
        harness_ids = {ep.task_id for ep in harness_data[0].episodes}
        assert harness_ids == {"fail_0", "fail_1", "fail_2"}
        assert len(harness_data[0].episodes) == 3

        # Weights: only successes
        assert len(weights_data) == 1
        weights_ids = {ep.task_id for ep in weights_data[0].episodes}
        assert weights_ids == {"succ_0", "succ_1"}
        assert len(weights_data[0].episodes) == 2

        # Router: all episodes
        assert len(router_data) == 1
        router_ids = {ep.task_id for ep in router_data[0].episodes}
        assert router_ids == {"fail_0", "fail_1", "fail_2", "succ_0", "succ_1"}
        assert len(router_data[0].episodes) == 5
