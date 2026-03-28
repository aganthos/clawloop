"""Tests for generation flush in the learning loop.

When harness.playbook_generation advances (structural playbook change from
insights), the loop flushes stale episodes from the Weights pending buffer
to prevent RL from learning pre-adaptation behavior.
"""

import logging

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.loop import AgentState, learning_loop
from clawloop.layers.harness import Harness


def _make_episode(
    task_id: str = "t1",
    reward: float = 0.8,
) -> Episode:
    return Episode(
        id=Episode.new_id(),
        state_id="deadbeef",
        task_id=task_id,
        bench="test",
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class _MockAdapter:
    """Adapter that returns canned episodes."""

    def __init__(self, reward: float = 0.8) -> None:
        self.reward = reward

    def run_episode(self, task, agent_state) -> Episode:
        return _make_episode(reward=self.reward, task_id=str(task))


class TestGenerationAdvanceFlushesWeightsBuffer:
    """When playbook_generation advances during optim_step, the loop must
    clear weights._pending.advantages so stale episodes are not used for RL.
    """

    def test_generation_advance_flushes_weights_buffer(self) -> None:
        adapter = _MockAdapter(reward=0.8)
        harness = Harness()
        state = AgentState(harness=harness)

        # Run one iteration to initialise _prev_playbook_generation
        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness"],
        )
        assert state._prev_playbook_generation == 0
        # Simulate: weights have stale pending advantages from a previous batch
        state.weights._pending.advantages = [
            ("ep-stale-1", 0.5),
            ("ep-stale-2", -0.3),
            ("ep-stale-3", 0.1),
        ]

        # Advance playbook_generation to simulate a structural change
        state.harness.playbook_generation = 1

        # Run another iteration (harness-only so weights optim doesn't drain
        # the buffer itself)
        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness"],
        )

        # The flush logic should have cleared the stale advantages
        assert state.weights._pending.advantages == [], (
            "Stale advantages should be flushed after playbook_generation advances"
        )
        # _prev_playbook_generation should now track the new generation
        assert state._prev_playbook_generation == 1

class TestNoFlushWhenGenerationUnchanged:
    """When playbook_generation stays the same, weights buffer is preserved."""

    def test_no_flush_when_generation_unchanged(self) -> None:
        adapter = _MockAdapter(reward=0.8)
        harness = Harness()
        state = AgentState(harness=harness)

        # Run one iteration to initialise _prev_playbook_generation
        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness"],
        )

        # Place advantages in the weights buffer
        stale_advantages = [("ep-1", 0.5), ("ep-2", -0.3)]
        state.weights._pending.advantages = list(stale_advantages)

        # Do NOT advance playbook_generation — it stays at 0
        assert state.harness.playbook_generation == 0

        # Run another iteration
        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness"],
        )

        # Buffer should be untouched — no flush because generation didn't change
        assert state.weights._pending.advantages == stale_advantages, (
            "Weights buffer should be preserved when playbook_generation is unchanged"
        )


class TestFlushLogsStaleCount:
    """The flush should log the number of stale episodes flushed."""

    def test_flush_logs_stale_count(self, caplog: pytest.LogCaptureFixture) -> None:
        adapter = _MockAdapter(reward=0.8)
        harness = Harness()
        state = AgentState(harness=harness)

        # Bootstrap _prev_playbook_generation
        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness"],
        )

        # Seed 5 stale advantages
        state.weights._pending.advantages = [
            (f"ep-stale-{i}", float(i) * 0.1) for i in range(5)
        ]

        # Advance generation
        state.harness.playbook_generation = 1

        with caplog.at_level(logging.INFO, logger="clawloop.core.loop"):
            state, _ = learning_loop(
                adapter=adapter,
                agent_state=state,
                tasks=["t1"],
                n_episodes=1,
                n_iterations=1,
                active_layers=["harness"],
            )

        # Find the flush log message
        flush_messages = [
            r.message for r in caplog.records
            if "flushed" in r.message and "stale" in r.message
        ]
        assert flush_messages, "Expected a log message about flushing stale episodes"

        # Verify the count in the message
        msg = flush_messages[0]
        assert "5" in msg, f"Expected stale count of 5 in log message, got: {msg}"
        assert "0->1" in msg or "Generation 0->1" in msg, (
            f"Expected generation transition 0->1 in log message, got: {msg}"
        )
