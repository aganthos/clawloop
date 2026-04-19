"""Tests for stale episode skipping in Harness.forward_backward."""

from clawloop.core.episode import Episode, EpisodeSummary, Message
from clawloop.core.reward import RewardSignal
from clawloop.core.types import Datum
from clawloop.learning_layers.harness import Harness, Playbook, PlaybookEntry


def _ep_with_generation(
    reward: float,
    scored_at_generation: int | None,
) -> Episode:
    summary = EpisodeSummary()
    summary.signals["outcome"] = RewardSignal("outcome", reward, 1.0)
    summary.scored_at_generation = scored_at_generation
    return Episode(
        id="ep-1",
        state_id="s1",
        task_id="t1",
        bench="test",
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 20),
        ],
        step_boundaries=[0],
        steps=[],
        summary=summary,
    )


class TestStalenessSkipping:
    def test_stale_episode_skipped(self) -> None:
        h = Harness()
        h.playbook_generation = 5
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])

        ep = _ep_with_generation(reward=0.8, scored_at_generation=3)
        result = h.forward_backward(Datum(episodes=[ep])).result()

        assert result.metrics["stale_skipped"] == 1
        # No signals should have been accumulated
        assert not h._pending.playbook_signals

    def test_current_episode_not_skipped(self) -> None:
        h = Harness()
        h.playbook_generation = 5
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])

        ep = _ep_with_generation(reward=0.8, scored_at_generation=5)
        result = h.forward_backward(Datum(episodes=[ep])).result()

        assert result.metrics["stale_skipped"] == 0
        assert h._pending.playbook_signals

    def test_none_generation_not_skipped(self) -> None:
        """Episodes without scored_at_generation are always processed."""
        h = Harness()
        h.playbook_generation = 5
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])

        ep = _ep_with_generation(reward=0.8, scored_at_generation=None)
        result = h.forward_backward(Datum(episodes=[ep])).result()

        assert result.metrics["stale_skipped"] == 0
        assert h._pending.playbook_signals

    def test_mixed_stale_and_current(self) -> None:
        h = Harness()
        h.playbook_generation = 3
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])

        episodes = [
            _ep_with_generation(reward=0.8, scored_at_generation=1),  # stale
            _ep_with_generation(reward=0.8, scored_at_generation=3),  # current
            _ep_with_generation(reward=0.8, scored_at_generation=None),  # no gen
        ]
        result = h.forward_backward(Datum(episodes=episodes)).result()

        assert result.metrics["stale_skipped"] == 1
        assert result.metrics["episodes_processed"] == 3
