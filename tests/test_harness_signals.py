"""Tests for Harness layer consuming reward signals instead of total_reward."""

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.reward import RewardSignal
from lfx.core.types import Datum
from lfx.layers.harness import Harness, Playbook, PlaybookEntry


def _ep_with_signal(name: str, value: float, confidence: float = 1.0) -> Episode:
    summary = EpisodeSummary()
    summary.signals[name] = RewardSignal(name, value, confidence)
    return Episode(
        id="ep-1", state_id="s1", task_id="t1", bench="test",
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 20),
        ],
        step_boundaries=[0], steps=[],
        summary=summary,
    )


class TestHarnessSignals:
    def test_positive_signal_increments_helpful(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        datum = Datum(episodes=[_ep_with_signal("outcome", 0.8)])
        h.forward_backward(datum)
        h.optim_step()
        assert h.playbook.entries[0].helpful == 1
        assert h.playbook.entries[0].harmful == 0

    def test_negative_signal_increments_harmful(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        datum = Datum(episodes=[_ep_with_signal("outcome", -0.5)])
        h.forward_backward(datum)
        h.optim_step()
        assert h.playbook.entries[0].helpful == 0
        assert h.playbook.entries[0].harmful == 1

    def test_neutral_signal_skipped(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        datum = Datum(episodes=[_ep_with_signal("execution", 0.0, confidence=0.3)])
        h.forward_backward(datum)
        h.optim_step()
        assert h.playbook.entries[0].helpful == 0
        assert h.playbook.entries[0].harmful == 0

    def test_user_signal_overrides(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        summary = EpisodeSummary()
        summary.signals["outcome"] = RewardSignal("outcome", 1.0, 1.0)
        summary.signals["user"] = RewardSignal("user", -1.0, 1.0)
        ep = Episode(
            id="ep-1", state_id="s1", task_id="t1", bench="test",
            messages=[
                Message(role="user", content="q"),
                Message(role="assistant", content="a" * 20),
            ],
            step_boundaries=[0], steps=[],
            summary=summary,
        )
        datum = Datum(episodes=[ep])
        h.forward_backward(datum)
        h.optim_step()
        assert h.playbook.entries[0].harmful == 1
        assert h.playbook.entries[0].helpful == 0
