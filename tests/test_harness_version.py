"""Tests for Harness playbook_version counter."""

from clawloop.core.episode import Episode, EpisodeSummary, Message
from clawloop.core.reward import RewardSignal
from clawloop.core.types import Datum
from clawloop.learning_layers.harness import Harness, Insight, Playbook, PlaybookEntry


def _ep_positive() -> Episode:
    summary = EpisodeSummary()
    summary.signals["outcome"] = RewardSignal("outcome", 1.0, 1.0)
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


class TestHarnessVersion:
    def test_initial_version_is_zero(self) -> None:
        h = Harness()
        assert h.playbook_version == 0

    def test_optim_step_increments_version(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        datum = Datum(episodes=[_ep_positive()])
        h.forward_backward(datum)
        h.optim_step()
        assert h.playbook_version == 1

    def test_noop_optim_does_not_increment(self) -> None:
        h = Harness()
        # No forward_backward — nothing pending
        h.optim_step()
        assert h.playbook_version == 0

    def test_version_increments_on_insights_only(self) -> None:
        h = Harness()
        # No playbook entries, but pending insights
        h._pending.insights.append(Insight(action="add", content="new tip"))
        h.optim_step()
        assert h.playbook_version == 1

    def test_version_in_to_dict(self) -> None:
        h = Harness()
        h.playbook_version = 3
        d = h.to_dict()
        assert d["playbook_version"] == 3

    def test_version_survives_load_state(self) -> None:
        h = Harness()
        state = h.to_dict()
        state["playbook_version"] = 7

        h2 = Harness()
        h2.load_state(state)
        assert h2.playbook_version == 7

    def test_load_state_defaults_version_to_zero(self) -> None:
        h = Harness()
        # state dict without playbook_version key (old serialized state)
        state = h.to_dict()
        del state["playbook_version"]

        h2 = Harness()
        h2.playbook_version = 5  # pre-set to non-zero
        h2.load_state(state)
        assert h2.playbook_version == 0
