"""Tests for entry-level attribution in Harness.forward_backward."""

import copy

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.reward import RewardSignal
from lfx.core.types import Datum
from lfx.layers.harness import Harness, Playbook, PlaybookEntry


def _ep_with_signal(
    name: str,
    value: float,
    confidence: float = 1.0,
    bench: str = "test",
    ep_id: str = "ep-1",
    scored_at_generation: int | None = None,
) -> Episode:
    summary = EpisodeSummary()
    summary.signals[name] = RewardSignal(name, value, confidence)
    if scored_at_generation is not None:
        summary.scored_at_generation = scored_at_generation
    return Episode(
        id=ep_id, state_id="s1", task_id="t1", bench=bench,
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 20),
        ],
        step_boundaries=[0], steps=[],
        summary=summary,
    )


class TestTagAttribution:
    """Tag-based entry attribution in forward_backward."""

    def test_tag_match_only_credits_relevant(self) -> None:
        """Entry tagged 'math' should only get credit from episodes with bench='math'."""
        h = Harness()
        math_entry = PlaybookEntry(id="e-math", content="use formulas", tags=["math"])
        code_entry = PlaybookEntry(id="e-code", content="write tests", tags=["code"])
        h.playbook = Playbook(entries=[math_entry, code_entry])

        # Episode with bench="math" — only e-math should be attributed
        ep = _ep_with_signal("outcome", 0.8, bench="math")
        datum = Datum(episodes=[ep])
        h.forward_backward(datum)
        h.optim_step()

        assert h.playbook.lookup("e-math").helpful == 1
        assert h.playbook.lookup("e-code").helpful == 0

    def test_untagged_entries_get_credit_as_fallback(self) -> None:
        """Entries with no tags get credit from all episodes when no tag match exists."""
        h = Harness()
        # No entry has tags matching bench="misc", so fallback to all active
        untagged = PlaybookEntry(id="e-gen", content="general tip")
        h.playbook = Playbook(entries=[untagged])

        ep = _ep_with_signal("outcome", 0.5, bench="misc")
        datum = Datum(episodes=[ep])
        h.forward_backward(datum)
        h.optim_step()

        assert h.playbook.lookup("e-gen").helpful == 1


class TestStaleAndNeutral:
    """Stale episode skipping and neutral reward filtering."""

    def test_stale_episode_skipped(self) -> None:
        """Episode with scored_at_generation < playbook_generation is skipped."""
        h = Harness()
        entry = PlaybookEntry(id="e1", content="tip")
        h.playbook = Playbook(entries=[entry])
        h.playbook_generation = 5

        # Episode scored at generation 3 — stale
        ep = _ep_with_signal("outcome", 1.0, scored_at_generation=3)
        datum = Datum(episodes=[ep])
        result = h.forward_backward(datum).result()

        assert result.metrics["stale_skipped"] == 1

        h.optim_step()
        assert h.playbook.lookup("e1").helpful == 0

    def test_neutral_reward_skipped(self) -> None:
        """Episode with reward=0 skips attribution entirely."""
        h = Harness()
        entry = PlaybookEntry(id="e1", content="tip")
        h.playbook = Playbook(entries=[entry])

        # execution signal with low confidence -> effective_reward() = 0.0
        ep = _ep_with_signal("execution", 0.5, confidence=0.3)
        datum = Datum(episodes=[ep])
        h.forward_backward(datum)
        h.optim_step()

        assert h.playbook.lookup("e1").helpful == 0
        assert h.playbook.lookup("e1").harmful == 0


class TestAttributionDeferral:
    """forward_backward must NOT mutate entries — mutations are deferred to optim_step."""

    def test_attribution_defers_last_activated(self) -> None:
        """forward_backward does NOT mutate entries; last_activated stays unchanged."""
        h = Harness()
        entry = PlaybookEntry(id="e1", content="tip")
        original_last_activated = entry.last_activated
        original_helpful = entry.helpful
        original_harmful = entry.harmful
        h.playbook = Playbook(entries=[entry])

        ep = _ep_with_signal("outcome", 0.9)
        datum = Datum(episodes=[ep])
        h.forward_backward(datum)

        # After forward_backward but BEFORE optim_step:
        # entry should be completely untouched
        assert entry.last_activated == original_last_activated
        assert entry.helpful == original_helpful
        assert entry.harmful == original_harmful

        # Only after optim_step do mutations take effect
        h.optim_step()
        assert entry.helpful == original_helpful + 1
