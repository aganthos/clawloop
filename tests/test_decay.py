"""Tests for PlaybookEntry.effective_score() temporal decay and needs_reembed()."""

import time

from lfx.layers.harness import PlaybookEntry


class TestEffectiveScoreDecay:
    """Temporal decay on PlaybookEntry.effective_score()."""

    def test_fresh_entry_no_decay(self) -> None:
        """Entry created just now should have effective_score very close to raw score."""
        entry = PlaybookEntry(id="e1", content="tip", helpful=5, harmful=1)
        raw = entry.score()
        effective = entry.effective_score()
        assert raw == 4.0
        # Freshly created — decay factor should be ~1.0
        assert abs(effective - raw) < 0.01

    def test_effective_score_decreases_with_age(self) -> None:
        """Entry created 30 days ago should have a lower effective_score than raw."""
        now = time.time()
        entry = PlaybookEntry(
            id="e1", content="tip", helpful=10, harmful=2,
            created_at=now - 30 * 86400,
            last_activated=now - 30 * 86400,
        )
        raw = entry.score()
        effective = entry.effective_score()
        assert raw == 8.0
        assert effective < raw
        assert effective > 0.0  # shouldn't have decayed to zero yet

    def test_last_activated_resets_decay(self) -> None:
        """Entry with old created_at but recent last_activated should decay less."""
        now = time.time()
        old_entry = PlaybookEntry(
            id="e1", content="tip", helpful=5, harmful=0,
            created_at=now - 60 * 86400,
            last_activated=now - 60 * 86400,  # never activated — decays from created_at
        )
        recent_entry = PlaybookEntry(
            id="e2", content="tip", helpful=5, harmful=0,
            created_at=now - 60 * 86400,
            last_activated=now - 1 * 86400,  # activated yesterday
        )
        # Recent activation should produce a higher effective score
        assert recent_entry.effective_score() > old_entry.effective_score()

    def test_never_used_decays_from_created_at(self) -> None:
        """Entry where last_activated == created_at decays from created_at."""
        now = time.time()
        days_old = 20
        ts = now - days_old * 86400
        entry = PlaybookEntry(
            id="e1", content="tip", helpful=4, harmful=0,
            created_at=ts,
            last_activated=ts,  # never separately activated
        )
        raw = entry.score()
        effective = entry.effective_score()
        assert raw == 4.0
        # Should have decayed over 20 days
        assert effective < raw
        # Verify the decay factor is roughly exp(-0.01 * 20) = exp(-0.2) ≈ 0.818
        import math
        expected = raw * math.exp(-entry.decay_rate * days_old)
        assert abs(effective - expected) < 0.01

    def test_zero_score_remains_zero(self) -> None:
        """0 helpful, 0 harmful -> effective_score stays 0 regardless of age."""
        now = time.time()
        entry = PlaybookEntry(
            id="e1", content="tip", helpful=0, harmful=0,
            created_at=now - 100 * 86400,
            last_activated=now - 100 * 86400,
        )
        assert entry.score() == 0.0
        assert entry.effective_score() == 0.0

    def test_custom_decay_rate(self) -> None:
        """Higher decay_rate should produce faster decay (lower effective_score)."""
        now = time.time()
        ts = now - 10 * 86400
        slow = PlaybookEntry(
            id="e1", content="tip", helpful=10, harmful=0,
            created_at=ts, last_activated=ts,
            decay_rate=0.01,
        )
        fast = PlaybookEntry(
            id="e2", content="tip", helpful=10, harmful=0,
            created_at=ts, last_activated=ts,
            decay_rate=0.1,
        )
        assert slow.effective_score() > fast.effective_score()


class TestNeedsReembed:
    """PlaybookEntry.needs_reembed() checks."""

    def test_needs_reembed_no_embedding(self) -> None:
        """needs_reembed returns True when embedding is None."""
        entry = PlaybookEntry(id="e1", content="tip", embedding=None)
        assert entry.needs_reembed("text-embedding-3-small") is True

    def test_needs_reembed_wrong_model(self) -> None:
        """needs_reembed returns True when model_id differs."""
        entry = PlaybookEntry(
            id="e1", content="tip",
            embedding=[0.1, 0.2, 0.3],
            embedding_model_id="old-model",
        )
        assert entry.needs_reembed("new-model") is True

    def test_needs_reembed_up_to_date(self) -> None:
        """needs_reembed returns False when model matches and embedding exists."""
        model = "text-embedding-3-small"
        entry = PlaybookEntry(
            id="e1", content="tip",
            embedding=[0.1, 0.2, 0.3],
            embedding_model_id=model,
        )
        assert entry.needs_reembed(model) is False
