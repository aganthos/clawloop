"""Tests for PlaybookCurator retrieve-classify-revise pipeline."""

from lfx.core.curator import PlaybookCurator
from lfx.core.embeddings import MockEmbedding, cosine_similarity
from lfx.layers.harness import Insight, Playbook, PlaybookEntry
from lfx.llm import MockLLMClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMB = MockEmbedding()


def _perturbed_embedding(text: str, n_flip: int) -> list[float]:
    """Return a perturbed version of MockEmbedding's vector for *text*.

    Flips the sign of the first *n_flip* dimensions.  More flips -> lower
    cosine similarity.  Empirically (64-dim MockEmbedding):
        n_flip=7  -> sim ~ 0.82   (conflict range)
        n_flip=10 -> sim ~ 0.70   (LLM classification range)
    """
    vec = list(_EMB.embed([text])[0])
    for i in range(n_flip):
        vec[i] = -vec[i]
    return vec


def _make_entry(
    content: str,
    entry_id: str = "e-1",
    helpful: int = 0,
    harmful: int = 0,
    tags: list[str] | None = None,
    embedding: list[float] | None = None,
) -> PlaybookEntry:
    """Create a PlaybookEntry with a pre-computed embedding."""
    if embedding is None:
        embedding = _EMB.embed([content])[0]
    return PlaybookEntry(
        id=entry_id,
        content=content,
        helpful=helpful,
        harmful=harmful,
        tags=tags or [],
        embedding=embedding,
        embedding_model_id=_EMB.model,
    )


def _make_insight(
    content: str,
    tags: list[str] | None = None,
    source_episode_ids: list[str] | None = None,
) -> Insight:
    return Insight(
        content=content,
        tags=tags or [],
        source_episode_ids=source_episode_ids or [],
    )


class _FailingEmbedding:
    """Embedding provider that always raises."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding service unavailable")


class _FailingLLM:
    """LLM client that always raises."""

    def complete(self, messages, **kwargs):
        raise RuntimeError("LLM service unavailable")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCurateInsight:
    """Tests for the curate_insight pipeline."""

    def test_add_when_no_similar(self) -> None:
        """Insight with no similar existing entries results in action='add'."""
        llm = MockLLMClient(responses=["unused"])
        curator = PlaybookCurator(embeddings=_EMB, llm=llm)

        # Playbook has one entry about a completely different topic.
        playbook = Playbook(entries=[
            _make_entry("Use batch processing for large datasets", entry_id="e-1"),
        ])
        insight = _make_insight("Always greet users warmly")

        result = curator.curate_insight(insight, playbook)

        assert result.action == "add"
        assert result.new_entry is not None
        assert result.new_entry.content == "Always greet users warmly"
        # New entry should have been added to the playbook.
        assert any(
            e.content == "Always greet users warmly" for e in playbook.entries
        )
        assert curator.metrics.added == 1

    def test_skip_redundant_identical(self) -> None:
        """High cosine sim (>0.95) -> skip_redundant, existing entry's helpful bumped."""
        llm = MockLLMClient(responses=["unused"])
        curator = PlaybookCurator(embeddings=_EMB, llm=llm)

        content = "Always validate user input before processing"
        existing = _make_entry(content, entry_id="e-dup", helpful=3)
        playbook = Playbook(entries=[existing])

        # Same exact text -> MockEmbedding produces identical vector -> sim=1.0.
        insight = _make_insight(content)

        result = curator.curate_insight(insight, playbook)

        assert result.action == "skip_redundant"
        assert result.new_entry is None
        assert "e-dup" in result.entries_affected
        # helpful should have been bumped by 1.
        assert existing.helpful == 4
        assert curator.metrics.skipped_redundant == 1

    def test_classify_conflicting_heuristic(self) -> None:
        """High sim + contradiction keywords -> 'conflict_resolved', originals get superseded_by."""
        llm = MockLLMClient(responses=["Resolved: use context-appropriate communication"])
        curator = PlaybookCurator(embeddings=_EMB, llm=llm)

        # The entry and insight need sim > 0.8 (conflict_threshold) but < 0.95.
        # We control this by giving the entry the same embedding as a text
        # that is similar-but-different to the insight.
        #
        # The insight must have >=2 more contradiction keywords than the entry
        # to trigger the heuristic.
        entry_text = "Always use direct communication with the client"
        insight_text = "Never use direct communication, instead avoid direct approaches"

        # Force high similarity by giving the existing entry the insight's embedding.
        # This makes cosine_sim = 1.0 which is > 0.95 (identical threshold).
        # Instead, tweak the entry embedding slightly so sim lands in [0.8, 0.95).
        insight_emb = _EMB.embed([insight_text])[0]
        # Perturb: flip sign of a few dimensions to reduce sim.
        entry_emb = list(insight_emb)
        for i in range(0, len(entry_emb), 8):  # flip every 8th dimension
            entry_emb[i] = -entry_emb[i]
        # Verify the sim is in the right range.
        sim = cosine_similarity(entry_emb, insight_emb)
        assert 0.8 <= sim < 0.95, f"setup check: sim={sim}"

        existing = _make_entry(
            entry_text, entry_id="e-conflict", embedding=entry_emb,
        )
        playbook = Playbook(entries=[existing])

        insight = _make_insight(insight_text)
        result = curator.curate_insight(insight, playbook)

        assert result.action == "conflict_resolved"
        assert "e-conflict" in result.entries_affected
        assert result.new_entry is not None
        # Original should be superseded.
        assert existing.superseded_by == result.new_entry.id
        assert curator.metrics.conflicts_resolved == 1

    def test_classify_complementary_llm(self) -> None:
        """Medium sim (0.6-0.8) -> LLM returns 'complementary' -> action='merge'."""
        llm = MockLLMClient(responses=["complementary", "Merged: comprehensive tip"])
        curator = PlaybookCurator(embeddings=_EMB, llm=llm)

        # Need sim in [0.6, 0.8) so heuristic returns None and LLM is called.
        base_text = "Handle errors gracefully in API responses"
        insight_text = "Log all error details for debugging API issues"

        # Flip first 10 dims of the insight's embedding -> sim ~ 0.7.
        entry_emb = _perturbed_embedding(insight_text, n_flip=10)
        sim = cosine_similarity(entry_emb, _EMB.embed([insight_text])[0])
        assert 0.6 <= sim < 0.8, f"setup check: sim={sim}"

        existing = _make_entry(
            base_text, entry_id="e-comp", embedding=entry_emb,
        )
        playbook = Playbook(entries=[existing])

        insight = _make_insight(insight_text)
        result = curator.curate_insight(insight, playbook)

        assert result.action == "merge"
        assert "e-comp" in result.entries_affected
        assert result.new_entry is not None
        assert curator.metrics.merged == 1

    def test_classify_unrelated_llm(self) -> None:
        """Medium sim -> LLM returns 'unrelated' -> action='add'."""
        llm = MockLLMClient(responses=["unrelated"])
        curator = PlaybookCurator(embeddings=_EMB, llm=llm)

        # Flip first 9 dims -> sim ~ 0.64 for this text.
        insight_text = "Use structured logging for all services"
        entry_emb = _perturbed_embedding(insight_text, n_flip=9)
        sim = cosine_similarity(entry_emb, _EMB.embed([insight_text])[0])
        assert 0.6 <= sim < 0.8, f"setup check: sim={sim}"

        existing = _make_entry(
            "Cache frequently accessed data", entry_id="e-unrel",
            embedding=entry_emb,
        )
        playbook = Playbook(entries=[existing])

        insight = _make_insight(insight_text)
        result = curator.curate_insight(insight, playbook)

        assert result.action == "add"
        assert result.new_entry is not None
        assert result.new_entry.content == insight_text
        assert curator.metrics.added == 1


class TestCuratorFallbacks:
    """Tests for graceful degradation when embedding or LLM fails."""

    def test_fallback_on_embedding_failure(self) -> None:
        """Embedding raises -> fallback_direct_adds incremented, action='add'."""
        llm = MockLLMClient(responses=["unused"])
        failing_emb = _FailingEmbedding()
        curator = PlaybookCurator(embeddings=failing_emb, llm=llm)

        playbook = Playbook(entries=[
            _make_entry("Existing entry", entry_id="e-1"),
        ])
        insight = _make_insight("New insight despite embedding failure")

        result = curator.curate_insight(insight, playbook)

        assert result.action == "add"
        assert result.new_entry is not None
        assert curator.metrics.fallback_direct_adds == 1

    def test_fallback_on_llm_failure(self) -> None:
        """LLM raises during classification -> fallback, no crash."""
        failing_llm = _FailingLLM()
        curator = PlaybookCurator(embeddings=_EMB, llm=failing_llm)

        # Need sim in [0.6, 0.8) so heuristic is ambiguous and LLM is called.
        # Flip first 12 dims -> sim ~ 0.71.
        insight_text = "Always write tests before code"
        entry_emb = _perturbed_embedding(insight_text, n_flip=12)
        sim = cosine_similarity(entry_emb, _EMB.embed([insight_text])[0])
        assert 0.6 <= sim < 0.8, f"setup check: sim={sim}"

        existing = _make_entry(
            "Code quality matters", entry_id="e-llm-fail",
            embedding=entry_emb,
        )
        playbook = Playbook(entries=[existing])

        insight = _make_insight(insight_text)

        # Should not crash — falls back to direct add.
        result = curator.curate_insight(insight, playbook)

        assert result.action == "add"
        assert result.new_entry is not None
        assert curator.metrics.fallback_direct_adds == 1


class TestCuratorMetricsTracking:
    """Verify CuratorMetrics counters after multiple operations."""

    def test_metrics_tracking(self) -> None:
        """Run a sequence of operations and verify all counters."""
        llm = MockLLMClient(responses=[
            "complementary",     # classification for 2nd insight
            "Merged entry text", # merge result for 2nd insight
        ])
        curator = PlaybookCurator(embeddings=_EMB, llm=llm)
        playbook = Playbook()

        # 1. Add — no existing entries, so action=add.
        insight1 = _make_insight("First tip about error handling")
        r1 = curator.curate_insight(insight1, playbook)
        assert r1.action == "add"

        # 2. Skip redundant — same content as first.
        insight2 = _make_insight("First tip about error handling")
        r2 = curator.curate_insight(insight2, playbook)
        assert r2.action == "skip_redundant"

        # 3. Add — completely different topic.
        insight3 = _make_insight("Always use HTTPS for API calls")
        r3 = curator.curate_insight(insight3, playbook)
        assert r3.action == "add"

        # Verify counters.
        m = curator.metrics
        assert m.insights_processed == 3
        assert m.added == 2
        assert m.skipped_redundant == 1
        assert m.fallback_direct_adds == 0


class TestSupersededEntriesHiddenInRender:
    """After conflict resolution, superseded entries should be hidden from render()."""

    def test_superseded_entries_hidden_in_render(self) -> None:
        llm = MockLLMClient(responses=["Resolved: balanced communication approach"])
        curator = PlaybookCurator(embeddings=_EMB, llm=llm)

        entry_text = "Always use direct communication with the client"
        insight_text = "Never use direct communication, instead avoid direct approaches"

        # Force sim in [0.8, 0.95) for conflict heuristic.
        insight_emb = _EMB.embed([insight_text])[0]
        entry_emb = list(insight_emb)
        for i in range(0, len(entry_emb), 8):
            entry_emb[i] = -entry_emb[i]
        sim = cosine_similarity(entry_emb, insight_emb)
        assert 0.8 <= sim < 0.95, f"setup check: sim={sim}"

        existing = _make_entry(
            entry_text, entry_id="e-old", embedding=entry_emb,
        )
        playbook = Playbook(entries=[existing])

        insight = _make_insight(insight_text)
        result = curator.curate_insight(insight, playbook)
        assert result.action == "conflict_resolved"

        # The old entry should be superseded.
        assert existing.superseded_by is not None

        # render() should NOT include the superseded entry's content.
        rendered = playbook.render()
        assert entry_text not in rendered
        # But the resolved entry SHOULD appear.
        assert result.new_entry is not None
        assert result.new_entry.content in rendered
