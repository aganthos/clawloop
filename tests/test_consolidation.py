"""Tests for PlaybookCurator.consolidate() — dreaming pass."""

import time

from clawloop.core.curator import (
    ConsolidationReport,
    CuratorConfig,
    PlaybookCurator,
)
from clawloop.core.embeddings import MockEmbedding
from clawloop.learning_layers.harness import Playbook, PlaybookEntry
from clawloop.llm import MockLLMClient


def _entry(
    content: str,
    *,
    entry_id: str | None = None,
    helpful: int = 1,
    harmful: int = 0,
) -> PlaybookEntry:
    return PlaybookEntry(
        id=entry_id or PlaybookEntry.new_id(prefix="t"),
        content=content,
        helpful=helpful,
        harmful=harmful,
        created_at=time.time(),
        last_activated=time.time(),
    )


class TestConsolidation:
    """PlaybookCurator.consolidate() clustering, merging, pruning, and capping."""

    def test_consolidation_merges_similar_entries(self) -> None:
        """Entries with identical content (cosine sim=1.0) are merged."""
        embeddings = MockEmbedding()
        llm = MockLLMClient(responses=["merged insight"])
        config = CuratorConfig(cluster_threshold=0.7)
        curator = PlaybookCurator(embeddings, llm, config)

        playbook = Playbook(entries=[
            _entry("always use structured logging", entry_id="e1"),
            _entry("always use structured logging", entry_id="e2"),
        ])

        report = curator.consolidate(playbook)

        # Two entries merged into one new entry; originals superseded.
        assert report.merged == 2
        active = playbook.active_entries()
        assert len(active) < 2  # merged cluster reduced count

    def test_consolidation_prunes_negative_score(self) -> None:
        """Entries with harmful > helpful (effective_score < 0) are pruned."""
        embeddings = MockEmbedding()
        llm = MockLLMClient(responses=["merged"])
        curator = PlaybookCurator(embeddings, llm)

        playbook = Playbook(entries=[
            _entry("good tip", entry_id="e1", helpful=5, harmful=0),
            _entry("bad tip", entry_id="e2", helpful=0, harmful=3),
        ])

        report = curator.consolidate(playbook)

        assert report.pruned >= 1
        remaining_ids = [e.id for e in playbook.entries]
        assert "e2" not in remaining_ids

    def test_consolidation_caps_at_max_entries(self) -> None:
        """Playbook is capped at max_playbook_entries after consolidation."""
        embeddings = MockEmbedding()
        llm = MockLLMClient(responses=["merged"])
        config = CuratorConfig(
            max_playbook_entries=3,
            cluster_threshold=0.99,  # high threshold so nothing clusters
        )
        curator = PlaybookCurator(embeddings, llm, config)

        playbook = Playbook(entries=[
            _entry(f"unique tip number {i}", entry_id=f"e{i}")
            for i in range(5)
        ])

        report = curator.consolidate(playbook)

        assert report.after <= 3

    def test_consolidation_report(self) -> None:
        """ConsolidationReport fields are correctly populated."""
        embeddings = MockEmbedding()
        llm = MockLLMClient(responses=["merged"])
        config = CuratorConfig(cluster_threshold=0.99)
        curator = PlaybookCurator(embeddings, llm, config)

        playbook = Playbook(entries=[
            _entry("tip A", entry_id="eA"),
            _entry("tip B", entry_id="eB", helpful=0, harmful=5),
        ])

        report = curator.consolidate(playbook)

        assert isinstance(report, ConsolidationReport)
        assert report.before == 2
        assert report.after <= report.before
        assert isinstance(report.merged, int)
        assert isinstance(report.pruned, int)
        assert isinstance(report.conflicts_resolved, int)

    def test_singleton_clusters_preserved(self) -> None:
        """Dissimilar entries stay as independent singletons (no merge)."""
        embeddings = MockEmbedding()
        llm = MockLLMClient(responses=["merged"])
        config = CuratorConfig(cluster_threshold=0.99)
        curator = PlaybookCurator(embeddings, llm, config)

        playbook = Playbook(entries=[
            _entry("handle network timeouts gracefully", entry_id="e1"),
            _entry("always validate user input against schema", entry_id="e2"),
            _entry("prefer batch operations over single-item", entry_id="e3"),
        ])

        report = curator.consolidate(playbook)

        assert report.merged == 0
        active = playbook.active_entries()
        assert len(active) == 3
