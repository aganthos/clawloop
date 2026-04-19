"""Tests for PlaybookCurator in lightweight mode (no embeddings, no LLM).

For narrow agents (e.g. n8n workflows) where playbooks are small and
embedding infrastructure is overkill, the curator should still provide
pruning, capping, and metrics without requiring external services.
"""

from clawloop.core.curator import CuratorConfig, PlaybookCurator
from clawloop.learning_layers.harness import Insight, Playbook, PlaybookEntry


class TestLightweightCurator:
    """Curator with embeddings=None, llm=None."""

    def _curator(self, **kwargs) -> PlaybookCurator:
        return PlaybookCurator(embeddings=None, llm=None, config=CuratorConfig(**kwargs))

    def test_curate_adds_directly(self) -> None:
        curator = self._curator()
        pb = Playbook()
        insight = Insight(content="Always verify inputs", tags=["validation"])
        result = curator.curate_insight(insight, pb)
        assert result.action == "add"
        assert len(pb.entries) == 1
        assert pb.entries[0].content == "Always verify inputs"
        assert curator.metrics.added == 1

    def test_multiple_insights_all_added(self) -> None:
        curator = self._curator()
        pb = Playbook()
        for i in range(5):
            curator.curate_insight(
                Insight(content=f"Strategy {i}"),
                pb,
            )
        assert len(pb.entries) == 5
        assert curator.metrics.insights_processed == 5
        assert curator.metrics.added == 5

    def test_consolidate_prunes_negative_score(self) -> None:
        curator = self._curator()
        pb = Playbook()
        pb.add(PlaybookEntry(id="good", content="good tip", helpful=5, harmful=0))
        pb.add(PlaybookEntry(id="bad", content="bad tip", helpful=0, harmful=3))
        report = curator.consolidate(pb)
        assert report.pruned > 0
        assert pb.lookup("good") is not None
        assert pb.lookup("bad") is None

    def test_consolidate_caps_at_max(self) -> None:
        curator = self._curator(max_playbook_entries=3)
        pb = Playbook()
        for i in range(6):
            pb.add(
                PlaybookEntry(
                    id=f"e{i}",
                    content=f"tip {i}",
                    helpful=i,
                    harmful=0,
                )
            )
        curator.consolidate(pb)
        # Should keep top 3 by effective_score
        active = pb.active_entries()
        assert len(active) <= 3
        # The highest-helpful entries should survive
        surviving_ids = {e.id for e in active}
        assert "e5" in surviving_ids
        assert "e4" in surviving_ids

    def test_consolidate_skips_clustering_without_embeddings(self) -> None:
        curator = self._curator()
        pb = Playbook()
        pb.add(PlaybookEntry(id="a", content="tip a", helpful=2))
        pb.add(PlaybookEntry(id="b", content="tip b", helpful=2))
        report = curator.consolidate(pb)
        # No merging without embeddings
        assert report.merged == 0
        assert len(pb.entries) == 2

    def test_metrics_tracked(self) -> None:
        curator = self._curator()
        pb = Playbook()
        curator.curate_insight(Insight(content="tip 1"), pb)
        curator.curate_insight(Insight(content="tip 2"), pb)
        curator.consolidate(pb)
        assert curator.metrics.insights_processed == 2
        assert curator.metrics.added == 2
        assert curator.metrics.consolidation_runs == 1

    def test_no_crash_on_coherence_check_without_llm(self) -> None:
        curator = self._curator()
        pb = Playbook()
        pb.add(PlaybookEntry(id="e1", content="be concise"))
        # Should return empty list, not crash
        conflicts = curator.check_prompt_playbook_coherence("Be verbose", pb)
        assert conflicts == []
