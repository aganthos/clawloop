"""Tests for curator graceful degradation — fallback on embedding/LLM failures."""

from clawloop.core.curator import CurationResult, CuratorConfig, PlaybookCurator
from clawloop.core.embeddings import MockEmbedding
from clawloop.learning_layers.harness import Insight, Playbook, PlaybookEntry
from clawloop.llm import MockLLMClient


class _FailingEmbedding:
    """Embedding provider that always raises."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("Embedding service unavailable")


class _FailingLLM:
    """LLM client that always raises."""

    def complete(self, messages, **kwargs):
        raise RuntimeError("LLM service unavailable")


def _insight(content: str = "new insight") -> Insight:
    return Insight(
        content=content,
        source_episode_ids=["ep-1"],
        tags=["test"],
    )


class TestFallback:
    """Curator falls back gracefully when embedding or LLM services fail."""

    def test_curator_fallback_on_embedding_failure(self) -> None:
        """When embeddings.embed() raises, curate_insight falls back to direct add."""
        failing_embeddings = _FailingEmbedding()
        llm = MockLLMClient(responses=["mock"])
        curator = PlaybookCurator(failing_embeddings, llm)  # type: ignore[arg-type]

        playbook = Playbook()
        result = curator.curate_insight(_insight(), playbook)

        assert isinstance(result, CurationResult)
        assert result.action == "add"
        assert curator.metrics.fallback_direct_adds >= 1

    def test_curator_fallback_on_llm_failure(self) -> None:
        """When LLM raises during classification, curate_insight falls back to direct add.

        To trigger LLM classification, we need entries with similarity in the
        ambiguous range (similar_threshold <= sim < identical_threshold).
        We use identical-content entries so MockEmbedding yields sim=1.0 which
        is above identical_threshold and would normally be classified heuristically.
        Instead, we use a FailingLLM and a fresh entry whose embedding will be
        produced by MockEmbedding — but the _ensure_embeddings call for existing
        entries will work, and then the pipeline will fail at the LLM classify step.

        Actually, the simpler path: embed succeeds, but the LLM call in
        _classify_llm raises. We set up entries with similarity in the
        ambiguous zone by using MockEmbedding with different-but-similar texts.
        The heuristic returns None (ambiguous), triggering _classify_llm which
        raises. The outer try/except catches and falls back.
        """
        embeddings = MockEmbedding()
        failing_llm = _FailingLLM()

        # Config with low similar_threshold so entries are found, but high
        # identical_threshold so heuristic returns None (ambiguous -> LLM).
        config = CuratorConfig(
            similar_threshold=0.0,  # everything is "similar"
            identical_threshold=1.1,  # nothing is "identical" by heuristic
            conflict_threshold=1.1,  # nothing is "conflicting" by heuristic
        )
        curator = PlaybookCurator(embeddings, failing_llm, config)  # type: ignore[arg-type]

        # Pre-populate playbook with an entry that has an embedding
        existing = PlaybookEntry(
            id="e1",
            content="existing tip",
            embedding=embeddings.embed(["existing tip"])[0],
        )
        playbook = Playbook(entries=[existing])

        result = curator.curate_insight(
            _insight("a somewhat related insight"), playbook,
        )

        assert isinstance(result, CurationResult)
        assert result.action == "add"
        assert curator.metrics.fallback_direct_adds >= 1

    def test_curator_never_hard_fails(self) -> None:
        """Even with both embedding and LLM broken, curate_insight returns a result."""
        failing_embeddings = _FailingEmbedding()
        failing_llm = _FailingLLM()
        curator = PlaybookCurator(failing_embeddings, failing_llm)  # type: ignore[arg-type]

        playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="existing entry"),
        ])

        # Must not raise
        result = curator.curate_insight(_insight(), playbook)

        assert isinstance(result, CurationResult)
        assert result.new_entry is not None
        assert len(playbook.entries) >= 2  # original + fallback add
