"""Tests for PlaybookEntry.needs_reembed() — embedding invalidation logic."""

from clawloop.learning_layers.harness import PlaybookEntry


class TestEmbeddingInvalidation:
    """PlaybookEntry.needs_reembed() detects stale or missing embeddings."""

    def test_needs_reembed_when_no_embedding(self) -> None:
        """Entry without an embedding always needs re-embedding."""
        entry = PlaybookEntry(
            id="e1", content="tip", embedding=None, embedding_model_id=None,
        )

        assert entry.needs_reembed("text-embedding-3-small") is True

    def test_needs_reembed_when_model_changed(self) -> None:
        """Entry embedded with a different model needs re-embedding."""
        entry = PlaybookEntry(
            id="e1",
            content="tip",
            embedding=[0.1, 0.2, 0.3],
            embedding_model_id="text-embedding-ada-002",
        )

        assert entry.needs_reembed("text-embedding-3-small") is True

    def test_no_reembed_when_current(self) -> None:
        """Entry with embedding from the current model does not need re-embedding."""
        entry = PlaybookEntry(
            id="e1",
            content="tip",
            embedding=[0.1, 0.2, 0.3],
            embedding_model_id="text-embedding-3-small",
        )

        assert entry.needs_reembed("text-embedding-3-small") is False
