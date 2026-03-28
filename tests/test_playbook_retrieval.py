"""Tests for embedding-first playbook retrieval."""

from clawloop.core.embeddings import MockEmbedding
from clawloop.layers.harness import Harness, Playbook, PlaybookEntry


def _make_playbook() -> Playbook:
    """Playbook with 3 entries across different domains."""
    return Playbook(entries=[
        PlaybookEntry(
            id="e-math", content="Show step-by-step work for arithmetic problems",
            tags=["math"], helpful=3,
        ),
        PlaybookEntry(
            id="e-code", content="Always validate input types before processing",
            tags=["coding"], helpful=2,
        ),
        PlaybookEntry(
            id="e-write", content="Use active voice and short sentences for clarity",
            tags=["writing"], helpful=1,
        ),
    ])


class TestSystemPromptBackwardCompat:
    """system_prompt() without context must behave exactly as before."""

    def test_no_context_uses_tag_path(self):
        h = Harness(system_prompts={"test": "Base."}, _embeddings=MockEmbedding())
        h.playbook = _make_playbook()
        prompt = h.system_prompt("test", task_tags={"math"})
        assert "step-by-step" in prompt
        assert "validate input" not in prompt

    def test_no_context_no_tags_returns_all(self):
        h = Harness(system_prompts={"test": "Base."})
        h.playbook = _make_playbook()
        prompt = h.system_prompt("test")
        assert "step-by-step" in prompt
        assert "validate input" in prompt
        assert "active voice" in prompt

    def test_empty_context_uses_tag_path(self):
        h = Harness(system_prompts={"test": "Base."}, _embeddings=MockEmbedding())
        h.playbook = _make_playbook()
        prompt = h.system_prompt("test", context="")
        # Empty context should NOT trigger embedding path
        assert "step-by-step" in prompt
        assert "validate input" in prompt

    def test_whitespace_context_uses_tag_path(self):
        h = Harness(system_prompts={"test": "Base."}, _embeddings=MockEmbedding())
        h.playbook = _make_playbook()
        prompt = h.system_prompt("test", context="   ")
        assert "step-by-step" in prompt


class TestEmbeddingRetrieval:
    """Embedding-first retrieval when context is provided."""

    def test_embedding_retrieval_returns_semantic_matches(self):
        emb = MockEmbedding(dim=8)
        h = Harness(
            system_prompts={"test": "Base."},
            _embeddings=emb,
            _retrieval_threshold=0.0,  # accept all matches for MockEmbedding
        )
        h.playbook = _make_playbook()

        # With threshold=0.0 and MockEmbedding, all entries with embeddings match
        prompt = h.system_prompt("test", context="arithmetic calculation")
        assert "semantic match" in prompt

    def test_embedding_miss_falls_back_to_tags(self):
        emb = MockEmbedding(dim=8)
        h = Harness(
            system_prompts={"test": "Base."},
            _embeddings=emb,
            _retrieval_threshold=0.999,  # impossibly high — no matches
        )
        h.playbook = _make_playbook()

        prompt = h.system_prompt("test", task_tags={"math"}, context="hello")
        # Should fall back to tags since embedding threshold is too high
        assert "step-by-step" in prompt
        assert "semantic match" not in prompt

    def test_no_embeddings_no_tags_returns_full_playbook(self):
        emb = MockEmbedding(dim=8)
        h = Harness(
            system_prompts={"test": "Base."},
            _embeddings=emb,
            _retrieval_threshold=0.999,  # no embedding matches
        )
        h.playbook = _make_playbook()

        prompt = h.system_prompt("test", context="something unrelated")
        # No embedding match, no tags → full playbook
        assert "step-by-step" in prompt
        assert "validate input" in prompt
        assert "active voice" in prompt

    def test_no_embeddings_provider_uses_tag_path(self):
        h = Harness(system_prompts={"test": "Base."})
        h.playbook = _make_playbook()
        # No embeddings provider, but context provided → falls to tag path
        prompt = h.system_prompt("test", task_tags={"coding"}, context="check types")
        assert "validate input" in prompt
        assert "step-by-step" not in prompt


class TestLazyEmbed:
    """Entries without embeddings get embedded on first retrieval."""

    def test_lazy_embed_populates_entry_embeddings(self):
        emb = MockEmbedding(dim=8)
        h = Harness(
            system_prompts={"test": "Base."},
            _embeddings=emb,
            _retrieval_threshold=0.0,
        )
        h.playbook = _make_playbook()

        # Entries start without embeddings
        for e in h.playbook.entries:
            assert e.embedding is None

        h.system_prompt("test", context="anything")

        # After retrieval, entries should have embeddings
        for e in h.playbook.entries:
            assert e.embedding is not None
            assert e.embedding_model_id == "mock-embedding"

    def test_already_embedded_entries_not_reembedded(self):
        emb = MockEmbedding(dim=8)
        h = Harness(
            system_prompts={"test": "Base."},
            _embeddings=emb,
            _retrieval_threshold=0.0,
        )
        h.playbook = _make_playbook()

        # First call embeds
        h.system_prompt("test", context="first")
        original_embeddings = [e.embedding[:] for e in h.playbook.entries]

        # Second call should not change embeddings (model_id matches)
        h.system_prompt("test", context="second")
        for i, e in enumerate(h.playbook.entries):
            assert e.embedding == original_embeddings[i]

    def test_stale_embeddings_get_refreshed(self):
        emb = MockEmbedding(dim=8)
        h = Harness(
            system_prompts={"test": "Base."},
            _embeddings=emb,
            _retrieval_threshold=0.0,
        )
        h.playbook = _make_playbook()

        # Pre-set embeddings with a different model_id (stale)
        for e in h.playbook.entries:
            e.embedding = [0.0] * 8
            e.embedding_model_id = "old-model"

        h.system_prompt("test", context="refresh")

        # Should be re-embedded with new model
        for e in h.playbook.entries:
            assert e.embedding_model_id == "mock-embedding"
            assert e.embedding != [0.0] * 8


class TestQueryTextTruncation:
    """Long context gets capped."""

    def test_long_context_truncated(self):
        emb = MockEmbedding(dim=8)
        h = Harness(
            system_prompts={"test": "Base."},
            _embeddings=emb,
            _retrieval_threshold=0.0,
        )
        h.playbook = _make_playbook()

        # 1000-char query should not crash or cause issues
        long_query = "x" * 1000
        prompt = h.system_prompt("test", context=long_query)
        assert "PLAYBOOK" in prompt


class TestMaxRetrievalEntries:
    """Full-playbook fallback is capped."""

    def test_full_fallback_capped(self):
        emb = MockEmbedding(dim=8)
        h = Harness(
            system_prompts={"test": "Base."},
            _embeddings=emb,
            _retrieval_threshold=0.999,
            _max_retrieval_entries=2,
        )
        # Create 5 entries
        entries = [
            PlaybookEntry(id=f"e-{i}", content=f"Entry {i}", helpful=i)
            for i in range(5)
        ]
        h.playbook = Playbook(entries=entries)

        entries_out, reason = h._retrieve_entries(None, "query")
        assert reason == "full"
        assert len(entries_out) == 2
        # Should be sorted by effective_score (highest first)
        assert entries_out[0].id == "e-4"
        assert entries_out[1].id == "e-3"


class TestRenderEntries:
    """Header rendering per retrieval reason."""

    def test_embedding_header(self):
        entries = [PlaybookEntry(id="e1", content="test", tags=["a"])]
        text = Harness._render_entries(entries, "embedding")
        assert "semantic match" in text

    def test_tags_header(self):
        entries = [PlaybookEntry(id="e1", content="test", tags=["a"])]
        text = Harness._render_entries(entries, "tags")
        assert "## PLAYBOOK" in text
        assert "semantic" not in text

    def test_full_header(self):
        entries = [PlaybookEntry(id="e1", content="test", tags=["a"])]
        text = Harness._render_entries(entries, "full")
        assert "## PLAYBOOK" in text

    def test_structured_entry_renders_as_skill(self):
        entries = [PlaybookEntry(
            id="s1", content="Break into sub-problems.",
            name="Divide and Conquer", description="Complex problems",
            anti_patterns="One-step solutions",
        )]
        text = Harness._render_entries(entries, "embedding")
        assert "### Divide and Conquer" in text
        assert "**When**: Complex problems" in text
        assert "**Anti-pattern**: One-step solutions" in text

    def test_empty_entries_returns_empty(self):
        assert Harness._render_entries([], "full") == ""
