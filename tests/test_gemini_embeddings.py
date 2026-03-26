"""Integration tests using real Gemini embeddings.

These tests call the Gemini API and cost real tokens.
Skipped automatically when GOOGLE_API_KEY is not set.

Run with: pytest tests/test_gemini_embeddings.py -v
"""

import os

import pytest

from clawloop.core.embeddings import GeminiEmbedding, cosine_similarity, find_similar
from clawloop.core.curator import PlaybookCurator, CuratorConfig
from clawloop.layers.harness import Insight, Playbook, PlaybookEntry
from clawloop.llm import MockLLMClient

# Load .env if present
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

_HAS_KEY = bool(os.environ.get("GOOGLE_API_KEY"))
skip_no_key = pytest.mark.skipif(not _HAS_KEY, reason="GOOGLE_API_KEY not set")


@skip_no_key
class TestGeminiEmbedding:
    def _emb(self) -> GeminiEmbedding:
        return GeminiEmbedding(model="gemini-embedding-001")

    def test_single_embed(self) -> None:
        emb = self._emb()
        result = emb.embed(["hello world"])
        assert len(result) == 1
        assert len(result[0]) > 100  # Gemini returns high-dim vectors

    def test_batch_embed(self) -> None:
        emb = self._emb()
        result = emb.embed(["hello", "world", "foo bar"])
        assert len(result) == 3
        # All same dimension
        dims = {len(v) for v in result}
        assert len(dims) == 1

    def test_similar_texts_high_similarity(self) -> None:
        emb = self._emb()
        vecs = emb.embed([
            "Always validate user inputs before processing",
            "Make sure to check user inputs for correctness",
        ])
        sim = cosine_similarity(vecs[0], vecs[1])
        assert sim > 0.7, f"Similar texts should have high similarity, got {sim}"

    def test_dissimilar_texts_low_similarity(self) -> None:
        emb = self._emb()
        vecs = emb.embed([
            "Always validate user inputs before processing",
            "The weather in Paris is sunny today",
        ])
        sim = cosine_similarity(vecs[0], vecs[1])
        assert sim < 0.5, f"Dissimilar texts should have low similarity, got {sim}"

    def test_identical_texts_near_one(self) -> None:
        emb = self._emb()
        vecs = emb.embed([
            "Use chain of thought for math problems",
            "Use chain of thought for math problems",
        ])
        sim = cosine_similarity(vecs[0], vecs[1])
        assert sim > 0.99, f"Identical texts should have sim ~1.0, got {sim}"


@skip_no_key
class TestGeminiCurator:
    """End-to-end curator tests with real Gemini embeddings + mock LLM."""

    def _curator(self) -> PlaybookCurator:
        emb = GeminiEmbedding(model="gemini-embedding-001")
        llm = MockLLMClient(responses=["complementary"])
        return PlaybookCurator(embeddings=emb, llm=llm)

    def test_add_dissimilar_insight(self) -> None:
        curator = self._curator()
        pb = Playbook()
        pb.add(PlaybookEntry(id="e1", content="Always validate user inputs"))

        insight = Insight(content="The optimal batch size for training is 32")
        result = curator.curate_insight(insight, pb)
        # Dissimilar content should be added directly
        assert result.action == "add"

    def test_skip_identical_insight(self) -> None:
        curator = self._curator()
        pb = Playbook()
        pb.add(PlaybookEntry(id="e1", content="Always validate user inputs before processing"))

        # Embed the existing entry first
        curator._ensure_embeddings(pb)

        insight = Insight(content="Always validate user inputs before processing")
        result = curator.curate_insight(insight, pb)
        assert result.action == "skip_redundant"

    def test_find_similar_with_real_embeddings(self) -> None:
        emb = GeminiEmbedding(model="gemini-embedding-001")
        pb = Playbook()
        pb.add(PlaybookEntry(id="e1", content="Validate all user inputs carefully"))
        pb.add(PlaybookEntry(id="e2", content="The weather forecast predicts rain"))
        pb.add(PlaybookEntry(id="e3", content="Check input parameters before API calls"))

        # Embed all entries
        vecs = emb.embed([e.content for e in pb.entries])
        for entry, vec in zip(pb.entries, vecs):
            entry.embedding = vec
            entry.embedding_model_id = emb.model

        # Query similar to input validation
        query = emb.embed(["Always verify user-provided data"])[0]
        similar = find_similar(query, pb.entries, threshold=0.5)

        # e1 and e3 should be similar, e2 (weather) should not
        similar_ids = {e.id for e, _sim in similar}
        assert "e1" in similar_ids, "Input validation entry should be similar"
        assert "e3" in similar_ids, "API input check entry should be similar"

    def test_model_attribute_set(self) -> None:
        emb = GeminiEmbedding(model="gemini-embedding-001")
        assert emb.model == "gemini-embedding-001"
