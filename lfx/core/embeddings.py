"""Semantic similarity infrastructure for playbook curation.

Provides embedding protocols, cosine similarity, and nearest-neighbour
lookup over ``PlaybookEntry`` objects.  Used by the Curator to detect
duplicates and cluster related entries.
"""

from __future__ import annotations

import hashlib
import math
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lfx.layers.harness import PlaybookEntry


# ---------------------------------------------------------------------------
# EmbeddingProvider protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding text into dense vectors."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""
        ...


# ---------------------------------------------------------------------------
# cosine_similarity — pure math, no numpy
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors.

    Returns 0.0 when either vector has zero magnitude.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )
    dot = sum(ai * bi for ai, bi in zip(a, b))
    mag_a = math.sqrt(sum(ai * ai for ai in a))
    mag_b = math.sqrt(sum(bi * bi for bi in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# find_similar — nearest-neighbour lookup over PlaybookEntry objects
# ---------------------------------------------------------------------------

def find_similar(
    query_embedding: list[float],
    entries: list[PlaybookEntry],
    threshold: float = 0.75,
) -> list[tuple[PlaybookEntry, float]]:
    """Return entries with cosine similarity above *threshold*, sorted descending.

    Entries whose ``embedding`` is ``None`` are silently skipped.
    """
    scored: list[tuple[PlaybookEntry, float]] = []
    for entry in entries:
        if entry.embedding is None:
            continue
        sim = cosine_similarity(query_embedding, entry.embedding)
        if sim >= threshold:
            scored.append((entry, sim))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# MockEmbedding — deterministic, hash-based vectors for tests
# ---------------------------------------------------------------------------

_MOCK_DIM = 64


class MockEmbedding:
    """Deterministic embedding provider for tests.

    Produces vectors derived from a SHA-256 hash of each input text,
    ensuring reproducible results across runs.
    """

    def __init__(self, dim: int = _MOCK_DIM) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand hash to fill *dim* floats.  SHA-256 gives 32 bytes;
        # cycle through them, mapping each byte to [-1, 1].
        raw: list[float] = []
        for i in range(self.dim):
            byte_val = digest[i % len(digest)]
            raw.append((byte_val / 127.5) - 1.0)
        # Normalise to unit length so cosine_similarity behaves nicely.
        mag = math.sqrt(sum(v * v for v in raw))
        if mag == 0.0:
            return raw
        return [v / mag for v in raw]


# ---------------------------------------------------------------------------
# LiteLLMEmbedding — production provider backed by litellm.embedding()
# ---------------------------------------------------------------------------

class LiteLLMEmbedding:
    """Embedding provider backed by ``litellm.embedding()``.

    Supports 100+ providers via LiteLLM's unified API.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.api_base = api_base

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via litellm."""
        import litellm

        kwargs: dict[str, object] = {}
        if self.api_key is not None:
            kwargs["api_key"] = self.api_key
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base

        response = litellm.embedding(
            model=self.model,
            input=texts,
            **kwargs,
        )
        # litellm returns data sorted by index; sort explicitly to be safe.
        items = sorted(response.data, key=lambda d: d["index"])
        return [item["embedding"] for item in items]
