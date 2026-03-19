"""Semantic similarity infrastructure for playbook curation.

Provides embedding protocols, cosine similarity, and nearest-neighbour
lookup over ``PlaybookEntry`` objects.  Used by the Curator to detect
duplicates and cluster related entries.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
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
    query_dim = len(query_embedding)
    scored: list[tuple[PlaybookEntry, float]] = []
    for entry in entries:
        if entry.embedding is None:
            continue
        if len(entry.embedding) != query_dim:
            continue  # skip dimension mismatch (stale model)
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

    def __init__(self, dim: int = _MOCK_DIM, model: str = "mock-embedding") -> None:
        self.dim = dim
        self.model = model

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


# ---------------------------------------------------------------------------
# GeminiEmbedding — direct Gemini API (no litellm dependency)
# ---------------------------------------------------------------------------

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/{model}"
_GEMINI_BATCH_LIMIT = 100  # max texts per batchEmbedContents call


class GeminiEmbedding:
    """Embedding provider using the Gemini API directly.

    Uses ``batchEmbedContents`` to embed multiple texts in a single API call,
    reducing rate-limit pressure. Falls back to per-text calls for batches
    larger than the API limit (100).

    No litellm dependency — uses ``urllib`` only.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self._api_key = api_key

    def _get_key(self) -> str:
        if self._api_key:
            return self._api_key
        import os
        key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise RuntimeError("No Gemini API key: set GOOGLE_API_KEY or pass api_key=")
        return key

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using batch API. Chunks into groups of 100."""
        key = self._get_key()
        results: list[list[float]] = []
        for i in range(0, len(texts), _GEMINI_BATCH_LIMIT):
            chunk = texts[i : i + _GEMINI_BATCH_LIMIT]
            results.extend(self._batch_embed(chunk, key))
        return results

    def _batch_embed(self, texts: list[str], key: str) -> list[list[float]]:
        url = _GEMINI_BASE.format(model=self.model) + f":batchEmbedContents?key={key}"
        body = json.dumps({
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                }
                for t in texts
            ],
        }).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req).read())
        return [e["values"] for e in resp["embeddings"]]
