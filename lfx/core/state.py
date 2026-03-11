"""Content-addressed state identity for reproducible learning iterations.

A ``StateID`` is the SHA-256 hash of the deterministic serialization of all
three layers (harness, router, weights).  This provides a compact, reproducible
fingerprint of the complete agent configuration at any point in time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from lfx.layers.harness import Harness
    from lfx.layers.router import Router
    from lfx.layers.weights import Weights


def _safe_default(obj: Any) -> str:
    """Fallback serializer that logs a warning before using str()."""
    log.warning(
        "Non-serializable object in StateID: %s (%s)", type(obj).__name__, obj,
    )
    return str(obj)


def _canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys, no extra whitespace)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_safe_default)


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


@dataclass(frozen=True)
class StateID:
    """Content-addressed identity of the full layer configuration."""

    harness_hash: str
    router_hash: str
    weights_hash: str
    combined_hash: str  # hash of all three
    created_at: float

    @classmethod
    def from_layers(
        cls,
        harness: Harness,
        router: Router,
        weights: Weights,
    ) -> StateID:
        """Compute a ``StateID`` from live layer instances."""
        h_hash = _sha256(_canonical_json(harness.to_dict()))
        r_hash = _sha256(_canonical_json(router.to_dict()))
        w_hash = _sha256(_canonical_json(weights.to_dict()))
        combined = _sha256(f"{h_hash}:{r_hash}:{w_hash}")
        return cls(
            harness_hash=h_hash,
            router_hash=r_hash,
            weights_hash=w_hash,
            combined_hash=combined,
            created_at=time.time(),
        )

    @classmethod
    def from_dicts(
        cls,
        harness_dict: dict[str, Any],
        router_dict: dict[str, Any],
        weights_dict: dict[str, Any],
    ) -> StateID:
        """Compute a ``StateID`` from raw dict representations."""
        h_hash = _sha256(_canonical_json(harness_dict))
        r_hash = _sha256(_canonical_json(router_dict))
        w_hash = _sha256(_canonical_json(weights_dict))
        combined = _sha256(f"{h_hash}:{r_hash}:{w_hash}")
        return cls(
            harness_hash=h_hash,
            router_hash=r_hash,
            weights_hash=w_hash,
            combined_hash=combined,
            created_at=time.time(),
        )
