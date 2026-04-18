"""Deterministic content hashing for reproducible identity."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

log = logging.getLogger(__name__)


def _safe_default(obj: Any) -> str:
    log.warning(
        "Non-serializable object in canonical_json: %s (%s)",
        type(obj).__name__,
        obj,
    )
    return str(obj)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization (sorted keys, compact separators)."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_safe_default)


def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def canonical_hash(obj: dict[str, Any]) -> str:
    return sha256_hex(canonical_json(obj))
