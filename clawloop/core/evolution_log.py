"""EvolutionLog — append-only JSONL tracking of per-iteration evolution data.

Each entry captures: state hash before, actions taken, state hash after,
reward delta. Seeds future learned evolvers by recording the (state, action,
reward_delta) tuples that a reinforcement-learning evolver would train on.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class EvolutionEntry:
    """One iteration's evolution record."""

    iteration: int
    state_hash_before: str
    state_hash_after: str
    actions: list[str]
    reward_before: float
    reward_after: float
    backend: str = "local"
    timestamp: float = field(default_factory=time.time)

    def reward_delta(self) -> float:
        return self.reward_after - self.reward_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "timestamp": self.timestamp,
            "state_hash_before": self.state_hash_before,
            "state_hash_after": self.state_hash_after,
            "actions": self.actions,
            "reward_before": self.reward_before,
            "reward_after": self.reward_after,
            "reward_delta": self.reward_delta(),
            "backend": self.backend,
        }


class EvolutionLog:
    """Append-only JSONL writer for evolution entries."""

    def __init__(self, output_dir: str | Path | None = None) -> None:
        self._path: Path | None = None
        if output_dir:
            self._path = Path(output_dir) / "evolution.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: EvolutionEntry) -> None:
        if self._path is None:
            return
        try:
            with open(self._path, "a") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
                f.flush()
        except Exception:
            log.exception("Failed to write evolution log entry")
