"""JsonlArchiveStore — append-only JSONL archive for local use.

Zero-dependency public implementation. Every call appends a single JSON line
tagged with ``record_type`` so the file is a self-describing stream.

Query/index capabilities live in the enterprise ``SqliteArchiveStore``;
this store is intentionally write-optimized and scan-only for reads.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 1


def _safe_run_id(run_id: str) -> str:
    """Reject run_id values that could escape the archive directory."""
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id or run_id.startswith("."):
        raise ValueError(f"unsafe run_id for filesystem path: {run_id!r}")
    return run_id


def _write_line(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, separators=(",", ":")) + "\n")
        f.flush()
        os.fsync(f.fileno())


class JsonlArchiveStore:
    """Append-only JSONL archive. One line per record, tagged by type."""

    def __init__(self, archive_dir: str | Path = "~/.clawloop/archive") -> None:
        self._archive_dir = Path(archive_dir).expanduser().resolve()
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._archive_dir.chmod(0o700)
        except OSError:
            pass
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _runs_path(self) -> Path:
        return self._archive_dir / "runs.jsonl"

    def _iterations_path(self) -> Path:
        return self._archive_dir / "iterations.jsonl"

    def _variants_path(self) -> Path:
        return self._archive_dir / "variants.jsonl"

    def _episodes_path(self, run_id: str) -> Path:
        return self._archive_dir / _safe_run_id(run_id) / "episodes.jsonl"

    # ------------------------------------------------------------------
    # Write operations (ArchiveStore protocol)
    # ------------------------------------------------------------------

    def log_run_start(self, run: RunRecord) -> None:
        payload = {
            "_schema": _SCHEMA_VERSION,
            "record_type": "run_start",
            **run.to_dict(),
        }
        with self._lock:
            _write_line(self._runs_path(), payload)

    def log_iteration(self, iteration: IterationRecord) -> None:
        payload = {
            "_schema": _SCHEMA_VERSION,
            "record_type": "iteration",
            **iteration.to_dict(),
        }
        with self._lock:
            _write_line(self._iterations_path(), payload)

    def log_episodes(self, episodes: list[EpisodeRecord]) -> None:
        if not episodes:
            return
        run_id = _safe_run_id(episodes[0].run_id)
        # Enforce homogeneous batch: store routes files by run_id. Mixing
        # run_ids in one call would silently land them under the first's dir.
        for ep in episodes[1:]:
            if ep.run_id != run_id:
                raise ValueError(
                    "log_episodes batches must share a run_id; "
                    f"got {run_id!r} and {ep.run_id!r}"
                )
        path = self._episodes_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(
                {"_schema": _SCHEMA_VERSION, "record_type": "episode", **ep.to_dict()},
                separators=(",", ":"),
            )
            for ep in episodes
        ]
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                f.flush()
                os.fsync(f.fileno())

    def log_variant(self, variant: AgentVariant) -> None:
        payload = {
            "_schema": _SCHEMA_VERSION,
            "record_type": "variant",
            **variant.to_dict(),
        }
        with self._lock:
            _write_line(self._variants_path(), payload)

    def log_run_complete(
        self,
        run_id: str,
        best_reward: float,
        improvement_delta: float,
        total_cost_tokens: int = 0,
    ) -> None:
        payload = {
            "_schema": _SCHEMA_VERSION,
            "record_type": "run_complete",
            "run_id": run_id,
            "best_reward": best_reward,
            "improvement_delta": improvement_delta,
            "total_cost_tokens": total_cost_tokens,
            "completed_at": time.time(),
        }
        with self._lock:
            _write_line(self._runs_path(), payload)

    # ------------------------------------------------------------------
    # Read operations
    #
    # Linear scans. Intentional: the public store optimizes for safe,
    # simple writes. Indexed queries live in enterprise SqliteArchiveStore.
    #
    # Reads are lock-free: the store is append-only, so readers see a
    # consistent prefix on POSIX. Partial tail lines from concurrent
    # writes are caught by the JSONDecodeError handler.
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> RunRecord | None:
        path = self._runs_path()
        if not path.exists():
            return None

        best: dict | None = None
        completion: dict | None = None
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("run_id") != run_id:
                    continue
                rtype = rec.get("record_type")
                if rtype == "run_start":
                    best = rec
                elif rtype == "run_complete":
                    completion = rec

        if best is None:
            return None

        if completion is not None:
            best = {
                **best,
                "best_reward": completion.get("best_reward", best.get("best_reward", 0.0)),
                "improvement_delta": completion.get(
                    "improvement_delta", best.get("improvement_delta", 0.0)
                ),
                "total_cost_tokens": completion.get(
                    "total_cost_tokens", best.get("total_cost_tokens", 0)
                ),
                "completed_at": completion.get("completed_at"),
            }
        best.pop("_schema", None)
        best.pop("record_type", None)
        return RunRecord.from_dict(best)

    def get_similar_runs(
        self,
        config_hash: str,
        domain_tags: list[str],
        limit: int = 10,
    ) -> list[RunRecord]:
        """Linear scan, returns matches ordered by best_reward desc.

        OR semantics: a run matches if its ``config_hash`` equals the query
        OR any of its ``domain_tags`` is in the query set.
        """
        path = self._runs_path()
        if not path.exists():
            return []

        tag_set = set(domain_tags)
        starts: dict[str, dict] = {}
        completions: dict[str, dict] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                run_id = rec.get("run_id")
                if not run_id:
                    continue
                rtype = rec.get("record_type")
                if rtype == "run_start":
                    starts[run_id] = rec
                elif rtype == "run_complete":
                    completions[run_id] = rec

        merged: list[RunRecord] = []
        for run_id, start in starts.items():
            if start.get("config_hash") != config_hash and not (
                tag_set and tag_set.intersection(start.get("domain_tags", []))
            ):
                continue
            completion = completions.get(run_id)
            if completion is not None:
                start = {
                    **start,
                    "best_reward": completion.get("best_reward", start.get("best_reward", 0.0)),
                    "improvement_delta": completion.get(
                        "improvement_delta", start.get("improvement_delta", 0.0)
                    ),
                    "total_cost_tokens": completion.get(
                        "total_cost_tokens", start.get("total_cost_tokens", 0)
                    ),
                    "completed_at": completion.get("completed_at"),
                }
            start.pop("_schema", None)
            start.pop("record_type", None)
            merged.append(RunRecord.from_dict(start))

        merged.sort(key=lambda r: r.best_reward, reverse=True)
        return merged[:limit]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """No-op: files are opened-and-closed per write for crash safety."""

    def __enter__(self) -> JsonlArchiveStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
