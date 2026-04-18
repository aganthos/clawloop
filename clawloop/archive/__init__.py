"""Evolution Archive — structured data capture for learning runs.

Public surface: ``ArchiveStore`` protocol, ``JsonlArchiveStore``,
``NullArchiveStore``, and the 4 schema dataclasses. Indexed SQLite/Postgres
stores, Parquet export, and curation tooling live in the enterprise package.
"""

from clawloop.archive.jsonl_store import JsonlArchiveStore
from clawloop.archive.null_store import NullArchiveStore
from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)
from clawloop.archive.store import ArchiveStore

__all__ = [
    "AgentVariant",
    "ArchiveStore",
    "EpisodeRecord",
    "IterationRecord",
    "JsonlArchiveStore",
    "NullArchiveStore",
    "RunRecord",
]
