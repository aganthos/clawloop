"""Thread-safe session resolver and turn counter for the proxy."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict


class SessionTracker:
    """Resolves session IDs and tracks per-session turn indices.

    Single-process only — the threading.Lock is in-process.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: defaultdict[str, int] = defaultdict(int)

    def resolve_session(
        self,
        run_id: str | None,
        session_id: str | None,
    ) -> tuple[str, bool]:
        """Resolve a session identifier with precedence: run_id > session_id > uuid4.

        Returns:
            (session_id_str, attributed) where attributed is True if a caller-supplied
            identifier was used, False if one was auto-generated.
        """
        if run_id is not None:
            return run_id, True
        if session_id is not None:
            return session_id, True
        return uuid.uuid4().hex, False

    def next_turn(self, session_id: str) -> int:
        """Return the next monotonic turn index for *session_id* (starting at 0)."""
        with self._lock:
            idx = self._turns[session_id]
            self._turns[session_id] = idx + 1
            return idx
