"""Tests for SessionTracker — session resolution and turn counting."""

import threading

from clawloop.proxy_session import SessionTracker


class TestResolveSession:
    def test_run_id_returns_attributed(self) -> None:
        tracker = SessionTracker()
        sid, attributed = tracker.resolve_session(run_id="run-abc", session_id=None)
        assert sid == "run-abc"
        assert attributed is True

    def test_session_id_returns_attributed(self) -> None:
        tracker = SessionTracker()
        sid, attributed = tracker.resolve_session(run_id=None, session_id="sess-xyz")
        assert sid == "sess-xyz"
        assert attributed is True

    def test_nothing_returns_uuid_not_attributed(self) -> None:
        tracker = SessionTracker()
        sid, attributed = tracker.resolve_session(run_id=None, session_id=None)
        assert len(sid) == 32  # uuid4 hex is 32 chars
        assert attributed is False

    def test_run_id_takes_precedence_over_session_id(self) -> None:
        tracker = SessionTracker()
        sid, attributed = tracker.resolve_session(run_id="run-123", session_id="sess-456")
        assert sid == "run-123"
        assert attributed is True


class TestNextTurn:
    def test_monotonically_increasing(self) -> None:
        tracker = SessionTracker()
        results = [tracker.next_turn("s1") for _ in range(5)]
        assert results == [0, 1, 2, 3, 4]

    def test_independent_counters_per_session(self) -> None:
        tracker = SessionTracker()
        assert tracker.next_turn("a") == 0
        assert tracker.next_turn("b") == 0
        assert tracker.next_turn("a") == 1
        assert tracker.next_turn("b") == 1

    def test_thread_safety(self) -> None:
        tracker = SessionTracker()
        results: list[int] = []
        lock = threading.Lock()

        def worker() -> None:
            local: list[int] = []
            for _ in range(100):
                local.append(tracker.next_turn("shared"))
            with lock:
                results.extend(local)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert sorted(results) == list(range(400))
