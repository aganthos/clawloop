"""Tests for EpisodeCollector — live mode episode construction."""

import threading

from lfx.collector import EpisodeCollector
from lfx.core.episode import Message
from lfx.core.reward import RewardPipeline
from lfx.extractors.formatting import FormattingFilter


class _TrackingCallback:
    """Records batches passed to on_batch."""
    def __init__(self):
        self.batches = []

    def __call__(self, episodes):
        self.batches.append(list(episodes))


class TestEpisodeCollector:
    def test_ingest_creates_episode(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        ep = collector.ingest(msgs, session_id="s1")
        assert ep.id
        assert ep.bench == "live"
        assert ep.task_id == "s1"
        assert len(ep.messages) == 2

    def test_batch_triggers_callback(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=cb,
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 20),
        ]
        collector.ingest(msgs, session_id="s1")
        assert len(cb.batches) == 0
        collector.ingest(msgs, session_id="s2")
        assert len(cb.batches) == 1
        assert len(cb.batches[0]) == 2

    def test_filtered_episodes_not_buffered(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=cb,
            formatting_filter=FormattingFilter(min_response_length=100),
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="short"),
        ]
        ep = collector.ingest(msgs, session_id="s1")
        assert ep.summary.filtered is True
        collector.ingest(msgs, session_id="s2")
        assert len(cb.batches) == 0

    def test_submit_feedback_updates_signal(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a long enough response here"),
        ]
        ep = collector.ingest(msgs, session_id="s1")
        assert collector.submit_feedback(ep.id, 1.0) is True
        assert "user" in ep.summary.signals
        assert ep.summary.signals["user"].value == 1.0

    def test_submit_feedback_unknown_id(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        assert collector.submit_feedback("nonexistent", 1.0) is False

    def test_thread_safety(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=10,
            on_batch=cb,
        )

        def ingest_many():
            for i in range(5):
                msgs = [
                    Message(role="user", content=f"q-{i}"),
                    Message(role="assistant", content="a" * 20),
                ]
                collector.ingest(msgs, session_id=f"s-{threading.current_thread().name}-{i}")

        threads = [threading.Thread(target=ingest_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_eps = sum(len(b) for b in cb.batches)
        assert total_eps == 20

    def test_metrics(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
            formatting_filter=FormattingFilter(min_response_length=100),
        )
        msgs_good = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 200),
        ]
        msgs_bad = [
            Message(role="user", content="q"),
            Message(role="assistant", content="short"),
        ]
        collector.ingest(msgs_good, session_id="s1")
        collector.ingest(msgs_bad, session_id="s2")
        collector.submit_feedback("nonexistent", 1.0)

        m = collector.metrics
        assert m["episodes_collected"] == 2
        assert m["episodes_filtered"] == 1
        assert m["feedback_received"] == 0
        assert m["feedback_missed"] == 1
