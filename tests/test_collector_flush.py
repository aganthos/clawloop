"""Tests for EpisodeCollector.flush_buffer()."""

from lfx.collector import EpisodeCollector
from lfx.core.episode import Message
from lfx.core.reward import RewardPipeline


class _TrackingCallback:
    """Records batches passed to on_batch."""

    def __init__(self):
        self.batches = []

    def __call__(self, episodes):
        self.batches.append(list(episodes))


class TestFlushBuffer:
    def test_flush_sends_buffered_episodes(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
            on_batch=cb,
        )
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        collector.ingest(msgs, session_id="s1")
        collector.ingest(msgs, session_id="s2")

        # batch_size=100 so natural flush has not triggered yet
        assert len(cb.batches) == 0

        flushed = collector.flush_buffer()

        assert flushed == 2
        assert len(cb.batches) == 1
        assert len(cb.batches[0]) == 2

    def test_flush_empty_buffer_returns_zero(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
            on_batch=cb,
        )

        flushed = collector.flush_buffer()

        assert flushed == 0
        assert len(cb.batches) == 0

    def test_flush_without_callback_returns_zero(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
            # no on_batch
        )
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        collector.ingest(msgs, session_id="s1")

        flushed = collector.flush_buffer()

        assert flushed == 0
