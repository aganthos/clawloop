"""Tests for AsyncLearner on_learn_complete callback."""

import time
from unittest.mock import MagicMock, patch

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.loop import AgentState
from lfx.core.types import FBResult, Future
from lfx.layers.harness import Harness, Playbook, PlaybookEntry
from lfx.learner import AsyncLearner


def _make_episodes(n: int, reward: float = 0.8) -> list[Episode]:
    eps = []
    for i in range(n):
        ep = Episode(
            id=f"ep-{i}", state_id="s1", task_id=f"t-{i}", bench="n8n",
            messages=[
                Message(role="user", content=f"q-{i}"),
                Message(role="assistant", content=f"a-{i}" * 20),
            ],
            step_boundaries=[0], steps=[],
            summary=EpisodeSummary(total_reward=reward),
        )
        eps.append(ep)
    return eps


class TestAsyncLearnerCallback:
    def test_on_learn_complete_called_on_success(self) -> None:
        state = AgentState()
        state.harness.playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="Be helpful"),
        ])
        callback = MagicMock()
        learner = AsyncLearner(
            agent_state=state,
            active_layers=["harness"],
            on_learn_complete=callback,
        )
        learner.start()
        learner.on_batch(_make_episodes(2, reward=0.9))
        time.sleep(1.0)
        learner.stop()
        assert callback.call_count >= 1
        call_args = callback.call_args
        assert call_args[1]["success"] is True
        assert call_args[1]["error"] is None

    def test_on_learn_complete_called_on_failure(self) -> None:
        state = AgentState()
        callback = MagicMock()
        learner = AsyncLearner(
            agent_state=state,
            active_layers=["harness"],
            on_learn_complete=callback,
        )

        with patch.object(
            state.harness, "forward_backward",
            side_effect=RuntimeError("boom"),
        ):
            learner._learn(_make_episodes(2))

        assert callback.call_count == 1
        call_args = callback.call_args
        assert call_args[1]["success"] is False
        # forward_backward exception is caught per-layer; the batch ends at
        # "no layers to optimize" since all FB results are "error".
        assert call_args[1]["error"] == "no layers to optimize"

    def test_no_callback_does_not_error(self) -> None:
        """Without callback, _learn should still work normally."""
        state = AgentState()
        state.harness.playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="Be helpful"),
        ])
        learner = AsyncLearner(
            agent_state=state,
            active_layers=["harness"],
        )
        # Should not raise
        learner._learn(_make_episodes(2, reward=0.9))
        assert learner.metrics["batches_trained"] >= 1
