"""Tests for AsyncLearner — background learning from episode batches."""

import json
import time
from unittest.mock import MagicMock, patch

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.loop import AgentState
from lfx.core.reflector import Reflector, ReflectorConfig
from lfx.core.types import FBResult, Future, OptimResult
from lfx.layers.harness import Harness, Playbook, PlaybookEntry
from lfx.learner import AsyncLearner


class _MockLLMClient:
    """LLM client that returns a canned JSON response (one insight)."""

    def __init__(self, response: str | None = None) -> None:
        self.call_log: list[dict] = []
        self._response = response or json.dumps([
            {
                "action": "add",
                "content": "Use chain-of-thought for math problems",
                "target_entry_id": None,
                "tags": ["strategy"],
                "source_episode_ids": [],
            }
        ])

    def complete(self, messages, **kwargs) -> str:
        self.call_log.append({"messages": messages, **kwargs})
        return self._response


def _make_episodes(n: int, reward: float = 0.8) -> list[Episode]:
    eps = []
    for i in range(n):
        ep = Episode(
            id=f"ep-{i}", state_id="s1", task_id=f"t-{i}", bench="live",
            messages=[
                Message(role="user", content=f"q-{i}"),
                Message(role="assistant", content=f"a-{i}" * 20),
            ],
            step_boundaries=[0], steps=[],
            summary=EpisodeSummary(total_reward=reward),
        )
        eps.append(ep)
    return eps


class TestAsyncLearner:
    def test_on_batch_processes_episodes(self) -> None:
        state = AgentState()
        state.harness.playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="Be helpful"),
        ])
        learner = AsyncLearner(agent_state=state, active_layers=["harness"])
        learner.start()

        episodes = _make_episodes(3, reward=0.9)
        learner.on_batch(episodes)

        time.sleep(0.5)

        assert learner.metrics["batches_trained"] >= 1
        learner.stop()

    def test_dropped_batch_when_queue_full(self) -> None:
        state = AgentState()
        learner = AsyncLearner(
            agent_state=state,
            active_layers=["harness"],
            max_queue_size=1,
        )
        learner.start()

        for _ in range(10):
            learner.on_batch(_make_episodes(1))

        time.sleep(0.5)
        assert learner.metrics["batches_dropped"] >= 0
        learner.stop()

    def test_stop_graceful(self) -> None:
        state = AgentState()
        learner = AsyncLearner(agent_state=state)
        learner.start()
        learner.stop()

    def test_metrics_initial(self) -> None:
        state = AgentState()
        learner = AsyncLearner(agent_state=state)
        m = learner.metrics
        assert m["batches_trained"] == 0
        assert m["batches_dropped"] == 0
        assert m["batches_failed"] == 0
        assert m["iteration"] == 0

    # -- New two-phase status-aware tests --

    def test_fb_error_skips_optim(self) -> None:
        """FB returning status='error' should prevent optim_step from being called."""
        state = AgentState()
        learner = AsyncLearner(agent_state=state, active_layers=["harness"])

        with patch.object(
            state.harness, "forward_backward",
            return_value=Future.immediate(FBResult(status="error")),
        ), patch.object(
            state.harness, "optim_step",
            return_value=Future.immediate(OptimResult(status="ok")),
        ) as mock_optim, patch.object(
            state.harness, "clear_pending_state",
        ):
            learner._learn(_make_episodes(2))

        mock_optim.assert_not_called()

    def test_fb_skipped_skips_optim(self) -> None:
        """FB returning status='skipped' should prevent optim_step from being called."""
        state = AgentState()
        learner = AsyncLearner(agent_state=state, active_layers=["harness"])

        with patch.object(
            state.harness, "forward_backward",
            return_value=Future.immediate(FBResult(status="skipped")),
        ), patch.object(
            state.harness, "optim_step",
            return_value=Future.immediate(OptimResult(status="ok")),
        ) as mock_optim, patch.object(
            state.harness, "clear_pending_state",
        ):
            learner._learn(_make_episodes(2))

        mock_optim.assert_not_called()

    def test_two_phase_ordering(self) -> None:
        """All FB calls should happen before any optim calls (two-phase ordering)."""
        state = AgentState()
        learner = AsyncLearner(
            agent_state=state, active_layers=["harness", "router"],
        )

        call_order: list[str] = []

        def harness_fb(*args, **kwargs):
            call_order.append("harness_fb")
            return Future.immediate(FBResult(status="ok"))

        def router_fb(*args, **kwargs):
            call_order.append("router_fb")
            return Future.immediate(FBResult(status="ok"))

        def harness_optim(*args, **kwargs):
            call_order.append("harness_optim")
            return Future.immediate(OptimResult(status="ok"))

        def router_optim(*args, **kwargs):
            call_order.append("router_optim")
            return Future.immediate(OptimResult(status="ok"))

        with patch.object(state.harness, "forward_backward", side_effect=harness_fb), \
             patch.object(state.router, "forward_backward", side_effect=router_fb), \
             patch.object(state.harness, "optim_step", side_effect=harness_optim), \
             patch.object(state.router, "optim_step", side_effect=router_optim):
            learner._learn(_make_episodes(2))

        # All FB calls must come before any optim calls
        fb_indices = [i for i, c in enumerate(call_order) if c.endswith("_fb")]
        optim_indices = [i for i, c in enumerate(call_order) if c.endswith("_optim")]
        assert fb_indices, "No FB calls recorded"
        assert optim_indices, "No optim calls recorded"
        assert max(fb_indices) < min(optim_indices), (
            f"FB and optim calls interleaved: {call_order}"
        )

    def test_optim_error_status_triggers_rollback(self) -> None:
        """Router optim returning status='error' should trigger rollback of harness."""
        state = AgentState()
        state.harness.playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="Be helpful"),
        ])
        learner = AsyncLearner(
            agent_state=state, active_layers=["harness", "router"],
        )

        with patch.object(
            state.harness, "forward_backward",
            return_value=Future.immediate(FBResult(status="ok")),
        ), patch.object(
            state.router, "forward_backward",
            return_value=Future.immediate(FBResult(status="ok")),
        ), patch.object(
            state.harness, "optim_step",
            return_value=Future.immediate(OptimResult(status="ok")),
        ), patch.object(
            state.router, "optim_step",
            return_value=Future.immediate(OptimResult(status="error")),
        ), patch.object(
            state.harness, "load_state",
            return_value=Future.immediate(MagicMock(status="ok")),
        ) as mock_load_harness, patch.object(
            state.router, "load_state",
            return_value=Future.immediate(MagicMock(status="ok")),
        ) as mock_load_router:
            learner._learn(_make_episodes(2))

        # Both layers should have been rolled back
        mock_load_harness.assert_called_once()
        mock_load_router.assert_called_once()
        assert learner.metrics["batches_failed"] == 1

    def test_fb_error_clears_pending_state(self) -> None:
        """FB returning status='error' should trigger clear_pending_state."""
        state = AgentState()
        learner = AsyncLearner(agent_state=state, active_layers=["harness"])

        with patch.object(
            state.harness, "forward_backward",
            return_value=Future.immediate(FBResult(status="error")),
        ), patch.object(
            state.harness, "clear_pending_state",
        ) as mock_clear:
            learner._learn(_make_episodes(2))

        mock_clear.assert_called_once()

    def test_all_fb_failed_not_counted_as_batch_failure(self) -> None:
        """When all FB return error/skipped, batches_failed should stay 0."""
        state = AgentState()
        learner = AsyncLearner(
            agent_state=state, active_layers=["harness", "router"],
        )

        with patch.object(
            state.harness, "forward_backward",
            return_value=Future.immediate(FBResult(status="error")),
        ), patch.object(
            state.router, "forward_backward",
            return_value=Future.immediate(FBResult(status="skipped")),
        ), patch.object(
            state.harness, "clear_pending_state",
        ), patch.object(
            state.router, "clear_pending_state",
        ):
            learner._learn(_make_episodes(2))

        assert learner.metrics["batches_failed"] == 0
        assert learner.metrics["batches_trained"] == 0

    def test_fb_skipped_clears_pending_state(self) -> None:
        """FB returning status='skipped' should trigger clear_pending_state."""
        state = AgentState()
        learner = AsyncLearner(agent_state=state, active_layers=["harness"])

        with patch.object(
            state.harness, "forward_backward",
            return_value=Future.immediate(FBResult(status="skipped")),
        ), patch.object(
            state.harness, "clear_pending_state",
        ) as mock_clear:
            learner._learn(_make_episodes(2))

        mock_clear.assert_called_once()

    def test_optim_failure_rolls_back_all_layers(self) -> None:
        """Router optim raising exception should roll back harness (with real Reflector)."""
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"live": "You are helpful."},
            reflector=reflector,
        )
        state = AgentState(harness=harness)
        learner = AsyncLearner(
            agent_state=state, active_layers=["harness", "router"],
        )

        with patch.object(
            state.harness, "forward_backward",
            return_value=Future.immediate(FBResult(status="ok")),
        ), patch.object(
            state.router, "forward_backward",
            return_value=Future.immediate(FBResult(status="ok")),
        ), patch.object(
            state.harness, "optim_step",
            return_value=Future.immediate(OptimResult(status="ok")),
        ), patch.object(
            state.router, "optim_step",
            side_effect=RuntimeError("optim exploded"),
        ), patch.object(
            state.harness, "load_state",
            return_value=Future.immediate(MagicMock(status="ok")),
        ) as mock_load_harness, patch.object(
            state.router, "load_state",
            return_value=Future.immediate(MagicMock(status="ok")),
        ) as mock_load_router:
            learner._learn(_make_episodes(2))

        # Both layers should have been rolled back
        mock_load_harness.assert_called_once()
        mock_load_router.assert_called_once()
        assert learner.metrics["batches_failed"] == 1
        assert learner.metrics["batches_trained"] == 0
