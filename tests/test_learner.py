"""Tests for AsyncLearner — background learning from episode batches."""

import time

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.loop import AgentState
from lfx.layers.harness import Playbook, PlaybookEntry
from lfx.learner import AsyncLearner


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
