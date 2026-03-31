"""Tests for BackgroundScheduler and related tasks."""

from dataclasses import dataclass
from unittest.mock import MagicMock

from clawloop.core.background import (
    BackgroundScheduler,
    BackgroundState,
    EpisodeDreamer,
    PlaybookConsolidation,
)
from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.learning_layers.harness import Playbook, PlaybookEntry


def _make_episode(task_id="t1", reward=0.5):
    return Episode(
        id=Episode.new_id(),
        state_id="deadbeef",
        task_id=task_id,
        bench="test",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


def _make_playbook():
    return Playbook(entries=[
        PlaybookEntry(id="e1", content="Always greet the user"),
    ])


@dataclass
class MockTask:
    name: str = "mock_task"
    _should_run: bool = True
    run_count: int = 0

    def should_run(self, state):
        return self._should_run

    def run(self, state):
        self.run_count += 1


class TestBackgroundScheduler:
    def test_scheduler_runs_task_when_conditions_met(self) -> None:
        task = MockTask(name="eager", _should_run=True)
        scheduler = BackgroundScheduler(tasks=[task])
        scheduler.tick(
            playbook=_make_playbook(),
            recent_episodes=[_make_episode()],
            is_user_idle=True,
        )
        assert task.run_count == 1

    def test_scheduler_skips_task_when_conditions_not_met(self) -> None:
        task = MockTask(name="lazy", _should_run=False)
        scheduler = BackgroundScheduler(tasks=[task])
        scheduler.tick(
            playbook=_make_playbook(),
            recent_episodes=[_make_episode()],
            is_user_idle=True,
        )
        assert task.run_count == 0

    def test_in_progress_guard_prevents_overlap(self) -> None:
        task = MockTask(name="slow")
        scheduler = BackgroundScheduler(tasks=[task])
        # Simulate an in-progress run by adding the task name directly
        scheduler._in_progress.add("slow")
        scheduler.tick(
            playbook=_make_playbook(),
            recent_episodes=[_make_episode()],
            is_user_idle=True,
        )
        assert task.run_count == 0

    def test_record_episodes_tracks_count(self) -> None:
        t1 = MockTask(name="alpha")
        t2 = MockTask(name="beta")
        scheduler = BackgroundScheduler(tasks=[t1, t2])

        scheduler.record_episodes(5)
        assert scheduler._episodes_since["alpha"] == 5
        assert scheduler._episodes_since["beta"] == 5

        scheduler.record_episodes(3)
        assert scheduler._episodes_since["alpha"] == 8
        assert scheduler._episodes_since["beta"] == 8

    def test_multiple_tasks_independent(self) -> None:
        runner = MockTask(name="runner", _should_run=True)
        skipper = MockTask(name="skipper", _should_run=False)
        scheduler = BackgroundScheduler(tasks=[runner, skipper])
        scheduler.tick(
            playbook=_make_playbook(),
            recent_episodes=[_make_episode()],
            is_user_idle=True,
        )
        assert runner.run_count == 1
        assert skipper.run_count == 0


class TestPlaybookConsolidation:
    def test_playbook_consolidation_should_run(self) -> None:
        consolidation = PlaybookConsolidation(
            episode_threshold=10,
            min_interval=60.0,
            curator=MagicMock(),
        )

        # All conditions met
        state = BackgroundState(
            episodes_since_last_run=15,
            time_since_last_run=120.0,
            is_user_idle=True,
            playbook=_make_playbook(),
            recent_episodes=[],
        )
        assert consolidation.should_run(state) is True

        # Not enough episodes
        state.episodes_since_last_run = 5
        assert consolidation.should_run(state) is False

        # Enough episodes but not enough time
        state.episodes_since_last_run = 15
        state.time_since_last_run = 30.0
        assert consolidation.should_run(state) is False

        # Enough episodes and time but user not idle
        state.time_since_last_run = 120.0
        state.is_user_idle = False
        assert consolidation.should_run(state) is False

        # No curator
        no_curator = PlaybookConsolidation(curator=None)
        state.is_user_idle = True
        state.episodes_since_last_run = 100
        state.time_since_last_run = 999.0
        assert no_curator.should_run(state) is False

    def test_playbook_consolidation_runs_curator(self) -> None:
        curator = MagicMock()
        curator.consolidate.return_value = MagicMock(
            before=10, after=8, merged=1, pruned=1,
        )
        consolidation = PlaybookConsolidation(curator=curator)
        playbook = _make_playbook()

        state = BackgroundState(
            episodes_since_last_run=100,
            time_since_last_run=600.0,
            is_user_idle=True,
            playbook=playbook,
            recent_episodes=[],
        )
        consolidation.run(state)
        curator.consolidate.assert_called_once_with(playbook)
