"""Tests for clawloop.integrations.wandb — WandbSink."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from clawloop.core.episode import (
    Episode,
    EpisodeSummary,
    Message,
    StepMeta,
    Timing,
    TokenUsage,
)
from clawloop.core.reward import RewardSignal
from clawloop.core.state import StateID

# ---------------------------------------------------------------------------
# Fake wandb module — injected via sys.modules before importing WandbSink
# ---------------------------------------------------------------------------

_LOGGED: list[tuple[dict, int | None]] = []
_FINISHED: list[bool] = []


class _FakeTable:
    def __init__(self, columns: list[str], data: list[list]) -> None:
        self.columns = columns
        self.data = data


class _FakeRun:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.config = kwargs.get("config", {})

    def log(self, metrics: dict, step: int | None = None) -> None:
        _LOGGED.append((dict(metrics), step))

    def finish(self) -> None:
        _FINISHED.append(True)


class _FakeWandb:
    Table = _FakeTable
    _last_run: _FakeRun | None = None

    @staticmethod
    def init(**kwargs: Any) -> _FakeRun:
        run = _FakeRun(**kwargs)
        _FakeWandb._last_run = run
        return run


@pytest.fixture(autouse=True)
def _reset_logged():
    _LOGGED.clear()
    _FINISHED.clear()
    _FakeWandb._last_run = None


@pytest.fixture(autouse=True)
def _inject_fake_wandb():
    """Ensure the fake wandb is importable before WandbSink is loaded."""
    original = sys.modules.get("wandb")
    sys.modules["wandb"] = _FakeWandb  # type: ignore[assignment]
    yield
    if original is not None:
        sys.modules["wandb"] = original
    else:
        sys.modules.pop("wandb", None)


# Import after fake is in place
from clawloop.integrations.wandb import WandbSink  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_TS = 1_700_000_000.0


def _make_episode(
    *,
    reward: float = 0.8,
    episode_id: str = "ep-001",
    task_id: str = "task-1",
    bench: str = "test",
    n_steps: int = 2,
) -> Episode:
    mapped = reward * 2.0 - 1.0
    signals = {
        "outcome": RewardSignal(name="outcome", value=mapped, confidence=1.0),
        "execution": RewardSignal(name="execution", value=0.5, confidence=0.9),
    }
    messages = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    steps = [
        StepMeta(
            t=i,
            reward=(reward if i == n_steps - 1 else 0.0),
            done=(i == n_steps - 1),
            timing_ms=100.0,
        )
        for i in range(n_steps)
    ]
    return Episode(
        id=episode_id,
        state_id="state-abc123",
        task_id=task_id,
        bench=bench,
        messages=messages,
        step_boundaries=[0],
        steps=steps,
        summary=EpisodeSummary(
            signals=signals,
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
            timing=Timing(total_ms=200.0, per_step_ms=[100.0, 100.0]),
        ),
        model="gpt-4o",
        created_at=_BASE_TS,
    )


def _find_logged(key: str) -> list[tuple[Any, int | None]]:
    """Return all (value, step) pairs where *key* was logged."""
    results = []
    for metrics, step in _LOGGED:
        if key in metrics:
            results.append((metrics[key], step))
    return results


# ---------------------------------------------------------------------------
# TestInit
# ---------------------------------------------------------------------------


class TestInit:
    def test_basic_init(self) -> None:
        sink = WandbSink()
        assert sink._run is not None

    def test_run_id_passed(self) -> None:
        WandbSink(run_id="my-run")
        assert _FakeWandb._last_run is not None
        assert _FakeWandb._last_run.init_kwargs["id"] == "my-run"
        assert _FakeWandb._last_run.init_kwargs["resume"] == "allow"

    def test_project_passed(self) -> None:
        WandbSink(project="test-proj")
        assert _FakeWandb._last_run.init_kwargs["project"] == "test-proj"

    def test_entity_passed(self) -> None:
        WandbSink(entity="my-team")
        assert _FakeWandb._last_run.init_kwargs["entity"] == "my-team"

    def test_config_passed(self) -> None:
        WandbSink(config={"lr": 0.01})
        assert _FakeWandb._last_run is not None
        assert _FakeWandb._last_run.init_kwargs["config"] == {"lr": 0.01}

    def test_import_error_without_wandb(self) -> None:
        original = sys.modules.get("wandb")
        sys.modules["wandb"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="wandb"):
                WandbSink()
        finally:
            if original is not None:
                sys.modules["wandb"] = original


# ---------------------------------------------------------------------------
# TestLogEpisodes
# ---------------------------------------------------------------------------


class TestLogEpisodes:
    def test_reward_scalars_logged(self) -> None:
        sink = WandbSink()
        eps = [_make_episode(reward=0.8)]
        sink.log_episodes(eps, iteration=0)

        means = _find_logged("reward/mean")
        assert len(means) == 1
        assert abs(means[0][0] - 0.6) < 1e-6  # effective_reward for 0.8 mapped
        assert means[0][1] == 0

    def test_min_max_reward(self) -> None:
        sink = WandbSink()
        eps = [_make_episode(reward=0.3), _make_episode(reward=0.9, episode_id="ep-002")]
        sink.log_episodes(eps, iteration=5)

        mins = _find_logged("reward/min")
        maxs = _find_logged("reward/max")
        assert len(mins) == 1
        assert len(maxs) == 1
        # 0.3 → mapped -0.4, 0.9 → mapped 0.8
        assert abs(mins[0][0] - (-0.4)) < 1e-6
        assert abs(maxs[0][0] - 0.8) < 1e-6

    def test_episode_count(self) -> None:
        sink = WandbSink()
        eps = [_make_episode(), _make_episode(episode_id="ep-002")]
        sink.log_episodes(eps, iteration=0)

        counts = _find_logged("episodes/count")
        assert counts[0][0] == 2

    def test_per_signal_mean(self) -> None:
        sink = WandbSink()
        eps = [_make_episode(reward=0.8)]
        sink.log_episodes(eps, iteration=0)

        outcome = _find_logged("reward/outcome_mean")
        assert len(outcome) == 1
        assert abs(outcome[0][0] - 0.6) < 1e-6

    def test_empty_episodes_noop(self) -> None:
        sink = WandbSink()
        sink.log_episodes([], iteration=0)
        assert len(_LOGGED) == 0

    def test_auto_step_increments(self) -> None:
        sink = WandbSink()
        sink.log_episodes([_make_episode()])
        sink.log_episodes([_make_episode()])
        steps = [step for _, step in _LOGGED]
        assert 0 in steps
        assert 1 in steps

    def test_log_iteration_syncs_auto_step(self) -> None:
        """log_iteration should advance _step so a subsequent log_episodes picks up."""
        sink = WandbSink()
        sink.log_iteration(5, [_make_episode()])
        sink.log_episodes([_make_episode()])  # should use step=6, not step=0
        steps = [step for _, step in _LOGGED]
        assert 6 in steps

    def test_episode_table_logged(self) -> None:
        sink = WandbSink()
        sink.log_episodes([_make_episode()], iteration=0)

        tables = _find_logged("episodes/table")
        assert len(tables) == 1
        table = tables[0][0]
        assert isinstance(table, _FakeTable)
        assert "episode_id" in table.columns
        assert len(table.data) == 1
        assert table.data[0][0] == "ep-001"

    def test_table_disabled(self) -> None:
        sink = WandbSink(log_episodes=False)
        sink.log_episodes([_make_episode()], iteration=0)

        tables = _find_logged("episodes/table")
        assert len(tables) == 0


# ---------------------------------------------------------------------------
# TestLogIteration
# ---------------------------------------------------------------------------


class TestLogIteration:
    def test_rewards_logged(self) -> None:
        sink = WandbSink()
        sink.log_iteration(3, [_make_episode(reward=0.7)])

        means = _find_logged("reward/mean")
        assert means[0][1] == 3

    def test_playbook_metrics(self) -> None:
        sink = WandbSink()

        harness = MagicMock()
        entry = MagicMock()
        entry.effective_score.return_value = 2.0
        entry.helpful = 5
        entry.harmful = 1
        harness.playbook.entries = [entry]

        sink.log_iteration(0, [_make_episode()], harness=harness)

        sizes = _find_logged("playbook/size")
        assert sizes[0][0] == 1

        scores = _find_logged("playbook/mean_score")
        assert scores[0][0] == 2.0

    def test_playbook_empty(self) -> None:
        sink = WandbSink()
        harness = MagicMock()
        harness.playbook.entries = []
        sink.log_iteration(0, [_make_episode()], harness=harness)

        sizes = _find_logged("playbook/size")
        assert sizes[0][0] == 0
        # No mean_score when empty
        assert len(_find_logged("playbook/mean_score")) == 0

    def test_state_hashes(self) -> None:
        sink = WandbSink()
        sid = StateID(
            harness_hash="aabbccddee11" * 6,
            router_hash="112233445566" * 6,
            weights_hash="ffeeddccbbaa" * 6,
            combined_hash="abcdef012345" * 6,
            created_at=_BASE_TS,
        )
        sink.log_iteration(0, [_make_episode()], state_id=sid)

        combined = _find_logged("state/combined")
        assert combined[0][0] == "abcdef012345"

        harness = _find_logged("state/harness")
        assert harness[0][0] == "aabbccddee11"

    def test_fb_results(self) -> None:
        sink = WandbSink()

        @dataclass
        class FBR:
            status: str = "ok"
            metrics: dict = field(
                default_factory=lambda: {"insights_generated": 3, "tokens_used": 100}
            )

        fb = {"harness": FBR()}
        sink.log_iteration(0, [_make_episode()], fb_results=fb)

        insights = _find_logged("fb/harness/insights_generated")
        assert insights[0][0] == 3

    def test_fb_results_skip_non_numeric(self) -> None:
        sink = WandbSink()

        @dataclass
        class FBR:
            status: str = "ok"
            metrics: dict = field(default_factory=lambda: {"detail": "some string", "count": 5})

        fb = {"harness": FBR()}
        sink.log_iteration(0, [_make_episode()], fb_results=fb)

        # String metrics should not be logged
        assert len(_find_logged("fb/harness/detail")) == 0
        assert len(_find_logged("fb/harness/count")) == 1


# ---------------------------------------------------------------------------
# TestAfterIteration
# ---------------------------------------------------------------------------


class TestAfterIteration:
    def test_callback_shape(self) -> None:
        """after_iteration should work as a learning_loop callback."""
        sink = WandbSink()

        agent_state = MagicMock()
        from clawloop.learning_layers.harness import Harness

        harness = Harness()
        agent_state.harness = harness
        agent_state.state_id.return_value = StateID(
            harness_hash="a" * 64,
            router_hash="b" * 64,
            weights_hash="c" * 64,
            combined_hash="d" * 64,
            created_at=_BASE_TS,
        )

        eps = [_make_episode()]
        sink.after_iteration(0, agent_state, eps)

        # Should have logged rewards + playbook + state hashes
        assert len(_find_logged("reward/mean")) == 1
        assert len(_find_logged("playbook/size")) == 1
        assert len(_find_logged("state/combined")) == 1


# ---------------------------------------------------------------------------
# TestFinish
# ---------------------------------------------------------------------------


class TestFinish:
    def test_finish_calls_run_finish(self) -> None:
        sink = WandbSink()
        sink.finish()
        assert len(_FINISHED) == 1


# ---------------------------------------------------------------------------
# TestFailSoft
# ---------------------------------------------------------------------------


class TestFailSoft:
    def test_log_episodes_swallows_error(self) -> None:
        sink = WandbSink()
        # Make the run.log raise
        sink._run.log = MagicMock(side_effect=RuntimeError("boom"))
        # Should not raise
        sink.log_episodes([_make_episode()], iteration=0)

    def test_log_iteration_swallows_error(self) -> None:
        sink = WandbSink()
        sink._run.log = MagicMock(side_effect=RuntimeError("boom"))
        sink.log_iteration(0, [_make_episode()])

    def test_finish_swallows_error(self) -> None:
        sink = WandbSink()
        sink._run.finish = MagicMock(side_effect=RuntimeError("boom"))
        sink.finish()
