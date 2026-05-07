"""Tests for clawloop.integrations.mlflow — MlflowSink."""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta, Timing, TokenUsage
from clawloop.core.reward import RewardSignal
from clawloop.core.state import StateID

_METRICS: list[tuple[dict[str, float], int | None]] = []
_ARTIFACTS: list[tuple[Any, str]] = []
_ENDED: list[bool] = []


class _FakeRun:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeMlflow:
    _last_run: _FakeRun | None = None
    _tracking_uri: str | None = None
    _experiment_name: str | None = None

    @staticmethod
    def set_tracking_uri(uri: str) -> None:
        _FakeMlflow._tracking_uri = uri

    @staticmethod
    def set_experiment(name: str) -> None:
        _FakeMlflow._experiment_name = name

    @staticmethod
    def start_run(**kwargs: Any) -> _FakeRun:
        run = _FakeRun(**kwargs)
        _FakeMlflow._last_run = run
        return run

    @staticmethod
    def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
        _METRICS.append((dict(metrics), step))

    @staticmethod
    def log_dict(dictionary: Any, artifact_file: str) -> None:
        _ARTIFACTS.append((dictionary, artifact_file))

    @staticmethod
    def end_run() -> None:
        _ENDED.append(True)


@pytest.fixture(autouse=True)
def _reset_logged():
    _METRICS.clear()
    _ARTIFACTS.clear()
    _ENDED.clear()
    _FakeMlflow._last_run = None
    _FakeMlflow._tracking_uri = None
    _FakeMlflow._experiment_name = None


@pytest.fixture(autouse=True)
def _inject_fake_mlflow():
    original = sys.modules.get("mlflow")
    sys.modules["mlflow"] = _FakeMlflow  # type: ignore[assignment]
    yield
    if original is not None:
        sys.modules["mlflow"] = original
    else:
        sys.modules.pop("mlflow", None)


from clawloop.integrations.mlflow import MlflowSink  # noqa: E402

_BASE_TS = 1_700_000_000.0


def _make_episode(
    *,
    reward: float = 0.8,
    episode_id: str = "ep-001",
    task_id: str = "task-1",
    bench: str = "test",
) -> Episode:
    mapped = reward * 2.0 - 1.0
    signals = {
        "outcome": RewardSignal(name="outcome", value=mapped, confidence=1.0),
        "execution": RewardSignal(name="execution", value=0.5, confidence=0.9),
    }
    return Episode(
        id=episode_id,
        state_id="state-abc123",
        task_id=task_id,
        bench=bench,
        messages=[
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(
            signals=signals,
            token_usage=TokenUsage(prompt_tokens=50, completion_tokens=20, total_tokens=70),
            timing=Timing(total_ms=100.0, per_step_ms=[100.0]),
        ),
        model="gpt-4o",
        created_at=_BASE_TS,
    )


def _find_metric(key: str) -> list[tuple[float, int | None]]:
    results = []
    for metrics, step in _METRICS:
        if key in metrics:
            results.append((metrics[key], step))
    return results


class TestInit:
    def test_basic_init(self) -> None:
        sink = MlflowSink()
        assert sink._run is not None

    def test_init_options_passed(self) -> None:
        MlflowSink(
            experiment_name="clawloop-demo",
            run_name="run-1",
            tracking_uri="file:///tmp/mlruns",
            tags={"suite": "demo"},
        )

        assert _FakeMlflow._tracking_uri == "file:///tmp/mlruns"
        assert _FakeMlflow._experiment_name == "clawloop-demo"
        assert _FakeMlflow._last_run is not None
        assert _FakeMlflow._last_run.kwargs["run_name"] == "run-1"
        assert _FakeMlflow._last_run.kwargs["tags"] == {"suite": "demo"}

    def test_import_error_without_mlflow(self) -> None:
        original = sys.modules.get("mlflow")
        sys.modules["mlflow"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="mlflow"):
                MlflowSink()
        finally:
            if original is not None:
                sys.modules["mlflow"] = original


class TestLogEpisodes:
    def test_reward_scalars_logged(self) -> None:
        sink = MlflowSink()
        sink.log_episodes([_make_episode(reward=0.8)], iteration=0)

        means = _find_metric("reward.mean")
        assert len(means) == 1
        assert abs(means[0][0] - 0.6) < 1e-6
        assert means[0][1] == 0

    def test_episode_summaries_logged_as_artifact(self) -> None:
        sink = MlflowSink()
        sink.log_episodes([_make_episode()], iteration=2)

        assert len(_ARTIFACTS) == 1
        payload, artifact_file = _ARTIFACTS[0]
        assert artifact_file == "iterations/2/episodes.json"
        assert payload[0]["episode_id"] == "ep-001"

    def test_episode_artifact_can_be_disabled(self) -> None:
        sink = MlflowSink(log_episodes=False)
        sink.log_episodes([_make_episode()], iteration=0)

        assert _ARTIFACTS == []

    def test_empty_episodes_noop(self) -> None:
        sink = MlflowSink()
        sink.log_episodes([], iteration=0)

        assert _METRICS == []
        assert _ARTIFACTS == []

    def test_auto_step_increments(self) -> None:
        sink = MlflowSink()
        sink.log_episodes([_make_episode()])
        sink.log_episodes([_make_episode()])

        steps = [step for _, step in _METRICS]
        assert 0 in steps
        assert 1 in steps


class TestLogIteration:
    def test_playbook_and_state_artifacts_logged(self) -> None:
        sink = MlflowSink()
        harness = MagicMock()
        entry = MagicMock()
        entry.effective_score.return_value = 2.0
        entry.helpful = 5
        entry.harmful = 1
        harness.playbook.entries = [entry]
        sid = StateID(
            harness_hash="aabbccddee11" * 6,
            router_hash="112233445566" * 6,
            weights_hash="ffeeddccbbaa" * 6,
            combined_hash="abcdef012345" * 6,
            created_at=_BASE_TS,
        )

        sink.log_iteration(3, [_make_episode()], harness=harness, state_id=sid)

        assert _find_metric("playbook.size")[0] == (1.0, 3)
        artifact_files = {artifact_file for _, artifact_file in _ARTIFACTS}
        assert "iterations/3/playbook_entries.json" in artifact_files
        assert "iterations/3/state_hashes.json" in artifact_files

    def test_log_iteration_syncs_auto_step(self) -> None:
        sink = MlflowSink()
        sink.log_iteration(5, [_make_episode()])
        sink.log_episodes([_make_episode()])

        steps = [step for _, step in _METRICS]
        assert 6 in steps

    def test_backend_errors_fail_soft(self) -> None:
        sink = MlflowSink()
        sink._mlflow.log_metrics = MagicMock(side_effect=RuntimeError("backend down"))

        sink.log_episodes([_make_episode()], iteration=0)


class TestFinish:
    def test_finish_ends_run(self) -> None:
        sink = MlflowSink()
        sink.finish()

        assert _ENDED == [True]
