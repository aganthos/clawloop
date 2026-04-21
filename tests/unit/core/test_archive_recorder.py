"""Unit tests for ``clawloop.core.archive_recorder.ArchiveRecorder``.

The recorder is a thin wrapper around an ``ArchiveStore`` that owns the
run-level counters (run_id, best_reward, initial_reward, total_cost,
prev_variant_hash) so ``learning_loop`` doesn't have to. These tests pin
its contract: what it writes, when, and that archive exceptions are
caught and logged rather than propagated.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)
from clawloop.core.archive_recorder import ArchiveRecorder
from clawloop.core.loop import AgentState
from clawloop.core.state import StateID
from clawloop.core.types import FBResult


class _RecordingStore:
    """Minimal ArchiveStore that remembers every write for inspection."""

    def __init__(self) -> None:
        self.runs: list[RunRecord] = []
        self.iterations: list[IterationRecord] = []
        self.episode_batches: list[list[EpisodeRecord]] = []
        self.variants: list[AgentVariant] = []
        self.completions: list[tuple[str, float, float, int]] = []

    def log_run_start(self, run: RunRecord) -> None:
        self.runs.append(run)

    def log_iteration(self, iteration: IterationRecord) -> None:
        self.iterations.append(iteration)

    def log_episodes(self, episodes: list[EpisodeRecord]) -> None:
        self.episode_batches.append(list(episodes))

    def log_variant(self, variant: AgentVariant) -> None:
        self.variants.append(variant)

    def log_run_complete(
        self,
        run_id: str,
        best_reward: float,
        improvement_delta: float,
        total_cost_tokens: int = 0,
    ) -> None:
        self.completions.append((run_id, best_reward, improvement_delta, total_cost_tokens))


def _mk_episode(episode_id: str = "ep1", task_id: str = "t1", reward: float = 0.5) -> Any:
    """Minimal duck-typed Episode stub — recorder only touches documented fields."""
    ep = MagicMock()
    ep.id = episode_id
    ep.task_id = task_id
    ep.bench = "unit"
    ep.model = "test-model"
    ep.messages = []
    ep.created_at = 1234567.0
    ep.summary.total_reward = reward
    ep.summary.normalized_reward.return_value = reward
    ep.summary.signals = {}
    ep.summary.token_usage = None
    ep.summary.timing = None
    ep.n_steps.return_value = 3
    return ep


def test_init_writes_run_start_and_initial_variant() -> None:
    store = _RecordingStore()
    agent_state = AgentState()
    recorder = ArchiveRecorder(
        store, agent_state, bench="unit-bench", domain_tags=["a", "b"], n_iterations=5
    )

    assert len(store.runs) == 1
    run = store.runs[0]
    assert run.bench == "unit-bench"
    assert run.domain_tags == ["a", "b"]
    assert run.n_iterations == 5
    assert run.best_reward == 0.0
    assert run.improvement_delta == 0.0
    assert run.total_cost_tokens == 0
    assert run.parent_run_id is None
    assert run.completed_at is None
    assert run.run_id  # non-empty
    # run_id is stable and exposed so the loop doesn't regenerate
    assert recorder.run_id == run.run_id

    assert len(store.variants) == 1
    variant = store.variants[0]
    assert variant.variant_hash == run.config_hash
    assert variant.first_seen_run_id == run.run_id


def test_archive_none_uses_null_store_silently() -> None:
    # archive=None should fall back to NullArchiveStore — no errors, no writes
    recorder = ArchiveRecorder(None, AgentState(), bench="b", domain_tags=None, n_iterations=1)
    recorder.record_episodes(0, [_mk_episode()])
    recorder.record_complete()
    # Nothing to assert except the absence of exceptions.
    assert recorder.run_id  # still generated


def test_record_episodes_writes_one_batch_with_correct_fields() -> None:
    store = _RecordingStore()
    recorder = ArchiveRecorder(store, AgentState(), bench="b", domain_tags=None, n_iterations=1)

    ep_a = _mk_episode("ep-a", "task-a", reward=0.7)
    ep_b = _mk_episode("ep-b", "task-b", reward=0.3)
    recorder.record_episodes(iteration=2, episodes=[ep_a, ep_b])

    assert len(store.episode_batches) == 1
    batch = store.episode_batches[0]
    assert len(batch) == 2
    assert [r.episode_id for r in batch] == ["ep-a", "ep-b"]
    assert all(r.iteration_num == 2 for r in batch)
    assert all(r.run_id == recorder.run_id for r in batch)
    assert batch[0].messages_ref == "traces/ep-a.json"


def test_record_episodes_noop_on_empty_list() -> None:
    store = _RecordingStore()
    recorder = ArchiveRecorder(store, AgentState(), bench="b", domain_tags=None, n_iterations=1)

    recorder.record_episodes(iteration=0, episodes=[])

    # record_episodes with no episodes must not emit an empty batch
    assert store.episode_batches == []


def test_record_iteration_writes_iteration_record_and_no_variant_when_unchanged() -> None:
    store = _RecordingStore()
    agent_state = AgentState()
    recorder = ArchiveRecorder(store, agent_state, bench="b", domain_tags=None, n_iterations=3)

    initial_variant_count = len(store.variants)
    state_id = agent_state.state_id()
    fb_results = {"harness": FBResult(status="ok", metrics={"tokens_used": 100})}
    ep = _mk_episode()

    recorder.record_iteration(
        iteration=0,
        agent_state=agent_state,
        state_id=state_id,
        fb_results=fb_results,
        episodes=[ep],
        avg_reward=0.5,
        prev_avg_reward=0.0,
    )

    assert len(store.iterations) == 1
    it = store.iterations[0]
    assert it.iteration_num == 0
    assert it.run_id == recorder.run_id
    assert it.mean_reward == 0.5
    assert it.reward_delta == 0.5
    assert it.cost_tokens == 100
    assert it.evolver_action == {"harness": {"tokens_used": 100}}
    # Variant hasn't changed — no new variant row
    assert len(store.variants) == initial_variant_count


def test_record_iteration_writes_new_variant_when_config_changes() -> None:
    store = _RecordingStore()
    agent_state = AgentState()
    recorder = ArchiveRecorder(store, agent_state, bench="b", domain_tags=None, n_iterations=1)
    initial_variant_count = len(store.variants)

    agent_state.harness.system_prompts["main"] = "you are a changed agent"

    recorder.record_iteration(
        iteration=0,
        agent_state=agent_state,
        state_id=agent_state.state_id(),
        fb_results={},
        episodes=[_mk_episode()],
        avg_reward=0.1,
        prev_avg_reward=0.0,
    )

    assert len(store.variants) == initial_variant_count + 1
    new_variant = store.variants[-1]
    assert new_variant.first_seen_run_id == recorder.run_id


def test_record_complete_writes_completion_with_accumulated_metrics() -> None:
    store = _RecordingStore()
    recorder = ArchiveRecorder(store, AgentState(), bench="b", domain_tags=None, n_iterations=2)

    recorder.record_iteration(
        iteration=0,
        agent_state=AgentState(),
        state_id=AgentState().state_id(),
        fb_results={"w": FBResult(status="ok", metrics={"tokens_used": 50})},
        episodes=[_mk_episode(reward=0.2)],
        avg_reward=0.2,
        prev_avg_reward=0.0,
    )
    recorder.record_iteration(
        iteration=1,
        agent_state=AgentState(),
        state_id=AgentState().state_id(),
        fb_results={"w": FBResult(status="ok", metrics={"tokens_used": 75})},
        episodes=[_mk_episode(reward=0.8)],
        avg_reward=0.8,
        prev_avg_reward=0.2,
    )

    recorder.record_complete()

    assert len(store.completions) == 1
    run_id, best_reward, improvement_delta, total_cost = store.completions[0]
    assert run_id == recorder.run_id
    assert best_reward == 0.8
    assert improvement_delta == pytest.approx(0.8 - 0.2)  # best - initial
    assert total_cost == 125


def test_record_complete_handles_zero_iterations() -> None:
    # best_reward and initial_reward stay None if no iterations ran — must not crash
    store = _RecordingStore()
    recorder = ArchiveRecorder(store, AgentState(), bench="b", domain_tags=None, n_iterations=0)

    recorder.record_complete()

    assert len(store.completions) == 1
    run_id, best, delta, cost = store.completions[0]
    assert best == 0.0
    assert delta == 0.0
    assert cost == 0


def test_archive_exceptions_are_swallowed_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    class _BrokenStore(_RecordingStore):
        def log_episodes(self, episodes: list[EpisodeRecord]) -> None:
            raise RuntimeError("archive disk full")

    store = _BrokenStore()
    recorder = ArchiveRecorder(store, AgentState(), bench="b", domain_tags=None, n_iterations=1)

    with caplog.at_level(logging.WARNING, logger="clawloop.core.archive_recorder"):
        recorder.record_episodes(iteration=0, episodes=[_mk_episode()])

    # Exception must not propagate; a warning must be emitted.
    assert any("failed to log episodes" in rec.message.lower() for rec in caplog.records)


def test_run_start_exception_does_not_block_construction(caplog: pytest.LogCaptureFixture) -> None:
    class _RunStartFails(_RecordingStore):
        def log_run_start(self, run: RunRecord) -> None:
            raise RuntimeError("db down")

    store = _RunStartFails()
    with caplog.at_level(logging.WARNING, logger="clawloop.core.archive_recorder"):
        recorder = ArchiveRecorder(
            store, AgentState(), bench="b", domain_tags=None, n_iterations=1
        )

    assert recorder.run_id  # still usable
    assert any("run start" in rec.message.lower() for rec in caplog.records)


def test_state_id_parameter_is_used_for_harness_snapshot_hash() -> None:
    store = _RecordingStore()
    agent_state = AgentState()
    recorder = ArchiveRecorder(store, agent_state, bench="b", domain_tags=None, n_iterations=1)

    sid = StateID(
        harness_hash="HHH",
        router_hash="R",
        weights_hash="W",
        combined_hash="C",
        created_at=0.0,
    )
    recorder.record_iteration(
        iteration=0,
        agent_state=agent_state,
        state_id=sid,
        fb_results={},
        episodes=[],
        avg_reward=0.0,
        prev_avg_reward=0.0,
    )

    assert store.iterations[0].harness_snapshot_hash == "HHH"


def test_cost_tracking_survives_archive_exception() -> None:
    # Regression: if log_iteration raises, tokens_used must still be accumulated
    # onto total_cost_tokens. Cost tracking is independent of archive availability.
    class _BrokenIter(_RecordingStore):
        def log_iteration(self, iteration: IterationRecord) -> None:
            raise RuntimeError("db down")

    store = _BrokenIter()
    recorder = ArchiveRecorder(store, AgentState(), bench="b", domain_tags=None, n_iterations=1)

    recorder.record_iteration(
        iteration=0,
        agent_state=AgentState(),
        state_id=AgentState().state_id(),
        fb_results={"h": FBResult(status="ok", metrics={"tokens_used": 42})},
        episodes=[_mk_episode(reward=0.1)],
        avg_reward=0.1,
        prev_avg_reward=0.0,
    )
    recorder.record_complete()

    assert len(store.completions) == 1
    _, _, _, total_cost = store.completions[0]
    assert total_cost == 42  # accumulated despite log_iteration raising
