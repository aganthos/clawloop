"""Tests for JsonlArchiveStore — public append-only JSONL archive."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from clawloop.archive.jsonl_store import JsonlArchiveStore
from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)


def _make_run(run_id: str = "r1", **overrides) -> RunRecord:
    kwargs = dict(
        run_id=run_id,
        bench="test-bench",
        domain_tags=["math", "arithmetic"],
        agent_config={"prompt": "hello"},
        config_hash="cfg-hash-1",
        n_iterations=3,
        best_reward=0.0,
        improvement_delta=0.0,
        total_cost_tokens=0,
        parent_run_id=None,
        created_at=time.time(),
        completed_at=None,
    )
    kwargs.update(overrides)
    return RunRecord(**kwargs)


def _make_iter(run_id: str = "r1", n: int = 0) -> IterationRecord:
    return IterationRecord(
        run_id=run_id,
        iteration_num=n,
        harness_snapshot_hash="snap",
        mean_reward=0.5,
        reward_trajectory=[0.4, 0.5, 0.6],
        evolver_action={"harness": {"insights_generated": 2}},
        cost_tokens=100,
        parent_variant_hash="p",
        child_variant_hash="c",
        reward_delta=0.1,
        created_at=time.time(),
    )


def _make_episode(run_id: str = "r1", ep_id: str = "e1") -> EpisodeRecord:
    return EpisodeRecord(
        run_id=run_id,
        iteration_num=0,
        episode_id=ep_id,
        task_id="t1",
        bench="test-bench",
        model="gpt-4o",
        reward=0.7,
        signals={},
        n_steps=3,
        n_tool_calls=1,
        token_usage={"total_tokens": 200},
        latency_ms=1200,
        messages_ref="traces/e1.json",
        created_at=time.time(),
    )


def _make_variant(vhash: str = "v1") -> AgentVariant:
    return AgentVariant(
        variant_hash=vhash,
        system_prompt="hi",
        playbook_snapshot={},
        model="gpt-4o",
        tools=[],
        first_seen_run_id="r1",
        created_at=time.time(),
    )


def test_log_and_read_run(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    run = _make_run()
    store.log_run_start(run)

    got = store.get_run(run.run_id)
    assert got is not None
    assert got.run_id == run.run_id
    assert got.config_hash == run.config_hash
    assert got.completed_at is None


def test_run_complete_updates_totals(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    store.log_run_start(_make_run())
    store.log_run_complete("r1", best_reward=0.9, improvement_delta=0.3, total_cost_tokens=500)

    got = store.get_run("r1")
    assert got is not None
    assert got.best_reward == 0.9
    assert got.improvement_delta == 0.3
    assert got.total_cost_tokens == 500
    assert got.completed_at is not None


def test_get_run_missing_returns_none(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    assert store.get_run("missing") is None


def test_log_iteration_appends_line(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    store.log_iteration(_make_iter())
    store.log_iteration(_make_iter(n=1))

    lines = (tmp_path / "iterations.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["record_type"] == "iteration"
    assert rec["iteration_num"] == 0


def test_log_episodes_writes_under_run_dir(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    store.log_episodes([_make_episode(ep_id="e1"), _make_episode(ep_id="e2")])

    ep_file = tmp_path / "r1" / "episodes.jsonl"
    assert ep_file.exists()
    lines = ep_file.read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["episode_id"] == "e1"
    assert json.loads(lines[1])["episode_id"] == "e2"


def test_log_episodes_empty_is_noop(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    store.log_episodes([])  # no exception, no file created
    assert not (tmp_path / "r1").exists()


def test_log_variant_appends(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    store.log_variant(_make_variant("v1"))
    store.log_variant(_make_variant("v2"))

    lines = (tmp_path / "variants.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2


def test_get_similar_runs_by_config_hash(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    store.log_run_start(_make_run(run_id="a", config_hash="H"))
    store.log_run_start(_make_run(run_id="b", config_hash="H", domain_tags=["nlp"]))
    store.log_run_start(_make_run(run_id="c", config_hash="OTHER", domain_tags=["other"]))
    store.log_run_complete("a", 0.8, 0.1, 0)
    store.log_run_complete("b", 0.9, 0.2, 0)
    store.log_run_complete("c", 0.95, 0.3, 0)

    hits = store.get_similar_runs(config_hash="H", domain_tags=[])
    hit_ids = {r.run_id for r in hits}
    assert hit_ids == {"a", "b"}
    # Ordered by best_reward desc
    assert hits[0].run_id == "b"


def test_get_similar_runs_by_tag(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    store.log_run_start(_make_run(run_id="a", config_hash="X", domain_tags=["math"]))
    store.log_run_start(_make_run(run_id="b", config_hash="Y", domain_tags=["nlp"]))
    store.log_run_start(_make_run(run_id="c", config_hash="Z", domain_tags=["vision"]))

    hits = store.get_similar_runs(config_hash="NO_MATCH", domain_tags=["math", "nlp"])
    assert {r.run_id for r in hits} == {"a", "b"}


def test_get_similar_runs_limit(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    for i in range(5):
        store.log_run_start(_make_run(run_id=f"r{i}", config_hash="H"))
        store.log_run_complete(f"r{i}", best_reward=0.1 * i, improvement_delta=0.0, total_cost_tokens=0)

    hits = store.get_similar_runs(config_hash="H", domain_tags=[], limit=3)
    assert len(hits) == 3
    assert hits[0].run_id == "r4"  # highest reward


def test_thread_safe_concurrent_writes(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)

    def writer(tid: int) -> None:
        for i in range(20):
            store.log_iteration(_make_iter(run_id=f"run-{tid}", n=i))

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = (tmp_path / "iterations.jsonl").read_text().strip().splitlines()
    assert len(lines) == 4 * 20
    for line in lines:
        # Every line must be complete valid JSON (no interleaving)
        json.loads(line)


def test_context_manager(tmp_path: Path) -> None:
    with JsonlArchiveStore(tmp_path) as store:
        store.log_run_start(_make_run())
    # no exception == pass
    assert (tmp_path / "runs.jsonl").exists()


def test_ignores_malformed_lines(tmp_path: Path) -> None:
    """Readers must tolerate a partial tail line from a crash."""
    store = JsonlArchiveStore(tmp_path)
    store.log_run_start(_make_run())
    # Simulate a crash mid-write by appending a partial line
    with open(tmp_path / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write('{"record_type": "run_start", "run_id": "par')

    got = store.get_run("r1")
    assert got is not None
    assert got.run_id == "r1"


def test_schema_version_tag(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    store.log_run_start(_make_run())
    rec = json.loads((tmp_path / "runs.jsonl").read_text().strip().splitlines()[0])
    assert rec.get("_schema") == 1


def _make_episode_custom(run_id: str) -> EpisodeRecord:
    ep = _make_episode()
    return EpisodeRecord(
        run_id=run_id,
        iteration_num=ep.iteration_num,
        episode_id=ep.episode_id,
        task_id=ep.task_id,
        bench=ep.bench,
        model=ep.model,
        reward=ep.reward,
        signals=ep.signals,
        n_steps=ep.n_steps,
        n_tool_calls=ep.n_tool_calls,
        token_usage=ep.token_usage,
        latency_ms=ep.latency_ms,
        messages_ref=ep.messages_ref,
        created_at=ep.created_at,
    )


def test_log_episodes_rejects_unsafe_run_id(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    with pytest.raises(ValueError):
        store.log_episodes([_make_episode_custom("../escape")])
    with pytest.raises(ValueError):
        store.log_episodes([_make_episode_custom("sub/dir")])


def test_get_run_handles_orphan_complete(tmp_path: Path) -> None:
    """run_complete without a prior run_start should not crash get_run."""
    store = JsonlArchiveStore(tmp_path)
    store.log_run_complete("ghost", 0.5, 0.1, 0)
    assert store.get_run("ghost") is None


def test_log_episodes_rejects_mixed_run_batch(tmp_path: Path) -> None:
    store = JsonlArchiveStore(tmp_path)
    ep_a = _make_episode(ep_id="ea")
    ep_b = _make_episode_custom("other-run")
    with pytest.raises(ValueError, match="share a run_id"):
        store.log_episodes([ep_a, ep_b])
