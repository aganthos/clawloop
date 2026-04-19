"""Tests for content hashing utils and archive schema types."""

import time

from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)
from clawloop.utils.content_hash import canonical_hash, canonical_json, sha256_hex


class TestCanonicalJson:
    def test_sorted_keys(self) -> None:
        assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'

    def test_no_whitespace(self) -> None:
        result = canonical_json({"key": "value"})
        assert " " not in result

    def test_deterministic(self) -> None:
        obj = {"z": [3, 2, 1], "a": {"nested": True}}
        assert canonical_json(obj) == canonical_json(obj)

    def test_nested_sort(self) -> None:
        obj = {"b": {"d": 1, "c": 2}, "a": 0}
        result = canonical_json(obj)
        assert result == '{"a":0,"b":{"c":2,"d":1}}'


class TestCanonicalHash:
    def test_returns_64_char_hex(self) -> None:
        h = canonical_hash({"x": 1})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic(self) -> None:
        obj = {"key": "value", "n": 42}
        assert canonical_hash(obj) == canonical_hash(obj)

    def test_key_order_irrelevant(self) -> None:
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_different_input_different_hash(self) -> None:
        assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})

    def test_sha256_hex_matches(self) -> None:
        obj = {"hello": "world"}
        expected = sha256_hex(canonical_json(obj))
        assert canonical_hash(obj) == expected


# --- Schema roundtrip tests ---

_NOW = time.time()


def _sample_run() -> RunRecord:
    return RunRecord(
        run_id=RunRecord.new_id(),
        bench="swe-bench-lite",
        domain_tags=["coding", "python"],
        agent_config={"model": "gpt-4", "temperature": 0.7},
        config_hash="a" * 64,
        n_iterations=10,
        best_reward=0.85,
        improvement_delta=0.15,
        total_cost_tokens=50000,
        parent_run_id=None,
        created_at=_NOW,
        completed_at=_NOW + 3600,
    )


def _sample_iteration() -> IterationRecord:
    return IterationRecord(
        run_id="abc123",
        iteration_num=3,
        harness_snapshot_hash="b" * 64,
        mean_reward=0.72,
        reward_trajectory=[0.5, 0.6, 0.72],
        evolver_action={"type": "mutate", "target": "system_prompt"},
        cost_tokens=5000,
        parent_variant_hash="c" * 64,
        child_variant_hash="d" * 64,
        reward_delta=0.12,
        created_at=_NOW,
    )


def _sample_episode() -> EpisodeRecord:
    return EpisodeRecord(
        run_id="abc123",
        iteration_num=3,
        episode_id="ep-001",
        task_id="task-42",
        bench="swe-bench-lite",
        model="gpt-4",
        reward=0.9,
        signals={"outcome": 0.9, "execution": 0.8},
        n_steps=5,
        n_tool_calls=3,
        token_usage={"prompt": 1000, "completion": 500},
        latency_ms=2500,
        messages_ref="s3://bucket/ep-001.jsonl",
        created_at=_NOW,
    )


def _sample_variant() -> AgentVariant:
    return AgentVariant(
        variant_hash="e" * 64,
        system_prompt="You are a helpful coding assistant.",
        playbook_snapshot={"rules": ["be concise"]},
        model="gpt-4",
        tools=["search", "bash", "edit"],
        first_seen_run_id="abc123",
        created_at=_NOW,
    )


class TestRunRecord:
    def test_to_dict_roundtrip(self) -> None:
        rec = _sample_run()
        d = rec.to_dict()
        restored = RunRecord.from_dict(d)
        assert restored == rec

    def test_to_dict_keys(self) -> None:
        rec = _sample_run()
        d = rec.to_dict()
        expected_keys = {
            "run_id",
            "bench",
            "domain_tags",
            "agent_config",
            "config_hash",
            "n_iterations",
            "best_reward",
            "improvement_delta",
            "total_cost_tokens",
            "parent_run_id",
            "created_at",
            "completed_at",
        }
        assert set(d.keys()) == expected_keys


class TestIterationRecord:
    def test_to_dict_roundtrip(self) -> None:
        rec = _sample_iteration()
        d = rec.to_dict()
        restored = IterationRecord.from_dict(d)
        assert restored == rec


class TestEpisodeRecord:
    def test_to_dict_roundtrip(self) -> None:
        rec = _sample_episode()
        d = rec.to_dict()
        restored = EpisodeRecord.from_dict(d)
        assert restored == rec


class TestAgentVariant:
    def test_to_dict_roundtrip(self) -> None:
        rec = _sample_variant()
        d = rec.to_dict()
        restored = AgentVariant.from_dict(d)
        assert restored == rec
