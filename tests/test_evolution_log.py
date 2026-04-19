"""Tests for EvolutionLog — append-only JSONL tracking of (state, action, reward_delta)."""

import json
import tempfile
from pathlib import Path

from clawloop.core.evolution_log import EvolutionEntry, EvolutionLog


def test_evolution_entry_to_dict():
    entry = EvolutionEntry(
        iteration=3,
        state_hash_before="abc123",
        state_hash_after="def456",
        actions=["add_insight", "mutate_prompt"],
        reward_before=0.45,
        reward_after=0.62,
        backend="local",
    )
    d = entry.to_dict()
    assert d["iteration"] == 3
    assert d["state_hash_before"] == "abc123"
    assert d["reward_delta"] == 0.62 - 0.45
    assert d["backend"] == "local"
    assert len(d["actions"]) == 2


def test_evolution_entry_reward_delta():
    entry = EvolutionEntry(
        iteration=0,
        state_hash_before="a",
        state_hash_after="b",
        actions=[],
        reward_before=0.5,
        reward_after=0.3,
    )
    assert entry.reward_delta() == -0.2


def test_evolution_log_writes_jsonl():
    with tempfile.TemporaryDirectory() as tmpdir:
        log = EvolutionLog(output_dir=tmpdir)
        log.append(
            EvolutionEntry(
                iteration=0,
                state_hash_before="s0",
                state_hash_after="s1",
                actions=["reflect"],
                reward_before=0.0,
                reward_after=0.5,
            )
        )
        log.append(
            EvolutionEntry(
                iteration=1,
                state_hash_before="s1",
                state_hash_after="s2",
                actions=["mutate"],
                reward_before=0.5,
                reward_after=0.7,
            )
        )

        path = Path(tmpdir) / "evolution.jsonl"
        assert path.exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert first["iteration"] == 0
        assert first["state_hash_before"] == "s0"
        assert first["reward_delta"] == 0.5

        second = json.loads(lines[1])
        assert second["iteration"] == 1


def test_evolution_log_none_dir_is_noop():
    log = EvolutionLog(output_dir=None)
    # Should not raise
    log.append(
        EvolutionEntry(
            iteration=0,
            state_hash_before="a",
            state_hash_after="b",
            actions=[],
            reward_before=0.0,
            reward_after=0.0,
        )
    )
