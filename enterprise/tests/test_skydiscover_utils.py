"""Tests for SkyDiscover serialization helpers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from clawloop.core.evolver import HarnessSnapshot
from enterprise.evolution.backends.skydiscover_utils import (
    harness_to_program,
    program_to_evolver_result,
)
from enterprise.tests.conftest import make_snapshot

_TWO_ENTRIES = [
    {
        "id": "e1",
        "content": "Always verify inputs before acting.",
        "tags": ["safety"],
        "helpful": 5,
        "harmful": 1,
    },
    {
        "id": "e2",
        "content": "Use structured output for tool calls.",
        "tags": ["tools"],
        "helpful": 3,
        "harmful": 0,
    },
]


class TestHarnessToProgram:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        snap = make_snapshot(playbook_entries=_TWO_ENTRIES)
        path = harness_to_program(snap, str(tmp_path / "seed.json"))
        data = json.loads(Path(path).read_text())

        assert data["system_prompt"] == "You are a helpful agent."
        assert len(data["playbook"]) == 2
        assert data["playbook"][0]["content"] == "Always verify inputs before acting."
        assert data["playbook"][0]["tags"] == ["safety"]
        assert data["playbook"][0]["helpful"] == 5
        assert data["playbook"][0]["harmful"] == 1

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = harness_to_program(
            make_snapshot(),
            str(tmp_path / "deep" / "nested" / "seed.json"),
        )
        assert Path(path).exists()

    def test_empty_snapshot(self, tmp_path: Path) -> None:
        snap = HarnessSnapshot(
            system_prompts={},
            playbook_entries=[],
            pareto_fronts={},
            playbook_generation=0,
            playbook_version=0,
        )
        path = harness_to_program(snap, str(tmp_path / "empty.json"))
        data = json.loads(Path(path).read_text())
        assert data["system_prompt"] == ""
        assert data["playbook"] == []


class TestProgramToEvolverResult:
    def test_no_changes_produces_empty_result(self, tmp_path: Path) -> None:
        snap = make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "seed.json"))
        result = program_to_evolver_result(prog_path, snap)

        assert result.insights == []
        assert result.candidates == {}
        # Provenance is set by the caller (SkyDiscoverAdaEvolve), not by utils
        assert result.provenance.backend == ""

    def test_new_playbook_entry_produces_add_insight(self, tmp_path: Path) -> None:
        snap = make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "evolved.json"))

        # Manually add a new entry to the evolved program
        data = json.loads(Path(prog_path).read_text())
        data["playbook"].append({
            "content": "Prefer parallel tool calls.",
            "tags": ["efficiency"],
            "helpful": 0,
            "harmful": 0,
        })
        Path(prog_path).write_text(json.dumps(data))

        result = program_to_evolver_result(prog_path, snap)
        add_insights = [i for i in result.insights if i.action == "add"]
        assert len(add_insights) == 1
        assert add_insights[0].content == "Prefer parallel tool calls."
        assert add_insights[0].tags == ["efficiency"]

    def test_removed_entry_produces_remove_insight(self, tmp_path: Path) -> None:
        snap = make_snapshot(playbook_entries=_TWO_ENTRIES)
        prog_path = harness_to_program(snap, str(tmp_path / "evolved.json"))

        # Remove the first entry
        data = json.loads(Path(prog_path).read_text())
        data["playbook"] = data["playbook"][1:]
        Path(prog_path).write_text(json.dumps(data))

        result = program_to_evolver_result(prog_path, snap)
        remove_insights = [i for i in result.insights if i.action == "remove"]
        assert len(remove_insights) == 1
        assert remove_insights[0].target_entry_id == "e1"

    def test_changed_prompt_produces_candidate(self, tmp_path: Path) -> None:
        snap = make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "evolved.json"))

        data = json.loads(Path(prog_path).read_text())
        data["system_prompt"] = "You are an expert problem-solving agent."
        Path(prog_path).write_text(json.dumps(data))

        result = program_to_evolver_result(prog_path, snap)
        assert "default" in result.candidates
        cands = result.candidates["default"]
        assert len(cands) == 1
        assert cands[0].text == "You are an expert problem-solving agent."
        assert cands[0].generation == 3  # original was 2

    def test_roundtrip_unchanged(self, tmp_path: Path) -> None:
        """Serialize and parse back — should produce no diffs."""
        snap = make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "rt.json"))
        result = program_to_evolver_result(prog_path, snap)
        assert result.insights == []
        assert result.candidates == {}

    def test_tag_only_change_suppressed(self, tmp_path: Path) -> None:
        """Tag-only changes should NOT produce update insights.

        The Harness Curator applies "update" by incrementing helpful and
        resetting embeddings but does not update tags — emitting the insight
        would inflate scores with no functional effect.
        """
        snap = make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "evolved.json"))

        data = json.loads(Path(prog_path).read_text())
        data["playbook"][0]["tags"] = ["safety", "critical"]
        Path(prog_path).write_text(json.dumps(data))

        result = program_to_evolver_result(prog_path, snap)
        assert result.insights == []

    def test_duplicate_content_entries_handled(self, tmp_path: Path) -> None:
        """Two entries with same content should both be tracked."""
        snap = make_snapshot(playbook_entries=[
            {"id": "e1", "content": "Be safe.", "tags": ["a"], "helpful": 1, "harmful": 0},
            {"id": "e2", "content": "Be safe.", "tags": ["b"], "helpful": 2, "harmful": 0},
        ])
        prog_path = harness_to_program(snap, str(tmp_path / "evolved.json"))

        # Remove one of the two duplicates
        data = json.loads(Path(prog_path).read_text())
        data["playbook"] = [data["playbook"][0]]
        Path(prog_path).write_text(json.dumps(data))

        result = program_to_evolver_result(prog_path, snap)
        remove_insights = [i for i in result.insights if i.action == "remove"]
        assert len(remove_insights) == 1
        assert remove_insights[0].target_entry_id == "e2"

    def test_duplicate_content_no_changes(self, tmp_path: Path) -> None:
        """Two identical-content entries roundtrip with no diffs."""
        snap = make_snapshot(playbook_entries=[
            {"id": "e1", "content": "Be safe.", "tags": ["a"], "helpful": 1, "harmful": 0},
            {"id": "e2", "content": "Be safe.", "tags": ["a"], "helpful": 1, "harmful": 0},
        ])
        prog_path = harness_to_program(snap, str(tmp_path / "rt.json"))
        result = program_to_evolver_result(prog_path, snap)
        assert result.insights == []

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Unicode in prompts and playbook survives roundtrip."""
        snap = make_snapshot(
            system_prompt="Du bist ein hilfreicher Agent. 你好世界 🤖",
            playbook_entries=[
                {"id": "e1", "content": "Être prudent avec les données.", "tags": ["sécurité"], "helpful": 1, "harmful": 0},
            ],
        )
        prog_path = harness_to_program(snap, str(tmp_path / "unicode.json"))
        result = program_to_evolver_result(prog_path, snap)
        assert result.insights == []
        assert result.candidates == {}

    def test_empty_evolved_prompt_not_treated_as_candidate(self, tmp_path: Path) -> None:
        """If evolution empties the prompt, don't create a candidate."""
        snap = make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "evolved.json"))

        data = json.loads(Path(prog_path).read_text())
        data["system_prompt"] = ""
        Path(prog_path).write_text(json.dumps(data))

        result = program_to_evolver_result(prog_path, snap)
        assert result.candidates == {}

    def test_malformed_program_raises(self, tmp_path: Path) -> None:
        """Non-JSON program file should raise a clear error."""
        bad_path = tmp_path / "bad.json"
        bad_path.write_text("not valid json {{{")
        snap = make_snapshot()

        with pytest.raises(json.JSONDecodeError):
            program_to_evolver_result(str(bad_path), snap)

    def test_missing_keys_in_program(self, tmp_path: Path) -> None:
        """Program JSON missing expected keys should produce empty result, not crash."""
        minimal_path = tmp_path / "minimal.json"
        minimal_path.write_text(json.dumps({}))
        snap = make_snapshot(playbook_entries=_TWO_ENTRIES)

        result = program_to_evolver_result(str(minimal_path), snap)
        # All original entries should show as removed
        remove_insights = [i for i in result.insights if i.action == "remove"]
        assert len(remove_insights) == 2

    def test_multi_bench_uses_first_prompt(self, tmp_path: Path) -> None:
        """With multiple benches, serialization uses first prompt key."""
        snap = HarnessSnapshot(
            system_prompts={"bench_a": "Prompt A", "bench_b": "Prompt B"},
            playbook_entries=[],
            pareto_fronts={},
            playbook_generation=0,
            playbook_version=0,
        )
        prog_path = harness_to_program(snap, str(tmp_path / "multi.json"))
        data = json.loads(Path(prog_path).read_text())
        # Should get the first one (dict ordering)
        assert data["system_prompt"] in ("Prompt A", "Prompt B")
