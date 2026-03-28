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


def _make_snapshot(
    system_prompt: str = "You are a helpful agent.",
    playbook_entries: list | None = None,
) -> HarnessSnapshot:
    entries = playbook_entries or [
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
    return HarnessSnapshot(
        system_prompts={"default": system_prompt},
        playbook_entries=entries,
        pareto_fronts={},
        playbook_generation=2,
        playbook_version=4,
    )


class TestHarnessToProgram:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
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
            _make_snapshot(),
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
        snap = _make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "seed.json"))
        result = program_to_evolver_result(prog_path, snap)

        assert result.insights == []
        assert result.candidates == {}
        assert result.provenance.backend == "skydiscover_adaevolve"

    def test_new_playbook_entry_produces_add_insight(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
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
        snap = _make_snapshot()
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
        snap = _make_snapshot()
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
        snap = _make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "rt.json"))
        result = program_to_evolver_result(prog_path, snap)
        assert result.insights == []
        assert result.candidates == {}

    def test_tag_update_produces_update_insight(self, tmp_path: Path) -> None:
        snap = _make_snapshot()
        prog_path = harness_to_program(snap, str(tmp_path / "evolved.json"))

        data = json.loads(Path(prog_path).read_text())
        data["playbook"][0]["tags"] = ["safety", "critical"]
        Path(prog_path).write_text(json.dumps(data))

        result = program_to_evolver_result(prog_path, snap)
        update_insights = [i for i in result.insights if i.action == "update"]
        assert len(update_insights) == 1
        assert update_insights[0].target_entry_id == "e1"
