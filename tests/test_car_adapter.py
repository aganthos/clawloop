# tests/test_car_adapter.py
"""Integration tests for CARAdapter with mock agentbeats-run."""

import json
import os
import stat
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from lfx.adapters.car import CARAdapter
from lfx.core.loop import AgentState


class TestCARAdapterResultsParsing:
    """CARAdapter parses results.json into Episodes."""

    def test_maps_results_to_episodes(self, tmp_path):
        adapter = CARAdapter()
        adapter._car_bench_path = tmp_path
        adapter._output_dir = tmp_path / "output"
        adapter._task_type = "base"
        adapter._task_split = "test"
        adapter._model = "test-model"
        adapter._iteration_count = 0
        adapter._agentbeats_cmd = "echo"

        # Write canned results directly
        iter_dir = tmp_path / "output" / "iter_0"
        iter_dir.mkdir(parents=True)
        results_path = iter_dir / "results.json"
        results_path.write_text(json.dumps({
            "detailed_results_by_split": {
                "base": [
                    {
                        "task_id": "base_0",
                        "reward": 1.0,
                        "reward_info": {"r_actions_final": 1.0},
                        "trajectory": [{"role": "user", "content": "Hi"}],
                        "total_agent_cost": 0.01,
                        "total_llm_latency_ms": 500.0,
                    }
                ]
            }
        }))

        episodes = adapter._parse_results(results_path, ["base_0"])
        assert len(episodes) == 1
        assert episodes[0].task_id == "car:base_0"
        assert episodes[0].bench == "car"
        assert episodes[0].summary.signals["outcome"].value == 1.0

    def test_parses_nested_results_format(self, tmp_path):
        """agentbeats-run wraps detailed_results_by_split inside results[0]."""
        adapter = CARAdapter()
        adapter._model = "test"
        adapter._output_dir = tmp_path

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({
            "participants": {},
            "results": [{
                "score": 1.0,
                "detailed_results_by_split": {
                    "base": [{
                        "task_id": "base_0",
                        "reward": 1.0,
                        "reward_info": {"r_actions_final": 1.0},
                        "trajectory": [{"role": "user", "content": "Hi"}],
                        "total_agent_cost": 0.01,
                        "total_llm_latency_ms": 500.0,
                    }]
                }
            }]
        }))

        episodes = adapter._parse_results(results_path, ["base_0"])
        assert len(episodes) == 1
        assert episodes[0].task_id == "car:base_0"
        assert episodes[0].summary.signals["outcome"].value == 1.0

    def test_missing_task_creates_failed_episode(self, tmp_path):
        adapter = CARAdapter()
        adapter._model = "test"
        adapter._output_dir = tmp_path
        adapter._iteration_count = 0

        iter_dir = tmp_path / "iter_0"
        iter_dir.mkdir(parents=True)
        results_path = iter_dir / "results.json"
        results_path.write_text(json.dumps({
            "detailed_results_by_split": {"base": []}
        }))

        episodes = adapter._parse_results(results_path, ["base_0", "base_1"])
        # Should have 2 failed episodes for missing tasks
        assert len(episodes) == 2
        assert all(ep.summary.signals["outcome"].value == -1.0 for ep in episodes)


class TestScenarioGeneration:
    """_generate_scenario produces valid TOML."""

    def test_generates_valid_scenario(self, tmp_path):
        adapter = CARAdapter()
        adapter._task_split = "test"
        adapter._car_bench_path = tmp_path
        adapter._model = "test-model"

        harness_file = str(tmp_path / "harness.json")
        scenario = adapter._generate_scenario(
            ["base_0", "base_2"], harness_file,
            green_port=8081, purple_port=9999,
        )
        assert "task_split" in scenario
        assert '"test"' in scenario
        assert "base_0" in scenario
        assert "9999" in scenario
        assert harness_file in scenario
        # Unused types zeroed out
        assert "tasks_hallucination_num_tasks = 0" in scenario
        assert "tasks_disambiguation_num_tasks = 0" in scenario

    def test_mixed_task_types(self, tmp_path):
        adapter = CARAdapter()
        adapter._task_split = "test"
        adapter._car_bench_path = tmp_path
        adapter._model = "test-model"

        harness_file = str(tmp_path / "harness.json")
        scenario = adapter._generate_scenario(
            ["base_0", "hallucination_1"], harness_file,
            green_port=8081, purple_port=9999,
        )
        assert "base_0" in scenario
        assert "hallucination_1" in scenario
