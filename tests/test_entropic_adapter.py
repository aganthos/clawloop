# tests/test_entropic_adapter.py
"""Integration tests for EntropicAdapter with mock green agent."""

import json
from pathlib import Path

import pytest

from lfx.adapters.entropic import EntropicAdapter
from lfx.core.loop import AgentState


class TestEntropicAdapterResultsParsing:
    """EntropicAdapter parses results.json into Episodes."""

    def test_maps_results_to_episodes(self, tmp_path):
        adapter = EntropicAdapter()
        adapter._bench_path = tmp_path
        adapter._output_dir = tmp_path / "output"
        adapter._model = "test-model"
        adapter._iteration_count = 0

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({
            "tasks": [
                {
                    "task_id": "task_0",
                    "category": "knowledge_qa",
                    "scores": {
                        "functional": 100.0,
                        "drift_adaptation": 80.0,
                        "token_efficiency": 60.0,
                        "query_efficiency": 70.0,
                        "error_recovery": 100.0,
                        "trajectory_efficiency": 90.0,
                        "hallucination_rate": 100.0,
                    },
                    "total_score": 85.6,
                    "trajectory": [
                        {"role": "user", "content": "How many leads?"},
                        {"role": "assistant", "content": "There are 42 leads."},
                    ],
                }
            ]
        }))

        episodes = adapter._parse_results(results_path, ["task_0"])
        assert len(episodes) == 1
        assert episodes[0].task_id == "entropic:task_0"
        assert episodes[0].bench == "entropic"
        assert episodes[0].summary.signals["outcome"].value == 1.0
        assert "functional" in episodes[0].summary.signals

    def test_parses_nested_results_format(self, tmp_path):
        """Results wrapped inside results[0].tasks."""
        adapter = EntropicAdapter()
        adapter._model = "test"
        adapter._output_dir = tmp_path

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({
            "results": [{
                "tasks": [{
                    "task_id": "task_0",
                    "scores": {"functional": 100.0},
                    "total_score": 100.0,
                    "trajectory": [{"role": "user", "content": "Hi"}],
                }]
            }]
        }))

        episodes = adapter._parse_results(results_path, ["task_0"])
        assert len(episodes) == 1
        assert episodes[0].task_id == "entropic:task_0"
        assert episodes[0].summary.signals["outcome"].value == 1.0

    def test_missing_task_creates_failed_episode(self, tmp_path):
        adapter = EntropicAdapter()
        adapter._model = "test"
        adapter._output_dir = tmp_path
        adapter._iteration_count = 0

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({"tasks": []}))

        episodes = adapter._parse_results(results_path, ["task_0", "task_1"])
        assert len(episodes) == 2
        assert all(ep.summary.signals["outcome"].value == -1.0 for ep in episodes)

    def test_parse_error_returns_failed_episodes(self, tmp_path):
        adapter = EntropicAdapter()
        adapter._model = "test"

        results_path = tmp_path / "nonexistent.json"
        episodes = adapter._parse_results(results_path, ["task_0"])
        assert len(episodes) == 1
        assert episodes[0].metadata["error"] == "parse_error"


class TestEvalConfigGeneration:
    """_build_eval_config produces correct eval config."""

    def test_generates_valid_config(self):
        adapter = EntropicAdapter()
        adapter._task_categories = None
        adapter._task_limit = None

        config = adapter._build_eval_config(["task_0", "task_1"], purple_port=9999)
        assert config["participants"]["agent"] == "http://127.0.0.1:9999"
        assert config["task_ids"] == ["task_0", "task_1"]
        assert config["org_type"] == "b2b"
        assert config["drift_level"] == "medium"
        assert config["rot_level"] == "medium"

    def test_includes_category_filter(self):
        adapter = EntropicAdapter()
        adapter._task_categories = ["knowledge_qa", "lead_qualification"]
        adapter._task_limit = 5

        config = adapter._build_eval_config(["task_0"], purple_port=8080)
        assert config["task_categories"] == ["knowledge_qa", "lead_qualification"]
        assert config["task_limit"] == 5


class TestEpisodeMapping:
    """_map_to_episode converts entropic results to lfx Episodes."""

    def test_maps_task_result(self):
        adapter = EntropicAdapter()
        adapter._model = "test-model"
        adapter._current_state_id = "abc123"

        episode = adapter._map_to_episode({
            "task_id": "task_42",
            "category": "handle_time",
            "scores": {"functional": 100.0, "drift_adaptation": 60.0},
            "total_score": 75.0,
            "trajectory": [{"role": "user", "content": "query"}],
            "drift_level": "medium",
            "rot_level": "medium",
        })

        assert episode.task_id == "entropic:task_42"
        assert episode.bench == "entropic"
        assert episode.metadata["entropic_category"] == "handle_time"
        assert episode.metadata["entropic_total_score"] == 75.0
        assert episode.steps[0].reward == 0.75  # 75/100

    def test_low_functional_maps_to_negative_outcome(self):
        adapter = EntropicAdapter()
        adapter._model = "test"

        episode = adapter._map_to_episode({
            "task_id": "task_0",
            "scores": {"functional": 30.0},
            "total_score": 20.0,
        })

        assert episode.summary.signals["outcome"].value == -1.0
