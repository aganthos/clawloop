# tests/test_entropic_adapter.py
"""Integration tests for EntropicAdapter with mock green agent output."""

import json
from pathlib import Path

import pytest

from lfx.adapters.entropic import EntropicAdapter
from lfx.core.loop import AgentState


class TestEntropicAdapterResultsParsing:
    """EntropicAdapter parses green agent results into Episodes."""

    def _make_task_result(self, idx="0", crm_reward=1, total_score=80.0, **overrides):
        result = {
            "task_idx": idx,
            "task_category": "knowledge_qa",
            "task_query": "How many leads?",
            "crm_reward": crm_reward,
            "total_score": total_score,
            "dimension_scores": {
                "FUNCTIONAL": 100.0,
                "DRIFT_ADAPTATION": 80.0,
                "TOKEN_EFFICIENCY": 60.0,
                "QUERY_EFFICIENCY": 70.0,
                "ERROR_RECOVERY": 100.0,
                "TRAJECTORY_EFFICIENCY": 90.0,
                "HALLUCINATION_RATE": 100.0,
            },
            "agent_answer": "There are 42 leads.",
            "success": crm_reward > 0,
            "timing": {"total_seconds": 1.5, "purple_agent_seconds": 1.0},
        }
        result.update(overrides)
        return result

    def test_maps_results_to_episodes(self, tmp_path):
        adapter = EntropicAdapter()
        adapter._model = "test-model"
        adapter._output_dir = tmp_path

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({
            "results": [self._make_task_result("0")],
            "summary": {"pass_rate": 1.0, "total_tasks": 1},
        }))

        episodes = adapter._parse_results(results_path, ["0"])
        assert len(episodes) == 1
        assert episodes[0].task_id == "entropic:0"
        assert episodes[0].bench == "entropic"
        assert episodes[0].summary.signals["outcome"].value == 1.0
        assert "functional" in episodes[0].summary.signals

    def test_maps_uppercase_dimension_keys(self, tmp_path):
        """Green agent returns uppercase dimension keys (FUNCTIONAL, etc.)."""
        adapter = EntropicAdapter()
        adapter._model = "test"

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({
            "results": [self._make_task_result("0")],
        }))

        episodes = adapter._parse_results(results_path, ["0"])
        assert "functional" in episodes[0].summary.signals
        assert "drift_adaptation" in episodes[0].summary.signals

    def test_missing_task_creates_failed_episode(self, tmp_path):
        adapter = EntropicAdapter()
        adapter._model = "test"

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({"results": []}))

        episodes = adapter._parse_results(results_path, ["0", "1"])
        assert len(episodes) == 2
        assert all(ep.summary.signals["outcome"].value == -1.0 for ep in episodes)

    def test_parse_error_returns_failed_episodes(self, tmp_path):
        adapter = EntropicAdapter()
        adapter._model = "test"

        episodes = adapter._parse_results(tmp_path / "nonexistent.json", ["0"])
        assert len(episodes) == 1
        assert episodes[0].metadata["error"] == "parse_error"

    def test_crm_reward_zero_maps_to_negative_outcome(self, tmp_path):
        adapter = EntropicAdapter()
        adapter._model = "test"

        results_path = tmp_path / "results.json"
        results_path.write_text(json.dumps({
            "results": [self._make_task_result("0", crm_reward=0, total_score=20.0)],
        }))

        episodes = adapter._parse_results(results_path, ["0"])
        assert episodes[0].summary.signals["outcome"].value == -1.0


class TestEvalConfigGeneration:
    """_build_eval_config produces correct EvalRequest structure."""

    def test_generates_valid_config(self):
        adapter = EntropicAdapter()
        adapter._task_categories = None
        adapter._task_limit = None
        adapter._task_ids = None

        config = adapter._build_eval_config(["0", "1"], purple_port=9999)
        assert config["participants"]["agent"] == "http://127.0.0.1:9999"
        assert config["config"]["task_ids"] == ["0", "1"]
        assert config["config"]["skip_original"] is True

    def test_explicit_task_ids_override(self):
        adapter = EntropicAdapter()
        adapter._task_categories = None
        adapter._task_limit = None
        adapter._task_ids = ["500", "501"]

        config = adapter._build_eval_config(["base_0", "base_1"], purple_port=9999)
        assert config["config"]["task_ids"] == ["500", "501"]

    def test_includes_category_filter(self):
        adapter = EntropicAdapter()
        adapter._task_categories = ["knowledge_qa", "lead_qualification"]
        adapter._task_limit = 5
        adapter._task_ids = None

        config = adapter._build_eval_config(["0"], purple_port=8080)
        assert config["config"]["task_categories"] == ["knowledge_qa", "lead_qualification"]
        assert config["config"]["task_limit"] == 5


class TestEpisodeMapping:
    """_map_to_episode converts green agent output to lfx Episodes."""

    def test_maps_task_result(self):
        adapter = EntropicAdapter()
        adapter._model = "test-model"
        adapter._current_state_id = "abc123"

        episode = adapter._map_to_episode({
            "task_idx": "42",
            "task_category": "handle_time",
            "task_query": "Average handle time?",
            "agent_answer": "15 minutes",
            "crm_reward": 1,
            "total_score": 75.0,
            "dimension_scores": {"FUNCTIONAL": 100.0, "DRIFT_ADAPTATION": 60.0},
            "success": True,
            "timing": {"total_seconds": 2.0},
        })

        assert episode.task_id == "entropic:42"
        assert episode.bench == "entropic"
        assert episode.metadata["entropic_category"] == "handle_time"
        assert episode.metadata["entropic_total_score"] == 75.0
        assert episode.steps[0].reward == 0.75
        assert len(episode.messages) == 2

    def test_failed_crm_reward(self):
        adapter = EntropicAdapter()
        adapter._model = "test"

        episode = adapter._map_to_episode({
            "task_idx": "0",
            "crm_reward": 0,
            "total_score": 20.0,
            "dimension_scores": {},
        })

        assert episode.summary.signals["outcome"].value == -1.0
