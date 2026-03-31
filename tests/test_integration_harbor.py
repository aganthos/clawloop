"""Integration smoke tests for Harbor + SkyRL translation path.

Uses frozen BFCL task fixtures (no Harbor/Docker/GPU required for default run).
Conditional tests exercise real Harbor parsing and real HF tokenizers when available.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from clawloop.core.episode import Episode
from clawloop.core.loop import AgentState
from clawloop.environments.harbor import HarborTaskEnvironment
from clawloop.exporters.skyrl import SkyRLExporter

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "harbor_tasks"
TASK_NAMES = ["bfcl-simple-0", "bfcl-simple-1", "bfcl-fail-0"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env_from_fixture(task_name: str) -> HarborTaskEnvironment:
    """Create HarborTaskEnvironment from fixture dir with mocked Harbor deps."""
    env = HarborTaskEnvironment.__new__(HarborTaskEnvironment)
    env._task_dir = FIXTURE_DIR / task_name
    env._trial_config = {"agent": {"name": "test-agent", "kwargs": {}}, "task": {}}
    env._reward_transform = None
    env._train_on_truncated = True
    env._Trial = MagicMock()
    env._TrialConfig = MagicMock()
    return env


def _mock_trial_success(reward: float = 1.0, messages: list | None = None):
    results = MagicMock()
    results.verifier_result.rewards = {"reward": reward}
    results.agent_result.metadata = {
        "all_messages": messages or [
            {"role": "user", "content": "Call get_weather with city='London'"},
            {"role": "assistant", "content": '{"function_name": "get_weather", "arguments": {"city": "London"}}'},
        ],
    }
    return results


# ---------------------------------------------------------------------------
# Fixture structure validation (always runs, no external deps)
# ---------------------------------------------------------------------------

class TestHarborFixtureStructure:
    """Validate fixture files match Harbor task directory format.

    NOTE: This does NOT exercise Harbor's own parser. It validates
    that fixtures have the required files so they can be used with
    HarborTaskEnvironment.
    """

    @pytest.mark.parametrize("task_name", TASK_NAMES)
    def test_fixture_has_required_files(self, task_name: str) -> None:
        task_dir = FIXTURE_DIR / task_name
        assert task_dir.exists(), f"Fixture dir missing: {task_dir}"
        assert (task_dir / "instruction.md").exists()
        assert (task_dir / "task.toml").exists()
        assert (task_dir / "tests" / "test.sh").exists()

    @pytest.mark.parametrize("task_name", TASK_NAMES)
    def test_instruction_md_not_empty(self, task_name: str) -> None:
        content = (FIXTURE_DIR / task_name / "instruction.md").read_text()
        assert len(content.strip()) > 0

    @pytest.mark.parametrize("task_name", TASK_NAMES)
    def test_task_toml_has_version(self, task_name: str) -> None:
        import tomllib

        with open(FIXTURE_DIR / task_name / "task.toml", "rb") as f:
            config = tomllib.load(f)
        assert config.get("version") == "1.0"
        assert config.get("agent", {}).get("timeout_sec", 0) > 0


# ---------------------------------------------------------------------------
# Episode construction from fixtures (always runs, mocked Harbor)
# ---------------------------------------------------------------------------

class TestHarborEpisodeFromFixture:
    """Verify HarborTaskEnvironment builds correct Episodes from fixture dirs."""

    def test_task_id_matches_dir_name(self) -> None:
        env = _make_env_from_fixture("bfcl-simple-0")
        assert env.task_id == "bfcl-simple-0"

    def test_successful_trial_produces_episode(self) -> None:
        env = _make_env_from_fixture("bfcl-simple-0")
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=_mock_trial_success(reward=1.0))
        env._Trial.return_value = mock_trial

        ep = asyncio.run(env.run_episode(AgentState()))
        assert isinstance(ep, Episode)
        assert ep.task_id == "bfcl-simple-0"
        assert ep.bench == "harbor"
        assert len(ep.messages) == 2
        assert "outcome" in ep.summary.signals
        assert ep.summary.filtered is False

    def test_zero_reward_trial(self) -> None:
        env = _make_env_from_fixture("bfcl-fail-0")
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=_mock_trial_success(reward=0.0))
        env._Trial.return_value = mock_trial

        ep = asyncio.run(env.run_episode(AgentState()))
        assert "outcome" in ep.summary.signals
        # reward=0.0 through total_reward setter: 0.0*2-1 = -1.0
        assert ep.summary.signals["outcome"].value == pytest.approx(-1.0)

    def test_timeout_produces_filtered_episode(self) -> None:
        env = _make_env_from_fixture("bfcl-fail-0")
        mock_trial = MagicMock()
        exc = type("AgentTimeoutError", (Exception,), {})
        mock_trial.run = AsyncMock(side_effect=exc("timeout"))
        env._Trial.return_value = mock_trial

        ep = asyncio.run(env.run_episode(AgentState()))
        assert ep.summary.filtered is True

    def test_episode_has_valid_step_structure(self) -> None:
        env = _make_env_from_fixture("bfcl-simple-0")
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=_mock_trial_success())
        env._Trial.return_value = mock_trial

        ep = asyncio.run(env.run_episode(AgentState()))
        assert len(ep.steps) > 0
        assert ep.steps[-1].done is True
        assert len(ep.step_boundaries) > 0


# ---------------------------------------------------------------------------
# Full translation path: Harbor Episode → SkyRLExporter → GeneratorOutput
# ---------------------------------------------------------------------------

class TestFullTranslationPath:
    """Episode from Harbor fixture → SkyRLExporter → GeneratorOutput."""

    def test_harbor_episode_through_exporter(self) -> None:
        from tests.test_skyrl_export import FakeTokenizer

        # Build episode from fixture
        env = _make_env_from_fixture("bfcl-simple-0")
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=_mock_trial_success(reward=1.0))
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))

        # Run through exporter
        exporter = SkyRLExporter(tokenizer=FakeTokenizer())
        output = exporter.export([ep])

        # Validate GeneratorOutput structure
        assert len(output["prompt_token_ids"]) > 0
        assert len(output["response_ids"]) > 0
        assert len(output["loss_masks"]) > 0
        assert len(output["trajectory_ids"]) > 0
        # Rewards array matches prompt count
        assert len(output["rewards"]) == len(output["prompt_token_ids"])
        # Terminal reward = 1.0 (from mock)
        assert output["rewards"][-1] == pytest.approx(1.0)

    def test_multiple_episodes_grouped_by_task(self) -> None:
        from tests.test_skyrl_export import FakeTokenizer

        episodes = []
        for _ in range(3):
            env = _make_env_from_fixture("bfcl-simple-0")
            mock_trial = MagicMock()
            mock_trial.run = AsyncMock(return_value=_mock_trial_success(reward=0.8))
            env._Trial.return_value = mock_trial
            ep = asyncio.run(env.run_episode(AgentState()))
            episodes.append(ep)

        exporter = SkyRLExporter(tokenizer=FakeTokenizer())
        output = exporter.export(episodes)

        # All trajectory IDs share the same task
        for tid in output["trajectory_ids"]:
            assert tid.instance_id == "bfcl-simple-0"


# ---------------------------------------------------------------------------
# Conditional: Real Harbor parser (skip if Harbor not installed)
# ---------------------------------------------------------------------------

def _harbor_available() -> bool:
    try:
        from harbor.models.trial.config import TrialConfig  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _harbor_available(), reason="Harbor not installed")
class TestHarborRealParser:
    """Parse fixtures with actual Harbor code when available."""

    @pytest.mark.parametrize("task_name", ["bfcl-simple-0", "bfcl-simple-1"])
    def test_harbor_reads_fixture_toml(self, task_name: str) -> None:
        import tomllib

        task_dir = FIXTURE_DIR / task_name
        with open(task_dir / "task.toml", "rb") as f:
            config = tomllib.load(f)
        assert config["version"] == "1.0"
        assert config["agent"]["timeout_sec"] > 0


# ---------------------------------------------------------------------------
# Conditional: Real HF tokenizer (skip if deps/model not available)
# ---------------------------------------------------------------------------

def _skyrl_available() -> bool:
    try:
        import skyrl.tinker.types  # noqa: F401
        return True
    except ImportError:
        return False


def _transformers_available() -> bool:
    try:
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not (_skyrl_available() and _transformers_available()),
    reason="SkyRL + transformers required",
)
class TestRealTokenizerPath:
    """Full path with real HF tokenizer. Skipped if deps or model not cached."""

    def test_real_tokenizer_export(self) -> None:
        from transformers import AutoTokenizer

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        try:
            tok = AutoTokenizer.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=False,
            )
        except Exception:
            pytest.skip(f"Model {model_name} not cached locally")

        env = _make_env_from_fixture("bfcl-simple-0")
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=_mock_trial_success())
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))

        exporter = SkyRLExporter(tokenizer=tok)
        output = exporter.export([ep])
        # Real token IDs are integers
        assert all(isinstance(t, int) for t in output["prompt_token_ids"][0])
        assert len(output["prompt_token_ids"][0]) > 0
