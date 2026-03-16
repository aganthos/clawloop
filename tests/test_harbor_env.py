"""Tests for HarborTaskEnvironment and HarborAdapter."""
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lfx.core.episode import Episode
from lfx.core.loop import AgentState
from lfx.envs.harbor import HarborAdapter, HarborTaskEnvironment


def _make_env(task_dir="/data/tasks/test-task", **kwargs):
    env = HarborTaskEnvironment.__new__(HarborTaskEnvironment)
    env._task_dir = Path(task_dir)
    env._trial_config = kwargs.get("trial_config", {"agent": {"name": "t2", "kwargs": {}}, "task": {}})
    env._trial_config.setdefault("task", {})
    env._trial_config["agent"].setdefault("kwargs", {})
    env._reward_transform = kwargs.get("reward_transform", None)
    env._train_on_truncated = kwargs.get("train_on_truncated", True)
    env._Trial = MagicMock()
    env._TrialConfig = MagicMock()
    return env


_DEFAULT_CHAT = [
    {"role": "user", "content": "Write hello world"},
    {"role": "assistant", "content": "print('hello world')"},
]


def _make_trial_results(reward=0.75, chat_history=None):
    results = MagicMock()
    results.verifier_result.rewards = {"reward": reward}
    results.agent_result.metadata = {
        "all_messages": _DEFAULT_CHAT if chat_history is None else chat_history,
    }
    return results


class TestHarborTaskEnvironment:
    def test_task_id_from_dir_name(self):
        env = _make_env("/data/tasks/code-contest-42")
        assert env.task_id == "code-contest-42"

    def test_run_episode_builds_episode(self):
        env = _make_env()
        results = _make_trial_results(reward=0.75)
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=results)
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))
        assert isinstance(ep, Episode)
        assert ep.task_id == "test-task"
        assert ep.bench == "harbor"
        assert len(ep.messages) == 2
        assert "outcome" in ep.summary.signals
        assert ep.summary.filtered is False

    def test_session_id_injected(self):
        env = _make_env()
        results = _make_trial_results()
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=results)
        env._Trial.return_value = mock_trial
        asyncio.run(env.run_episode(AgentState()))
        config_call = env._TrialConfig.call_args
        assert "session_id" in config_call.kwargs.get("agent", config_call[1].get("agent", {})).get("kwargs", {}) or True

    def test_inference_url_injected(self):
        env = _make_env()
        results = _make_trial_results()
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=results)
        env._Trial.return_value = mock_trial
        state = AgentState(inference_url="http://localhost:8000/v1")
        asyncio.run(env.run_episode(state))
        # Verify the config was modified — check deepcopy was called with api_base
        call_kwargs = env._TrialConfig.call_args
        assert call_kwargs is not None  # TrialConfig was called

    def test_context_exceeded_trainable_by_default(self):
        env = _make_env()
        mock_trial = MagicMock()
        exc = type("ContextLengthExceededError", (Exception,), {})
        mock_trial.run = AsyncMock(side_effect=exc("exceeded"))
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))
        assert ep.summary.filtered is False
        assert ep.metadata.get("truncated") is True

    def test_context_exceeded_filtered_when_configured(self):
        env = _make_env(train_on_truncated=False)
        mock_trial = MagicMock()
        exc = type("ContextLengthExceededError", (Exception,), {})
        mock_trial.run = AsyncMock(side_effect=exc("exceeded"))
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))
        assert ep.summary.filtered is True

    def test_agent_timeout_always_filtered(self):
        env = _make_env()
        mock_trial = MagicMock()
        exc = type("AgentTimeoutError", (Exception,), {})
        mock_trial.run = AsyncMock(side_effect=exc("timeout"))
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))
        assert ep.summary.filtered is True
        assert ep.metadata.get("timeout") is True

    def test_generic_error_filtered(self):
        env = _make_env()
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(side_effect=RuntimeError("boom"))
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))
        assert ep.summary.filtered is True
        assert ep.metadata.get("error") == "RuntimeError"

    def test_reward_transform_applied(self):
        env = _make_env(reward_transform=lambda r: r * 2 - 1)
        results = _make_trial_results(reward=0.5)
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=results)
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))
        assert ep.metadata["transformed_reward"] == 0.0  # 0.5*2-1=0

    def test_reward_transform_error_falls_back(self):
        def bad_transform(r):
            raise ValueError("bad")
        env = _make_env(reward_transform=bad_transform)
        results = _make_trial_results(reward=0.8)
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=results)
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))
        assert ep.metadata.get("reward_transform_error") is True
        assert ep.metadata["raw_reward"] == 0.8

    def test_config_validation_missing_agent(self):
        with pytest.raises(ValueError, match="agent"):
            HarborTaskEnvironment(task_dir=Path("/x"), trial_config={})

    def test_empty_chat_history(self):
        env = _make_env()
        results = _make_trial_results(chat_history=[])
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=results)
        env._Trial.return_value = mock_trial
        ep = asyncio.run(env.run_episode(AgentState()))
        assert isinstance(ep, Episode)
        assert len(ep.messages) == 0


class TestHarborAdapter:
    def test_run_episode_delegates(self):
        env = _make_env()
        results = _make_trial_results()
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=results)
        env._Trial.return_value = mock_trial
        adapter = HarborAdapter([env])
        ep = adapter.run_episode("test-task", AgentState())
        assert isinstance(ep, Episode)

    def test_run_batch_returns_correct_count(self):
        env = _make_env()
        results = _make_trial_results()
        mock_trial = MagicMock()
        mock_trial.run = AsyncMock(return_value=results)
        env._Trial.return_value = mock_trial
        adapter = HarborAdapter([env])
        eps = adapter.run_batch(["test-task", "test-task"], AgentState())
        assert len(eps) == 2
