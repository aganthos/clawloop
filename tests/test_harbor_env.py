"""Tests for lfx.envs.harbor — HarborTaskEnvironment + HarborAdapter."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lfx.core.episode import Episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(task_dir="/data/tasks/test-task", **kwargs):
    """Create a HarborTaskEnvironment with mocked Harbor."""
    from lfx.envs.harbor import HarborTaskEnvironment

    env = HarborTaskEnvironment.__new__(HarborTaskEnvironment)
    env._task_dir = Path(task_dir)
    env._trial_config = kwargs.get(
        "trial_config", {"agent": {"name": "t2", "kwargs": {}}}
    )
    env._trial_config.setdefault("task", {})
    env._trial_config["agent"].setdefault("kwargs", {})
    env._reward_transform = kwargs.get("reward_transform", None)
    env._train_on_truncated = kwargs.get("train_on_truncated", True)
    env._Trial = MagicMock()
    env._TrialConfig = MagicMock()
    return env


def _make_trial_results(reward=1.0, messages=None, rewards_dict=None):
    """Build a mock trial result object."""
    if messages is None:
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Do the task"},
            {"role": "assistant", "content": "Done"},
        ]
    if rewards_dict is None:
        rewards_dict = {"reward": reward}

    results = MagicMock()
    results.verifier_result.rewards = rewards_dict
    results.agent_result.metadata = {"all_messages": messages}
    return results


def _make_agent_state(inference_url=None, harness=None, state_id_hash="abc123"):
    """Build a mock agent state."""
    state = MagicMock()
    state.inference_url = inference_url
    state.harness = harness

    sid = MagicMock()
    sid.combined_hash = state_id_hash
    state.state_id.return_value = sid
    return state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTaskId:
    def test_task_id_from_dir_name(self):
        env = _make_env(task_dir="/data/tasks/my-special-task")
        assert env.task_id == "my-special-task"

    def test_task_id_nested_path(self):
        env = _make_env(task_dir="/a/b/c/deep-task")
        assert env.task_id == "deep-task"


class TestConfigValidation:
    def test_config_validation_missing_agent(self):
        from lfx.envs.harbor import HarborTaskEnvironment

        with pytest.raises(ValueError, match="trial_config must contain 'agent'"):
            HarborTaskEnvironment(
                task_dir=Path("/tmp/test"),
                trial_config={"task": {}},
            )


class TestRunEpisode:
    def test_run_episode_builds_episode(self):
        env = _make_env()
        trial_results = _make_trial_results(reward=1.0)
        trial_instance = AsyncMock()
        trial_instance.run.return_value = trial_results
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state()
        ep = asyncio.run(env.run_episode(agent_state))

        assert isinstance(ep, Episode)
        assert ep.task_id == "test-task"
        assert len(ep.messages) == 4
        assert ep.summary.signals["outcome"].value == 1.0
        assert ep.summary.filtered is False

    def test_run_episode_injects_session_id(self):
        env = _make_env()
        trial_results = _make_trial_results()
        trial_instance = AsyncMock()
        trial_instance.run.return_value = trial_results
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state()
        asyncio.run(env.run_episode(agent_state))

        # TrialConfig was called with **config
        config_call = env._TrialConfig.call_args[1]
        assert "agent" in config_call
        assert "session_id" in config_call["agent"]["kwargs"]
        assert len(config_call["agent"]["kwargs"]["session_id"]) == 32  # uuid hex

    def test_run_episode_injects_inference_url(self):
        env = _make_env()
        trial_results = _make_trial_results()
        trial_instance = AsyncMock()
        trial_instance.run.return_value = trial_results
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state(inference_url="http://localhost:8000/v1")
        asyncio.run(env.run_episode(agent_state))

        config_call = env._TrialConfig.call_args[1]
        assert config_call["agent"]["kwargs"]["api_base"] == "http://localhost:8000/v1"

    def test_run_episode_injects_system_prompt(self):
        from lfx.core.types import Future, SampleResult

        harness = MagicMock()
        sample_future = Future.immediate(SampleResult(output="Be helpful"))
        harness.sample.return_value = sample_future

        env = _make_env()
        trial_results = _make_trial_results()
        trial_instance = AsyncMock()
        trial_instance.run.return_value = trial_results
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state(harness=harness)
        asyncio.run(env.run_episode(agent_state))

        config_call = env._TrialConfig.call_args[1]
        assert config_call["agent"]["kwargs"]["system_prompt_override"] == "Be helpful"

    def test_state_id_set_on_episode(self):
        env = _make_env()
        trial_results = _make_trial_results()
        trial_instance = AsyncMock()
        trial_instance.run.return_value = trial_results
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state(state_id_hash="deadbeef42")
        ep = asyncio.run(env.run_episode(agent_state))

        assert ep.state_id == "deadbeef42"


class TestExceptionHandling:
    def test_context_exceeded_trainable_by_default(self):
        env = _make_env(train_on_truncated=True)
        trial_instance = AsyncMock()

        exc = type("ContextLengthExceededError", (Exception,), {})("too long")
        trial_instance.run.side_effect = exc
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state()
        ep = asyncio.run(env.run_episode(agent_state))

        assert ep.summary.filtered is False
        assert ep.terminal_reward() == 0.0
        assert ep.metadata.get("truncated") is True

    def test_context_exceeded_filtered_when_configured(self):
        env = _make_env(train_on_truncated=False)
        trial_instance = AsyncMock()

        exc = type("ContextLengthExceededError", (Exception,), {})("too long")
        trial_instance.run.side_effect = exc
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state()
        ep = asyncio.run(env.run_episode(agent_state))

        assert ep.summary.filtered is True
        assert ep.metadata.get("truncated") is True

    def test_agent_timeout_always_filtered(self):
        env = _make_env()
        trial_instance = AsyncMock()

        exc = type("AgentTimeoutError", (Exception,), {})("timed out")
        trial_instance.run.side_effect = exc
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state()
        ep = asyncio.run(env.run_episode(agent_state))

        assert ep.summary.filtered is True
        assert ep.metadata.get("timeout") is True

    def test_generic_error_filtered(self):
        env = _make_env()
        trial_instance = AsyncMock()
        trial_instance.run.side_effect = RuntimeError("something broke")
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state()
        ep = asyncio.run(env.run_episode(agent_state))

        assert ep.summary.filtered is True
        assert ep.metadata.get("error") == "RuntimeError"


class TestRewardTransform:
    def test_reward_transform_applied(self):
        env = _make_env(reward_transform=lambda r: r * 2 - 1)
        trial_results = _make_trial_results(reward=0.75)
        trial_instance = AsyncMock()
        trial_instance.run.return_value = trial_results
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state()
        ep = asyncio.run(env.run_episode(agent_state))

        # 0.75 * 2 - 1 = 0.5
        assert ep.metadata["transformed_reward"] == pytest.approx(0.5)
        assert ep.summary.signals["outcome"].value == pytest.approx(0.5)

    def test_reward_transform_error_falls_back(self):
        def bad_transform(r):
            raise ValueError("oops")

        env = _make_env(reward_transform=bad_transform)
        trial_results = _make_trial_results(reward=0.8)
        trial_instance = AsyncMock()
        trial_instance.run.return_value = trial_results
        env._Trial.return_value = trial_instance

        agent_state = _make_agent_state()
        ep = asyncio.run(env.run_episode(agent_state))

        assert ep.metadata["transformed_reward"] == pytest.approx(0.8)
        assert ep.metadata["reward_transform_error"] is True
        assert ep.summary.signals["outcome"].value == pytest.approx(0.8)


class TestHarborAdapter:
    def test_adapter_run_episode_delegates(self):
        from lfx.envs.harbor import HarborAdapter

        env = _make_env(task_dir="/data/tasks/task-a")
        expected_ep = MagicMock(spec=Episode)

        async def mock_run(agent_state):
            return expected_ep

        env.run_episode = mock_run

        adapter = HarborAdapter([env])
        agent_state = _make_agent_state()
        result = adapter.run_episode("task-a", agent_state)

        assert result is expected_ep

    def test_adapter_run_batch_concurrent(self):
        from lfx.envs.harbor import HarborAdapter

        envs = []
        for name in ["task-a", "task-b", "task-c"]:
            env = _make_env(task_dir=f"/data/tasks/{name}")
            ep = MagicMock(spec=Episode)
            ep.task_id = name

            async def mock_run(agent_state, _ep=ep):
                return _ep

            env.run_episode = mock_run
            envs.append(env)

        adapter = HarborAdapter(envs)
        agent_state = _make_agent_state()
        results = adapter.run_batch(["task-a", "task-b", "task-c"], agent_state, n_per_task=1)

        assert len(results) == 3

    def test_adapter_run_batch_n_per_task(self):
        from lfx.envs.harbor import HarborAdapter

        env = _make_env(task_dir="/data/tasks/task-x")

        call_count = 0

        async def mock_run(agent_state):
            nonlocal call_count
            call_count += 1
            return MagicMock(spec=Episode)

        env.run_episode = mock_run

        adapter = HarborAdapter([env])
        agent_state = _make_agent_state()
        results = adapter.run_batch(["task-x"], agent_state, n_per_task=3)

        assert len(results) == 3
        assert call_count == 3


class TestHelpers:
    def test_compute_step_boundaries_basic(self):
        from lfx.core.episode import Message
        from lfx.envs.harbor import _compute_step_boundaries

        messages = [
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
            Message(role="user", content="do it"),
            Message(role="assistant", content="done"),
        ]
        boundaries = _compute_step_boundaries(messages)
        assert boundaries == [0, 2]

    def test_compute_step_boundaries_empty(self):
        from lfx.envs.harbor import _compute_step_boundaries

        assert _compute_step_boundaries([]) == []

    def test_build_steps_terminal_reward(self):
        from lfx.envs.harbor import _build_steps

        steps = _build_steps([], [0, 2], reward=1.0)
        assert len(steps) == 2
        assert steps[0].reward == 0.0
        assert steps[0].done is False
        assert steps[1].reward == 1.0
        assert steps[1].done is True

    def test_build_steps_empty(self):
        from lfx.envs.harbor import _build_steps

        assert _build_steps([], [], reward=1.0) == []
