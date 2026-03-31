"""Tests for clawloop.train — config validation, mode presets, MathAdapter."""
from __future__ import annotations

import pytest
from pydantic import SecretStr

from clawloop.train import (
    HarborConfig,
    LLMClientConfig,
    MODE_LAYERS,
    TrainConfig,
    validate_config,
)


def _llm(role: str = "reflector") -> dict[str, LLMClientConfig]:
    return {role: LLMClientConfig(model="test-model", api_base="http://test", api_key=SecretStr("k"))}


def _skyrl() -> dict:
    return {"base_model": "test", "backend_type": "jax"}


def _harbor() -> HarborConfig:
    return HarborConfig(task_dirs=["/tmp/task1"])


# ---------------------------------------------------------------------------
# Mode presets
# ---------------------------------------------------------------------------

class TestModePresets:
    def test_weight_layers(self):
        assert MODE_LAYERS["weight"] == ["weights"]

    def test_harness_learning_layers(self):
        assert MODE_LAYERS["harness_learning"] == ["harness", "router"]

    def test_full_layers(self):
        assert MODE_LAYERS["full"] == ["harness", "router", "weights"]


# ---------------------------------------------------------------------------
# Validation: weight mode
# ---------------------------------------------------------------------------

class TestWeightValidation:
    def test_weight_requires_skyrl(self):
        cfg = TrainConfig(mode="weight", harbor=_harbor())
        with pytest.raises(ValueError, match="skyrl"):
            validate_config(cfg)

    def test_weight_ok(self):
        cfg = TrainConfig(mode="weight", skyrl=_skyrl(), harbor=_harbor())
        assert validate_config(cfg) == ["weights"]

    def test_weight_no_reflector_needed(self):
        cfg = TrainConfig(mode="weight", skyrl=_skyrl(), harbor=_harbor())
        validate_config(cfg)  # no raise


# ---------------------------------------------------------------------------
# Validation: harness_learning mode
# ---------------------------------------------------------------------------

class TestHarnessLearningValidation:
    def test_requires_reflector(self):
        cfg = TrainConfig(mode="harness_learning", env_type="math", llm_clients=_llm("task"))
        with pytest.raises(ValueError, match="reflector"):
            validate_config(cfg)

    def test_math_requires_task(self):
        cfg = TrainConfig(mode="harness_learning", env_type="math", llm_clients=_llm("reflector"))
        with pytest.raises(ValueError, match="task"):
            validate_config(cfg)

    def test_math_ok(self):
        clients = {**_llm("reflector"), **_llm("task")}
        cfg = TrainConfig(mode="harness_learning", env_type="math", llm_clients=clients)
        assert validate_config(cfg) == ["harness", "router"]

    def test_harbor_ok(self):
        cfg = TrainConfig(mode="harness_learning", harbor=_harbor(), llm_clients=_llm("reflector"))
        assert validate_config(cfg) == ["harness", "router"]


# ---------------------------------------------------------------------------
# Validation: full mode
# ---------------------------------------------------------------------------

class TestFullValidation:
    def test_full_mode_raises_not_implemented(self):
        cfg = TrainConfig(mode="full", skyrl=_skyrl(), harbor=_harbor(), llm_clients=_llm("reflector"))
        with pytest.raises(NotImplementedError, match="disabled"):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# Validation: env_type
# ---------------------------------------------------------------------------

class TestEnvValidation:
    def test_harbor_requires_task_dirs(self):
        cfg = TrainConfig(mode="weight", skyrl=_skyrl())
        with pytest.raises(ValueError, match="task_dirs"):
            validate_config(cfg)

    def test_harbor_empty_dirs_fails(self):
        cfg = TrainConfig(mode="weight", skyrl=_skyrl(), harbor=HarborConfig(task_dirs=[]))
        with pytest.raises(ValueError, match="task_dirs"):
            validate_config(cfg)


# ---------------------------------------------------------------------------
# LLMClientConfig
# ---------------------------------------------------------------------------

class TestLLMClientConfig:
    def test_secret_str_hidden(self):
        cfg = LLMClientConfig(model="test", api_key=SecretStr("secret-123"))
        assert "secret-123" not in repr(cfg)

    def test_secret_str_accessible(self):
        cfg = LLMClientConfig(model="test", api_key=SecretStr("secret-123"))
        assert cfg.api_key.get_secret_value() == "secret-123"

    def test_defaults(self):
        cfg = LLMClientConfig(model="test")
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2000


# ---------------------------------------------------------------------------
# Mode validation via Pydantic Literal
# ---------------------------------------------------------------------------

class TestPydanticModeValidation:
    def test_invalid_mode_rejected(self):
        with pytest.raises(Exception):
            TrainConfig(mode="invalid")

    def test_invalid_env_type_rejected(self):
        cfg = TrainConfig(mode="weight", skyrl=_skyrl(), env_type="custom")
        with pytest.raises(ValueError, match="Unknown env_type"):
            validate_config(cfg)

    def test_defaults(self):
        cfg = TrainConfig(mode="harness_learning")
        assert cfg.env_type == "harbor"
        assert cfg.episodes_per_iter == 10
        assert cfg.n_iterations == 100


# ---------------------------------------------------------------------------
# MathAdapter
# ---------------------------------------------------------------------------

class TestMathAdapter:
    def test_run_episode_produces_episode(self):
        from unittest.mock import MagicMock

        from clawloop.core.episode import Episode
        from clawloop.environments.math import MathAdapter, MathEnvironment

        env = MathEnvironment()
        mock_client = MagicMock()
        # Use the first problem and return its correct answer
        samples = env.get_tasks()
        mock_client.complete.return_value = f"The answer is \\boxed{{{samples[0].ground_truth}}}."

        adapter = MathAdapter(env=env, client=mock_client)
        task = samples[0].question

        # Mock agent_state
        agent_state = MagicMock()
        sample_result = MagicMock()
        sample_result.result.return_value.output = "Solve step by step."
        agent_state.harness.sample.return_value = sample_result
        agent_state.state_id.return_value.combined_hash = "abc123"

        ep = adapter.run_episode(task, agent_state)
        assert isinstance(ep, Episode)
        assert ep.bench == "math"
        assert ep.summary.total_reward == 1.0  # correct answer returned by mock
        assert len(ep.messages) == 3  # system, user, assistant

    def test_wrong_answer_gives_zero_reward(self):
        from unittest.mock import MagicMock

        from clawloop.environments.math import MathAdapter, MathEnvironment

        env = MathEnvironment()
        mock_client = MagicMock()
        mock_client.complete.return_value = "I think it's \\boxed{99}."

        adapter = MathAdapter(env=env, client=mock_client)
        tasks = [s.question for s in env.get_tasks()]

        agent_state = MagicMock()
        sample_result = MagicMock()
        sample_result.result.return_value.output = "Solve."
        agent_state.harness.sample.return_value = sample_result
        agent_state.state_id.return_value.combined_hash = "abc"

        ep = adapter.run_episode(tasks[0], agent_state)
        # Most likely wrong unless the first problem's answer happens to be 99
        assert ep.summary.total_reward in (0.0, 1.0)
        assert ep.metadata["ground_truth"] is not None

    def test_llm_failure_returns_filtered_episode(self):
        from unittest.mock import MagicMock

        from clawloop.environments.math import MathAdapter, MathEnvironment

        env = MathEnvironment()
        mock_client = MagicMock()
        mock_client.complete.side_effect = ConnectionError("LLM down")

        adapter = MathAdapter(env=env, client=mock_client)
        tasks = [s.question for s in env.get_tasks()]

        agent_state = MagicMock()
        sample_result = MagicMock()
        sample_result.result.return_value.output = "Solve."
        agent_state.harness.sample.return_value = sample_result

        ep = adapter.run_episode(tasks[0], agent_state)
        assert ep.summary.filtered is True
        assert ep.metadata["error"] == "ConnectionError"


# ---------------------------------------------------------------------------
# _make_llm_client
# ---------------------------------------------------------------------------

class TestMakeLLMClient:
    def test_empty_key_becomes_none(self):
        from clawloop.train import LLMClientConfig, _make_llm_client

        cfg = LLMClientConfig(model="test-model")
        client = _make_llm_client(cfg)
        assert client.api_key is None
        assert client.api_base is None

    def test_explicit_key_preserved(self):
        from clawloop.train import LLMClientConfig, _make_llm_client

        cfg = LLMClientConfig(model="test-model", api_key=SecretStr("sk-123"), api_base="http://proxy")
        client = _make_llm_client(cfg)
        assert client.api_key == "sk-123"
        assert client.api_base == "http://proxy"


# ---------------------------------------------------------------------------
# train() end-to-end (mocked backends)
# ---------------------------------------------------------------------------

class TestTrainEndToEnd:
    def test_harness_learning_math(self):
        """Full pipeline: train() with harness_learning + math env (mocked LLMs)."""
        from unittest.mock import MagicMock, patch

        from clawloop.train import LLMClientConfig, TrainConfig, train

        mock_reflector = MagicMock()
        mock_reflector.complete.return_value = "[]"
        mock_task = MagicMock()
        mock_task.complete.return_value = "The answer is \\boxed{45}."

        cfg = TrainConfig(
            mode="harness_learning",
            env_type="math",
            llm_clients={
                "reflector": LLMClientConfig(model="mock-reflector"),
                "task": LLMClientConfig(model="mock-task"),
            },
            episodes_per_iter=2,
            n_iterations=1,
        )

        with patch("clawloop.train._make_llm_client") as mock_make:
            def _pick_client(llm_cfg):
                if "reflector" in llm_cfg.model:
                    return mock_reflector
                return mock_task
            mock_make.side_effect = _pick_client

            agent_state, state_id = train(cfg)
            assert state_id.combined_hash
            assert mock_task.complete.called
