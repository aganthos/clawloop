"""Tests for lfx.backends.skyrl — SkyRLWeightsBackend + SkyRLWeightsConfig."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig
from lfx.backends.base import BackendError
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.types import Datum, FBResult, OptimResult, SampleContext, SaveResult, LoadResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_episode(task_id: str = "t1", reward: float = 0.8) -> Episode:
    return Episode(
        id=Episode.new_id(),
        state_id="abc",
        task_id=task_id,
        bench="test",
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


def _make_backend_with_mocks() -> SkyRLWeightsBackend:
    """Create a SkyRLWeightsBackend with mocked internals."""
    from tests.test_skyrl_export import FakeTokenizer
    from lfx.exporters.skyrl import SkyRLExporter

    backend = SkyRLWeightsBackend.__new__(SkyRLWeightsBackend)
    backend._config = SkyRLWeightsConfig(
        base_model="test-model",
        backend_type="jax",
        backend_config={},
        lora_config={"rank": 8},
        training_config={"loss_fn": "ppo", "adam_params": {"learning_rate": 1e-5}},
        tokenizer_name="test",
    )
    backend._backend = MagicMock()
    backend._model_id = "lfx-test-model"
    backend._adapter_refs: list[str] = []
    backend.inference_url = None
    backend._exporter = SkyRLExporter(tokenizer=FakeTokenizer())
    return backend


# ---------------------------------------------------------------------------
# SkyRLWeightsConfig
# ---------------------------------------------------------------------------

class TestSkyRLWeightsConfig:
    def test_config_fields(self) -> None:
        cfg = SkyRLWeightsConfig()
        assert cfg.base_model == ""
        assert cfg.backend_type == "jax"
        assert cfg.backend_config == {}
        assert cfg.lora_config == {}
        assert cfg.training_config == {}
        assert cfg.tokenizer_name == ""

    def test_config_custom_values(self) -> None:
        cfg = SkyRLWeightsConfig(
            base_model="llama-7b",
            backend_type="skyrl_train",
            backend_config={"devices": 4},
            lora_config={"rank": 16},
            training_config={"loss_fn": "grpo"},
            tokenizer_name="llama-tokenizer",
        )
        assert cfg.base_model == "llama-7b"
        assert cfg.backend_type == "skyrl_train"
        assert cfg.backend_config == {"devices": 4}
        assert cfg.lora_config == {"rank": 16}
        assert cfg.training_config == {"loss_fn": "grpo"}
        assert cfg.tokenizer_name == "llama-tokenizer"


# ---------------------------------------------------------------------------
# forward_backward
# ---------------------------------------------------------------------------

class TestForwardBackward:
    def test_forward_backward_calls_exporter_and_backend(self) -> None:
        backend = _make_backend_with_mocks()
        ep = _make_episode()
        datum = Datum(episodes=[ep])

        # Mock backend.forward_backward to return a dict with metrics
        backend._backend.forward_backward.return_value = {
            "loss": 0.5,
            "grad_norm": 1.2,
        }

        result = backend.forward_backward(datum).result()

        assert result.status == "ok"
        # Backend's forward_backward must have been called once
        backend._backend.forward_backward.assert_called_once()
        # Verify the call received a dict with the expected keys
        call_args = backend._backend.forward_backward.call_args
        fb_input = call_args[0][0]
        assert "prompt_token_ids" in fb_input
        assert "response_ids" in fb_input
        assert "rewards" in fb_input
        assert "trajectory_ids" in fb_input

    def test_forward_backward_error_returns_error_result(self) -> None:
        backend = _make_backend_with_mocks()
        ep = _make_episode()
        datum = Datum(episodes=[ep])

        backend._backend.forward_backward.side_effect = RuntimeError("GPU exploded")

        result = backend.forward_backward(datum).result()

        assert result.status == "error"
        assert "error" in result.metrics
        err = result.metrics["error"]
        assert isinstance(err, BackendError)
        assert "GPU exploded" in err.message

    def test_forward_backward_passes_raw_rewards(self) -> None:
        """LfX must NOT compute advantages — raw rewards go to SkyRL."""
        backend = _make_backend_with_mocks()
        ep = _make_episode(reward=0.9)
        datum = Datum(episodes=[ep])

        backend._backend.forward_backward.return_value = {"loss": 0.3}

        backend.forward_backward(datum).result()

        call_args = backend._backend.forward_backward.call_args
        fb_input = call_args[0][0]
        # The rewards list should contain the raw terminal reward
        assert 0.9 in fb_input["rewards"]


# ---------------------------------------------------------------------------
# optim_step
# ---------------------------------------------------------------------------

class TestOptimStep:
    def test_optim_step_calls_backend(self) -> None:
        backend = _make_backend_with_mocks()
        backend._backend.optim_step.return_value = {"status": "ok"}

        result = backend.optim_step().result()

        assert result.status == "ok"
        backend._backend.optim_step.assert_called_once()
        # Verify model_id is passed
        call_args = backend._backend.optim_step.call_args
        assert call_args[0][0] == "lfx-test-model"

    def test_optim_step_error_returns_error_result(self) -> None:
        backend = _make_backend_with_mocks()
        backend._backend.optim_step.side_effect = RuntimeError("loss is nan")

        result = backend.optim_step().result()

        assert result.status == "error"
        assert "error" in result.metrics
        err = result.metrics["error"]
        assert isinstance(err, BackendError)


# ---------------------------------------------------------------------------
# to_dict
# ---------------------------------------------------------------------------

class TestToDict:
    def test_to_dict_includes_all_config(self) -> None:
        backend = _make_backend_with_mocks()
        d = backend.to_dict()

        assert d["model_ref"] == "test-model"
        assert d["backend_type"] == "jax"
        assert d["backend_config"] == {}
        assert d["lora_config"] == {"rank": 8}
        assert d["training_config"] == {"loss_fn": "ppo", "adam_params": {"learning_rate": 1e-5}}
        assert d["adapter_refs"] == []


# ---------------------------------------------------------------------------
# clear_pending_state
# ---------------------------------------------------------------------------

class TestClearPendingState:
    def test_clear_pending_is_noop(self) -> None:
        backend = _make_backend_with_mocks()
        # Should not raise
        backend.clear_pending_state()


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

class TestSample:
    def test_sample_returns_model_ref(self) -> None:
        backend = _make_backend_with_mocks()
        result = backend.sample(SampleContext(bench="test")).result()
        assert result.output == "test-model"


# ---------------------------------------------------------------------------
# save_state / load_state
# ---------------------------------------------------------------------------

class TestSaveLoadState:
    def test_save_state_appends_adapter_ref(self) -> None:
        backend = _make_backend_with_mocks()
        backend._backend.save_checkpoint.return_value = None

        result = backend.save_state("ckpt-001").result()

        assert result.status == "ok"
        assert result.name == "ckpt-001"
        assert "ckpt-001" in backend._adapter_refs
        backend._backend.save_checkpoint.assert_called_once()

    def test_load_state_empty_adapters_skips_checkpoint(self) -> None:
        backend = _make_backend_with_mocks()

        state = {
            "model_ref": "test-model",
            "backend_type": "jax",
            "backend_config": {},
            "lora_config": {"rank": 8},
            "training_config": {},
            "adapter_refs": [],
        }
        result = backend.load_state(state).result()

        assert result.status == "ok"
        backend._backend.load_checkpoint.assert_not_called()

    def test_load_state_with_adapters_calls_load_checkpoint(self) -> None:
        backend = _make_backend_with_mocks()
        backend._backend.load_checkpoint.return_value = None

        state = {
            "model_ref": "test-model",
            "backend_type": "jax",
            "backend_config": {},
            "lora_config": {"rank": 8},
            "training_config": {},
            "adapter_refs": ["ckpt-001", "ckpt-002"],
        }
        result = backend.load_state(state).result()

        assert result.status == "ok"
        # Should load the latest (last) adapter
        backend._backend.load_checkpoint.assert_called_once_with(
            backend._model_id, "ckpt-002"
        )
        assert backend._adapter_refs == ["ckpt-001", "ckpt-002"]

    def test_load_state_validates_keys(self) -> None:
        backend = _make_backend_with_mocks()

        # Missing required key
        state = {"model_ref": "test-model"}
        result = backend.load_state(state).result()

        assert result.status.startswith("error")
