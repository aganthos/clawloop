"""Tests for lfx.backends.skyrl — SkyRLWeightsBackend + SkyRLWeightsConfig.

Mock tests bypass __init__ (no real SkyRL/GPU needed).
Real-type tests are conditional on SkyRL submodule availability.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from lfx.backends.base import BackendError
from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.types import Datum, SampleContext


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
    """Create a SkyRLWeightsBackend with mocked internals (no SkyRL needed)."""
    from lfx.exporters.skyrl import SkyRLExporter
    from tests.test_skyrl_export import FakeTokenizer

    backend = SkyRLWeightsBackend.__new__(SkyRLWeightsBackend)
    backend._config = SkyRLWeightsConfig(
        base_model="test-model",
        backend_type="jax",
        backend_config={},
        lora_config={"rank": 8},
        training_config={
            "loss_fn": "ppo",
            "adam_params": {"learning_rate": 1e-5},
        },
        tokenizer_name="test",
    )
    backend._backend = MagicMock()
    backend._model_id = "lfx-test-model"
    backend._adapter_refs: list[str] = []
    backend.inference_url = None
    backend._exporter = SkyRLExporter(tokenizer=FakeTokenizer())
    return backend


def _skyrl_available() -> bool:
    try:
        sys.path.insert(0, "lfx/skyrl")
        from skyrl.tinker.types import PreparedModelPassBatch  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestSkyRLWeightsConfig:
    def test_config_defaults(self) -> None:
        cfg = SkyRLWeightsConfig()
        assert cfg.base_model == ""
        assert cfg.backend_type == "jax"

    def test_config_custom(self) -> None:
        cfg = SkyRLWeightsConfig(
            base_model="llama-7b",
            backend_type="skyrl_train",
            lora_config={"rank": 16},
        )
        assert cfg.base_model == "llama-7b"
        assert cfg.lora_config == {"rank": 16}


# ---------------------------------------------------------------------------
# forward_backward (mocked backend)
# ---------------------------------------------------------------------------

class TestForwardBackwardMocked:
    def test_calls_backend(self) -> None:
        backend = _make_backend_with_mocks()
        backend._backend.forward_backward.return_value = {}

        result = backend.forward_backward(Datum(episodes=[_make_episode()])).result()
        assert result.status == "ok"
        backend._backend.forward_backward.assert_called_once()

    def test_error_returns_error_result(self) -> None:
        backend = _make_backend_with_mocks()
        backend._backend.forward_backward.side_effect = RuntimeError("GPU exploded")

        result = backend.forward_backward(Datum(episodes=[_make_episode()])).result()
        assert result.status == "error"
        assert isinstance(result.metrics["error"], BackendError)


# ---------------------------------------------------------------------------
# forward_backward with REAL SkyRL types (conditional)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _skyrl_available(), reason="SkyRL submodule not available")
class TestForwardBackwardRealTypes:
    """Validate _to_prepared_batch constructs a real PreparedModelPassBatch."""

    def test_prepared_batch_is_valid_type(self) -> None:
        from skyrl.tinker.types import PreparedModelPassBatch

        backend = _make_backend_with_mocks()
        gen_output = backend._exporter.export([_make_episode()])
        batch = backend._to_prepared_batch(gen_output)

        assert isinstance(batch, PreparedModelPassBatch)

    def test_prepared_batch_has_correct_sequence_count(self) -> None:
        backend = _make_backend_with_mocks()
        episodes = [_make_episode(task_id="t1", reward=0.9),
                     _make_episode(task_id="t1", reward=0.3)]
        gen_output = backend._exporter.export(episodes)
        batch = backend._to_prepared_batch(gen_output)

        n = len(gen_output["prompt_token_ids"])
        assert len(batch.all_input_ids) == n
        assert len(batch.all_targets) == n
        assert len(batch.all_token_weights) == n
        assert len(batch.all_sampling_logprobs) == n
        assert len(batch.all_advantages) == n
        assert len(batch.all_model_ids) == n
        assert len(batch.all_loss_fns) == n

    def test_input_ids_are_prompt_plus_response(self) -> None:
        backend = _make_backend_with_mocks()
        gen_output = backend._exporter.export([_make_episode()])
        batch = backend._to_prepared_batch(gen_output)

        for i in range(len(gen_output["prompt_token_ids"])):
            expected = gen_output["prompt_token_ids"][i] + gen_output["response_ids"][i]
            assert batch.all_input_ids[i] == expected

    def test_targets_are_response_ids(self) -> None:
        backend = _make_backend_with_mocks()
        gen_output = backend._exporter.export([_make_episode()])
        batch = backend._to_prepared_batch(gen_output)

        for i in range(len(gen_output["response_ids"])):
            assert batch.all_targets[i] == gen_output["response_ids"][i]

    def test_token_weights_match_loss_masks(self) -> None:
        backend = _make_backend_with_mocks()
        gen_output = backend._exporter.export([_make_episode()])
        batch = backend._to_prepared_batch(gen_output)

        for i in range(len(gen_output["loss_masks"])):
            assert batch.all_token_weights[i] == [float(w) for w in gen_output["loss_masks"][i]]

    def test_advantages_are_grpo_computed(self) -> None:
        """GRPO: advantage = reward - group_mean."""
        backend = _make_backend_with_mocks()
        episodes = [
            _make_episode(task_id="t1", reward=0.9),
            _make_episode(task_id="t1", reward=0.3),
            _make_episode(task_id="t1", reward=0.6),
        ]
        gen_output = backend._exporter.export(episodes)
        batch = backend._to_prepared_batch(gen_output)

        # Group mean for t1: (0.9 + 0.3 + 0.6) / 3 = 0.6
        # Advantages: 0.3, -0.3, 0.0
        # Each advantage is broadcast to all response tokens
        rewards = gen_output["rewards"]
        mean_reward = sum(rewards) / len(rewards)
        for i in range(len(rewards)):
            expected_adv = rewards[i] - mean_reward
            # All tokens in the sequence get the same advantage
            for adv in batch.all_advantages[i]:
                assert adv == pytest.approx(expected_adv, abs=1e-6)

    def test_model_ids_and_loss_fns_set(self) -> None:
        backend = _make_backend_with_mocks()
        gen_output = backend._exporter.export([_make_episode()])
        batch = backend._to_prepared_batch(gen_output)

        for mid in batch.all_model_ids:
            assert mid == "lfx-test-model"
        for lfn in batch.all_loss_fns:
            assert lfn == "ppo"

    def test_request_batch_slices_valid(self) -> None:
        backend = _make_backend_with_mocks()
        gen_output = backend._exporter.export([_make_episode()])
        batch = backend._to_prepared_batch(gen_output)

        assert len(batch.request_batch_slices) == len(gen_output["prompt_token_ids"])
        for req_id, model_id, start, end in batch.request_batch_slices:
            assert isinstance(req_id, str)
            assert model_id == "lfx-test-model"
            assert isinstance(start, int)
            assert isinstance(end, int)

    def test_full_pipeline_episode_to_prepared_batch(self) -> None:
        """End-to-end: Episode → SkyRLExporter → _to_prepared_batch → valid PreparedModelPassBatch."""
        from skyrl.tinker.types import PreparedModelPassBatch

        backend = _make_backend_with_mocks()
        episodes = [
            _make_episode(task_id="task-a", reward=1.0),
            _make_episode(task_id="task-a", reward=0.0),
            _make_episode(task_id="task-b", reward=0.5),
        ]
        gen_output = backend._exporter.export(episodes)
        batch = backend._to_prepared_batch(gen_output)

        assert isinstance(batch, PreparedModelPassBatch)
        # All arrays have consistent lengths
        n = len(batch.all_input_ids)
        assert n > 0
        assert len(batch.all_targets) == n
        assert len(batch.all_advantages) == n
        # GRPO: task-a group has mean 0.5, task-b group has mean 0.5
        # (rewards are per-step not per-episode in exporter output, so exact values depend on step structure)


# ---------------------------------------------------------------------------
# optim_step (mocked)
# ---------------------------------------------------------------------------

class TestOptimStep:
    def test_calls_backend(self) -> None:
        backend = _make_backend_with_mocks()
        backend._backend.optim_step.return_value = MagicMock(metrics={"grad_norm": 0.1})

        result = backend.optim_step().result()
        assert result.status == "ok"
        assert result.updates_applied == 1
        backend._backend.optim_step.assert_called_once()

    def test_error_returns_error_result(self) -> None:
        backend = _make_backend_with_mocks()
        backend._backend.optim_step.side_effect = RuntimeError("loss is nan")

        result = backend.optim_step().result()
        assert result.status == "error"


@pytest.mark.skipif(not _skyrl_available(), reason="SkyRL submodule not available")
class TestOptimStepRealTypes:
    """Validate optim_step constructs a real OptimStepInput."""

    def test_optim_step_passes_real_type(self) -> None:
        from skyrl.tinker.types import OptimStepInput, OptimStepOutput

        backend = _make_backend_with_mocks()
        backend._backend.optim_step.return_value = OptimStepOutput(metrics={"grad_norm": 0.05})

        result = backend.optim_step().result()
        assert result.status == "ok"
        assert result.metrics["grad_norm"] == 0.05

        # Verify the call received a real OptimStepInput
        call_args = backend._backend.optim_step.call_args
        assert call_args[0][0] == "lfx-test-model"
        optim_input = call_args[0][1]
        assert isinstance(optim_input, OptimStepInput)
        assert optim_input.adam_params.learning_rate == 1e-5
        assert optim_input.adam_params.beta1 == 0.9
        assert optim_input.adam_params.beta2 == 0.999


# ---------------------------------------------------------------------------
# Other protocol methods
# ---------------------------------------------------------------------------

class TestToDict:
    def test_includes_all_config(self) -> None:
        backend = _make_backend_with_mocks()
        d = backend.to_dict()
        assert d["model_ref"] == "test-model"
        assert d["backend_type"] == "jax"
        assert d["adapter_refs"] == []


class TestClearPendingState:
    def test_is_noop(self) -> None:
        backend = _make_backend_with_mocks()
        backend.clear_pending_state()  # Should not raise


class TestSample:
    def test_returns_model_ref(self) -> None:
        backend = _make_backend_with_mocks()
        result = backend.sample(SampleContext()).result()
        assert result.output == "test-model"


class TestSaveLoadState:
    def test_save_appends_adapter(self) -> None:
        backend = _make_backend_with_mocks()
        result = backend.save_state("ckpt-1").result()
        assert result.status == "ok"
        assert "ckpt-1" in backend._adapter_refs

    def test_load_empty_adapters_skips_checkpoint(self) -> None:
        backend = _make_backend_with_mocks()
        state = {
            "model_ref": "m", "backend_type": "jax", "backend_config": {},
            "lora_config": {}, "training_config": {}, "adapter_refs": [],
        }
        result = backend.load_state(state).result()
        assert result.status == "ok"
        backend._backend.load_checkpoint.assert_not_called()

    def test_load_with_adapters_restores(self) -> None:
        backend = _make_backend_with_mocks()
        state = {
            "model_ref": "m", "backend_type": "jax", "backend_config": {},
            "lora_config": {}, "training_config": {}, "adapter_refs": ["a", "b"],
        }
        result = backend.load_state(state).result()
        assert result.status == "ok"
        backend._backend.load_checkpoint.assert_called_once_with(backend._model_id, "b")

    def test_load_validates_keys(self) -> None:
        backend = _make_backend_with_mocks()
        result = backend.load_state({"model_ref": "m"}).result()
        assert result.status.startswith("error")
