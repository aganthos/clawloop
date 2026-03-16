"""Tests for TrainConfig and the train() entry point."""

import pytest

from lfx.train import HarborConfig, TrainConfig


class TestTrainConfig:
    def test_weight_mode(self):
        cfg = TrainConfig(
            mode="weight",
            harbor=HarborConfig(task_dirs=["/data/tasks"]),
            skyrl={"base_model": "Qwen/Qwen3-8B", "backend_type": "jax"},
        )
        assert cfg.mode == "weight"

    def test_harness_learning_mode(self):
        cfg = TrainConfig(
            mode="harness_learning",
            harbor=HarborConfig(task_dirs=["/data/tasks"]),
        )
        assert cfg.mode == "harness_learning"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            TrainConfig(mode="invalid")

    def test_harbor_config(self):
        hc = HarborConfig(
            task_dirs=["/a", "/b"],
            trial_config={"agent": {"name": "t2"}},
            train_on_truncated=False,
        )
        assert len(hc.task_dirs) == 2
        assert hc.train_on_truncated is False

    def test_default_params(self):
        cfg = TrainConfig(mode="harness_learning")
        assert cfg.episodes_per_iter == 10
        assert cfg.n_iterations == 100
        assert cfg.env_type == "harbor"


class TestTrainFunction:
    def test_weight_mode_requires_skyrl(self):
        from lfx.train import train

        cfg = TrainConfig(
            mode="weight",
            harbor=HarborConfig(task_dirs=["/data/tasks"]),
        )
        with pytest.raises(ValueError, match="skyrl"):
            train(cfg)

    def test_no_envs_raises(self):
        from lfx.train import train

        cfg = TrainConfig(mode="harness_learning")
        with pytest.raises(ValueError, match="environments"):
            train(cfg)
