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


class TestTrainEndToEnd:
    """End-to-end: train() with harness_learning mode + mocked Harbor trials."""

    def test_harness_learning_with_harbor_fixtures(self):
        """Full pipeline: TrainConfig → train() → learning_loop → HarborAdapter → Episodes."""
        import asyncio
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock

        from lfx.train import train

        fixture_dir = Path(__file__).parent / "fixtures" / "harbor_tasks"
        if not fixture_dir.exists():
            pytest.skip("Harbor task fixtures not found")

        cfg = TrainConfig(
            mode="harness_learning",
            harbor=HarborConfig(
                task_dirs=[str(fixture_dir / "bfcl-simple-0")],
                trial_config={"agent": {"name": "test", "kwargs": {}}},
            ),
            episodes_per_iter=1,
            n_iterations=1,
        )

        # Monkey-patch HarborTaskEnvironment to use mocked Trial
        import lfx.envs.harbor as harbor_mod

        _orig_init = harbor_mod.HarborTaskEnvironment.__init__

        def _patched_init(self, task_dir, trial_config, **kwargs):
            self._task_dir = Path(task_dir)
            self._trial_config = trial_config
            self._trial_config.setdefault("task", {})
            self._trial_config["agent"].setdefault("kwargs", {})
            self._reward_transform = kwargs.get("reward_transform")
            self._train_on_truncated = kwargs.get("train_on_truncated", True)

            mock_results = MagicMock()
            mock_results.verifier_result.rewards = {"reward": 0.8}
            mock_results.agent_result.metadata = {
                "all_messages": [
                    {"role": "user", "content": "Call get_weather"},
                    {"role": "assistant", "content": '{"name": "get_weather"}'},
                ],
            }
            self._Trial = MagicMock()
            mock_trial = MagicMock()
            mock_trial.run = AsyncMock(return_value=mock_results)
            self._Trial.return_value = mock_trial
            self._TrialConfig = MagicMock()

        harbor_mod.HarborTaskEnvironment.__init__ = _patched_init
        try:
            agent_state, state_id = train(cfg)
            assert state_id.combined_hash  # Got a valid state ID
        finally:
            harbor_mod.HarborTaskEnvironment.__init__ = _orig_init
