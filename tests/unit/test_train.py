"""Unit tests for env_type=openspiel + weight_backend=tinker wiring in train.py."""

from __future__ import annotations

import pytest


def test_build_openspiel_tasks_repeat_per_seed():
    from clawloop.train import ENV_BUILDERS, TrainConfig

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="tinker",
        openspiel={
            "game_name": "blackjack",
            "seeds": [0, 1],
            "episodes_per_seed": 4,
        },
        tinker={"base_model": "Qwen/Qwen3-8B"},
        n_iterations=1,
    )
    builder = ENV_BUILDERS["openspiel"]
    adapter, tasks = builder(cfg, cfg.llm_clients)
    assert len(tasks) == 8
    assert tasks.count("blackjack_seed_0") == 4
    assert tasks.count("blackjack_seed_1") == 4
    # Adapter dispatches by task_id.
    assert "blackjack_seed_0" in adapter._envs_by_task_id
    assert "blackjack_seed_1" in adapter._envs_by_task_id


def test_effective_episodes_per_iter_for_openspiel_single_game():
    from clawloop.train import TrainConfig, effective_episodes_per_iter

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="tinker",
        openspiel={
            "game_name": "blackjack",
            "seeds": [0, 1, 2],
            "episodes_per_seed": 5,
        },
        tinker={"base_model": "Qwen/Qwen3-8B"},
        n_iterations=1,
    )
    # Validator must NOT mutate the user's config — the derived count comes
    # from effective_episodes_per_iter, not a side effect.
    assert effective_episodes_per_iter(cfg) == 15  # 3 seeds * 5 per seed


def test_build_openspiel_mixed_games_interleaves_tasks():
    """`openspiel.games: [...]` -> envs from multiple games, task_id preserves
    `{game}_seed_{n}` so GRPO grouping stays per-(game, seed)."""
    from clawloop.train import ENV_BUILDERS, TrainConfig

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="tinker",
        openspiel={
            "games": [
                {
                    "game_name": "blackjack",
                    "seeds": [0, 1],
                    "episodes_per_seed": 3,
                    "max_turns": 10,
                },
                {"game_name": "2048", "seeds": [10, 11], "episodes_per_seed": 2, "max_turns": 200},
            ],
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 64,
        },
        tinker={"base_model": "Qwen/Qwen3-8B"},
        n_iterations=1,
    )
    builder = ENV_BUILDERS["openspiel"]
    adapter, tasks = builder(cfg, cfg.llm_clients)
    # 2 blackjack seeds * 3 + 2 2048 seeds * 2 = 10
    assert len(tasks) == 10
    assert tasks.count("blackjack_seed_0") == 3
    assert tasks.count("blackjack_seed_1") == 3
    assert tasks.count("2048_seed_10") == 2
    assert tasks.count("2048_seed_11") == 2
    assert set(adapter._envs_by_task_id.keys()) == {
        "blackjack_seed_0",
        "blackjack_seed_1",
        "2048_seed_10",
        "2048_seed_11",
    }


def test_effective_episodes_per_iter_for_mixed_games():
    from clawloop.train import TrainConfig, effective_episodes_per_iter

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="tinker",
        openspiel={
            "games": [
                {"game_name": "blackjack", "seeds": [0, 1, 2], "episodes_per_seed": 4},
                {"game_name": "2048", "seeds": [0, 1], "episodes_per_seed": 3},
            ],
        },
        tinker={"base_model": "Qwen/Qwen3-8B"},
    )
    # (3 * 4) + (2 * 3) = 18
    assert effective_episodes_per_iter(cfg) == 18


def test_validate_config_rejects_mixed_game_without_game_name():
    from clawloop.train import TrainConfig, validate_config

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="tinker",
        openspiel={"games": [{"seeds": [0]}]},
        tinker={"base_model": "Qwen/Qwen3-8B"},
    )
    with pytest.raises(ValueError, match="game_name"):
        validate_config(cfg)


def test_validate_config_rejects_openspiel_without_game_name():
    from clawloop.train import TrainConfig, validate_config

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="tinker",
        openspiel={"seeds": [0, 1]},
        tinker={"base_model": "Qwen/Qwen3-8B"},
    )
    with pytest.raises(ValueError, match="game_name"):
        validate_config(cfg)


def test_validate_config_rejects_empty_seeds():
    from clawloop.train import TrainConfig, validate_config

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="tinker",
        openspiel={"game_name": "blackjack", "seeds": []},
        tinker={"base_model": "Qwen/Qwen3-8B"},
    )
    with pytest.raises(ValueError, match="seeds"):
        validate_config(cfg)


def test_validate_config_requires_tinker_config_when_backend_tinker():
    from clawloop.train import TrainConfig, validate_config

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="tinker",
        openspiel={"game_name": "blackjack", "seeds": [0]},
        tinker=None,
    )
    with pytest.raises(ValueError, match="tinker"):
        validate_config(cfg)


def test_validate_config_requires_skyrl_config_when_backend_skyrl():
    from clawloop.train import TrainConfig, validate_config

    cfg = TrainConfig(
        mode="weight",
        env_type="openspiel",
        weight_backend="skyrl",
        openspiel={"game_name": "blackjack", "seeds": [0]},
        skyrl=None,
    )
    with pytest.raises(ValueError, match="skyrl"):
        validate_config(cfg)
