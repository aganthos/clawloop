# tests/test_car_rewards.py
"""Tests for CAR-bench reward mapping."""

from clawloop.adapters._car_rewards import map_car_scores, DEFAULT_CAR_WEIGHTS


class TestMapCarScores:
    """map_car_scores converts CAR metrics to ClawLoop RewardSignals."""

    def test_perfect_scores(self):
        """All metrics 1.0 → outcome signal near +1.0."""
        reward_info = {k: 1.0 for k in DEFAULT_CAR_WEIGHTS}
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        assert signals["outcome"].value == 1.0
        assert signals["outcome"].confidence == 1.0
        for name in DEFAULT_CAR_WEIGHTS:
            assert signals[name].value == 1.0
            assert signals[name].confidence == 1.0

    def test_zero_scores(self):
        """All metrics 0.0 → outcome signal -1.0."""
        reward_info = {k: 0.0 for k in DEFAULT_CAR_WEIGHTS}
        signals, breakdown = map_car_scores(reward_info, task_reward=0.0)

        assert signals["outcome"].value == -1.0
        for name in DEFAULT_CAR_WEIGHTS:
            assert signals[name].value == -1.0

    def test_mixed_scores(self):
        """Mixed metrics produce weighted composite."""
        reward_info = {
            "r_actions_final": 1.0,
            "r_actions_intermediate": 0.0,
            "r_tool_subset": 1.0,
            "r_tool_execution_errors": 1.0,
            "r_policy_errors": 0.0,
            "r_user_end_conversation": 1.0,
        }
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        # task_reward drives outcome
        assert signals["outcome"].value == 1.0
        assert signals["r_actions_final"].value == 1.0
        assert signals["r_actions_intermediate"].value == -1.0

    def test_missing_metrics(self):
        """Missing metrics default to 0 with warning, not crash."""
        signals, breakdown = map_car_scores({}, task_reward=0.0)

        assert signals["outcome"].value == -1.0
        assert "r_actions_final" not in signals  # not created if missing

    def test_unknown_metrics_stored_in_breakdown(self):
        """Extra CAR metrics are stored but not mapped to signals."""
        reward_info = {"r_actions_final": 1.0, "r_new_metric": 0.5}
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        assert "r_new_metric" not in signals
        assert breakdown["r_new_metric"] == 0.5

    def test_out_of_range_clamped(self):
        """Values outside [0,1] are clamped."""
        reward_info = {"r_actions_final": 1.5, "r_policy_errors": -0.3}
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        assert signals["r_actions_final"].value == 1.0  # clamped 1.5→1.0→mapped to +1
        assert signals["r_policy_errors"].value == -1.0  # clamped -0.3→0.0→mapped to -1

    def test_non_numeric_metric(self):
        """Non-numeric values are skipped with warning."""
        reward_info = {"r_actions_final": "bad", "r_policy_errors": 1.0}
        signals, breakdown = map_car_scores(reward_info, task_reward=1.0)

        assert "r_actions_final" not in signals
        assert signals["r_policy_errors"].value == 1.0

    def test_custom_weights(self):
        """Custom weights override defaults."""
        custom = {"r_actions_final": 1.0}
        reward_info = {"r_actions_final": 1.0, "r_policy_errors": 0.0}
        signals, breakdown = map_car_scores(
            reward_info, task_reward=1.0, weights=custom
        )

        # Only r_actions_final mapped (custom weights has only that)
        assert "r_actions_final" in signals
        assert "r_policy_errors" not in signals

    def test_non_binary_confidence(self):
        """Non-binary values (not 0.0 or 1.0) get confidence 0.8."""
        reward_info = {"r_actions_final": 0.5}
        signals, _ = map_car_scores(reward_info, task_reward=0.5)

        assert signals["r_actions_final"].confidence == 0.8

    def test_breakdown_contains_all_known(self):
        """Breakdown includes all validated metrics."""
        reward_info = {k: 1.0 for k in DEFAULT_CAR_WEIGHTS}
        _, breakdown = map_car_scores(reward_info, task_reward=1.0)

        for name in DEFAULT_CAR_WEIGHTS:
            assert name in breakdown
