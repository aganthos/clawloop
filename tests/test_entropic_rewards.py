# tests/test_entropic_rewards.py
"""Tests for Entropic CRMArenaPro reward mapping."""

from clawloop.environments._entropic_rewards import DEFAULT_ENTROPIC_WEIGHTS, map_entropic_scores


class TestMapEntropicScores:
    """map_entropic_scores converts 7-dimension scores to ClawLoop RewardSignals."""

    def test_perfect_scores(self):
        """All dimensions 100 → outcome +1.0, all signals +1.0."""
        scores = {k: 100.0 for k in DEFAULT_ENTROPIC_WEIGHTS}
        signals, breakdown = map_entropic_scores(scores, task_reward=1.0)

        assert signals["outcome"].value == 1.0
        assert signals["outcome"].confidence == 1.0
        for name in DEFAULT_ENTROPIC_WEIGHTS:
            assert signals[name].value == 1.0
            assert signals[name].confidence == 1.0

    def test_zero_scores(self):
        """All dimensions 0 → outcome -1.0, all signals -1.0."""
        scores = {k: 0.0 for k in DEFAULT_ENTROPIC_WEIGHTS}
        signals, breakdown = map_entropic_scores(scores, task_reward=0.0)

        assert signals["outcome"].value == -1.0
        for name in DEFAULT_ENTROPIC_WEIGHTS:
            assert signals[name].value == -1.0

    def test_mixed_scores(self):
        """Mixed dimension scores produce correct per-signal values."""
        scores = {
            "functional": 100.0,
            "drift_adaptation": 0.0,
            "token_efficiency": 50.0,
            "query_efficiency": 75.0,
            "error_recovery": 100.0,
            "trajectory_efficiency": 0.0,
            "hallucination_rate": 100.0,
        }
        signals, breakdown = map_entropic_scores(scores, task_reward=1.0)

        assert signals["outcome"].value == 1.0
        assert signals["functional"].value == 1.0
        assert signals["drift_adaptation"].value == -1.0
        assert signals["token_efficiency"].value == 0.0  # 50/100 → 0.5 → 0.0

    def test_missing_dimensions(self):
        """Missing dimensions are skipped, not crashed."""
        signals, breakdown = map_entropic_scores({}, task_reward=0.0)

        assert signals["outcome"].value == -1.0
        assert "functional" not in signals

    def test_unknown_dimensions_stored_in_breakdown(self):
        """Extra dimensions stored in breakdown, not mapped."""
        scores = {"functional": 100.0, "new_dimension": 42.0}
        signals, breakdown = map_entropic_scores(scores, task_reward=1.0)

        assert "new_dimension" not in signals
        assert breakdown["new_dimension"] == 42.0

    def test_out_of_range_clamped(self):
        """Values outside [0, 100] are clamped."""
        scores = {"functional": 150.0, "drift_adaptation": -20.0}
        signals, breakdown = map_entropic_scores(scores, task_reward=1.0)

        assert signals["functional"].value == 1.0  # 150→clamped to 100→1.0
        assert signals["drift_adaptation"].value == -1.0  # -20→clamped to 0→-1.0

    def test_non_numeric_dimension(self):
        """Non-numeric values skipped with warning."""
        scores = {"functional": "bad", "drift_adaptation": 80.0}
        signals, breakdown = map_entropic_scores(scores, task_reward=1.0)

        assert "functional" not in signals
        assert signals["drift_adaptation"].value > 0

    def test_custom_weights(self):
        """Custom weights override defaults."""
        custom = {"functional": 1.0}
        scores = {"functional": 100.0, "drift_adaptation": 0.0}
        signals, breakdown = map_entropic_scores(scores, task_reward=1.0, weights=custom)

        assert "functional" in signals
        assert "drift_adaptation" not in signals

    def test_non_binary_confidence(self):
        """Non-extreme values (not 0 or 100) get confidence 0.8."""
        scores = {"functional": 50.0}
        signals, _ = map_entropic_scores(scores, task_reward=0.5)

        assert signals["functional"].confidence == 0.8

    def test_breakdown_contains_validated(self):
        """Breakdown includes all validated dimensions."""
        scores = {k: 75.0 for k in DEFAULT_ENTROPIC_WEIGHTS}
        _, breakdown = map_entropic_scores(scores, task_reward=1.0)

        for name in DEFAULT_ENTROPIC_WEIGHTS:
            assert name in breakdown

    def test_hundred_scale_normalisation(self):
        """Scores on 0-100 scale are normalised to [0,1] then mapped to [-1,1]."""
        scores = {"functional": 50.0}
        signals, breakdown = map_entropic_scores(scores, task_reward=1.0)

        # 50/100 = 0.5 → val*2-1 = 0.0
        assert signals["functional"].value == 0.0
        assert breakdown["functional"] == 0.5
