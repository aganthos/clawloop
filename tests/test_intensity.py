"""Tests for lfx.core.intensity — adaptive reflection scheduling."""

from lfx.core.intensity import AdaptiveIntensity


class TestShouldReflect:
    def test_initial_should_reflect_true(self) -> None:
        ai = AdaptiveIntensity()
        assert ai.should_reflect(0) is True

    def test_improving_reflects_less(self) -> None:
        """After recording improving rewards, only reflects every Nth iteration."""
        ai = AdaptiveIntensity(reflect_every_n=3, stagnation_window=5)
        # Record steadily improving rewards (enough to exceed stagnation window)
        for i in range(6):
            ai.record_reward(1.0 + i * 0.5)

        # iteration 0 -> always True (tested separately)
        # iteration 1 -> not 0, has data, not stagnating, 1%3!=0 -> False
        assert ai.should_reflect(1) is False
        # iteration 2 -> 2%3!=0 -> False
        assert ai.should_reflect(2) is False
        # iteration 3 -> 3%3==0 -> True
        assert ai.should_reflect(3) is True
        # iteration 4 -> 4%3!=0 -> False
        assert ai.should_reflect(4) is False

    def test_stagnating_always_reflects(self) -> None:
        """Flat rewards cause stagnation, so should_reflect is always True."""
        ai = AdaptiveIntensity(stagnation_window=5, stagnation_threshold=0.02)
        # Record flat rewards
        for _ in range(6):
            ai.record_reward(1.0)

        # Every iteration should reflect because stagnation is detected
        for iteration in range(10):
            assert ai.should_reflect(iteration) is True


class TestIsStagnating:
    def test_is_stagnating(self) -> None:
        """Flat rewards within threshold -> stagnating."""
        ai = AdaptiveIntensity(stagnation_window=5, stagnation_threshold=0.02)
        for _ in range(5):
            ai.record_reward(1.0)
        assert ai.is_stagnating() is True

    def test_not_stagnating_with_improvement(self) -> None:
        """Increasing rewards -> not stagnating."""
        ai = AdaptiveIntensity(stagnation_window=5, stagnation_threshold=0.02)
        for i in range(5):
            ai.record_reward(1.0 + i * 0.1)
        assert ai.is_stagnating() is False

    def test_not_stagnating_insufficient_data(self) -> None:
        """Too few data points -> not stagnating."""
        ai = AdaptiveIntensity(stagnation_window=5)
        ai.record_reward(1.0)
        ai.record_reward(1.0)
        assert ai.is_stagnating() is False


class TestImprovementSignal:
    def test_improvement_signal(self) -> None:
        """Positive signal when improving."""
        ai = AdaptiveIntensity()
        ai.record_reward(1.0)
        ai.record_reward(1.5)
        assert ai.improvement_signal() == 0.5

    def test_improvement_signal_no_data(self) -> None:
        """0.0 when no data recorded."""
        ai = AdaptiveIntensity()
        assert ai.improvement_signal() == 0.0
