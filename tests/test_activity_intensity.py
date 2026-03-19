"""Tests for activity-aware learning intensity (cooldown during user interaction)."""

import time
from unittest.mock import patch

from lfx.core.intensity import AdaptiveIntensity


class TestActivityCooldown:
    def test_cooldown_defers_reflection(self) -> None:
        """After record_user_activity(), should_reflect returns False within cooldown."""
        ai = AdaptiveIntensity(cooldown_after_request=30.0)
        ai.record_reward(0.5)
        ai.record_reward(0.5)

        ai.record_user_activity()
        # Still within the 30-second cooldown window
        assert ai.should_reflect(2) is False

    def test_expired_cooldown_allows_reflection(self) -> None:
        """After the cooldown window has passed, should_reflect returns True again."""
        ai = AdaptiveIntensity(cooldown_after_request=30.0, reflect_every_n=3)
        ai.record_reward(0.5)
        ai.record_reward(0.5)

        # Place user activity 60 seconds in the past — well past the 30s cooldown
        ai._last_user_request = time.time() - 60.0
        # iteration 3 hits the cadence (3 % 3 == 0), so cooldown is the only gate
        assert ai.should_reflect(3) is True

    def test_no_activity_does_not_defer(self) -> None:
        """With no user activity recorded (_last_user_request=0), cooldown is skipped."""
        ai = AdaptiveIntensity(cooldown_after_request=30.0)
        assert ai._last_user_request == 0.0

        # iteration 0 → always True (no cooldown gate because _last_user_request == 0)
        assert ai.should_reflect(0) is True

        # With enough rewards, normal cadence rules apply — no deferral
        ai.record_reward(0.5)
        ai.record_reward(0.6)
        ai.record_reward(0.7)
        # reflect_every_n=3 → iteration 3 should fire
        assert ai.should_reflect(3) is True

    def test_record_user_activity_updates_timestamp(self) -> None:
        """record_user_activity() sets _last_user_request to the current time."""
        ai = AdaptiveIntensity()
        assert ai._last_user_request == 0.0

        before = time.time()
        ai.record_user_activity()
        after = time.time()

        assert before <= ai._last_user_request <= after

    def test_short_cooldown_allows_quick_resume(self) -> None:
        """With a very short cooldown, reflection resumes almost immediately."""
        ai = AdaptiveIntensity(cooldown_after_request=0.01, reflect_every_n=3)
        ai.record_reward(0.5)
        ai.record_reward(0.5)

        # Place user activity 0.02s in the past — past the 0.01s cooldown
        ai._last_user_request = time.time() - 0.02
        # iteration 3 hits cadence, so only the cooldown gate matters
        assert ai.should_reflect(3) is True

    def test_cooldown_blocks_even_stagnation(self) -> None:
        """Cooldown takes priority over stagnation — no reflection while user is active."""
        ai = AdaptiveIntensity(
            cooldown_after_request=30.0,
            stagnation_window=5,
            stagnation_threshold=0.02,
        )
        # Record flat rewards to trigger stagnation
        for _ in range(6):
            ai.record_reward(1.0)
        assert ai.is_stagnating() is True

        # User is active — cooldown overrides stagnation
        ai._last_user_request = time.time()
        assert ai.should_reflect(5) is False

    def test_cooldown_blocks_iteration_zero(self) -> None:
        """Cooldown even prevents reflection on iteration 0."""
        ai = AdaptiveIntensity(cooldown_after_request=30.0)
        ai._last_user_request = time.time()
        assert ai.should_reflect(0) is False

    def test_mock_time_for_cooldown(self) -> None:
        """Use unittest.mock.patch to control time.time() for deterministic tests."""
        ai = AdaptiveIntensity(cooldown_after_request=30.0)
        ai.record_reward(0.5)
        ai.record_reward(0.5)

        with patch("lfx.core.intensity.time") as mock_time:
            # record_user_activity at t=1000
            mock_time.time.return_value = 1000.0
            ai.record_user_activity()

            # should_reflect called at t=1010 — within cooldown
            mock_time.time.return_value = 1010.0
            assert ai.should_reflect(2) is False

            # should_reflect called at t=1040 — past cooldown
            # iteration 3 hits cadence (3 % 3 == 0), so only cooldown gate matters
            mock_time.time.return_value = 1040.0
            assert ai.should_reflect(3) is True
