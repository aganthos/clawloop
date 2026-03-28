"""Adaptive reflection intensity — SkyDiscover-inspired scheduling.

Controls when the Reflector fires based on the improvement signal.
Saves LLM calls when things are going well; reflects more aggressively
when progress stagnates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class AdaptiveIntensity:
    """Decides whether the reflector should run on a given iteration.

    Parameters
    ----------
    reflect_every_n:
        Base cadence — reflect every *n*-th iteration when things look fine.
    stagnation_window:
        Number of recent rewards to inspect for stagnation.
    stagnation_threshold:
        If ``max - min`` of the last *stagnation_window* rewards is below
        this value, the agent is considered stagnating.
    cooldown_after_request:
        Seconds to wait after the last user request before allowing
        reflection. Prevents quality dips during active interaction.
    """

    reflect_every_n: int = 3
    stagnation_window: int = 5
    stagnation_threshold: float = 0.02
    cooldown_after_request: float = 30.0
    _rewards: list[float] = field(default_factory=list)
    _last_user_request: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_reward(self, avg_reward: float) -> None:
        """Append an observed average reward to the history."""
        self._rewards.append(avg_reward)

    def record_user_activity(self) -> None:
        """Record that the user just sent a request."""
        self._last_user_request = time.time()

    def should_reflect(self, iteration: int) -> bool:
        """Return whether the reflector should fire at *iteration*.

        Rules (evaluated in order):
        0. Defer if user is active (within cooldown window).
        1. Always reflect on the very first iteration.
        2. Always reflect when there are fewer than 2 recorded rewards.
        3. Always reflect when the agent is stagnating.
        4. Otherwise reflect only on every *reflect_every_n*-th iteration.
        """
        if self._last_user_request > 0:
            elapsed = time.time() - self._last_user_request
            if elapsed < self.cooldown_after_request:
                return False
        if iteration == 0 or len(self._rewards) < 2:
            return True
        if self.is_stagnating():
            return True
        return iteration % self.reflect_every_n == 0

    def is_stagnating(self) -> bool:
        """Return ``True`` if recent rewards show negligible variation."""
        if len(self._rewards) < self.stagnation_window:
            return False
        recent = self._rewards[-self.stagnation_window :]
        return max(recent) - min(recent) < self.stagnation_threshold

    def improvement_signal(self) -> float:
        """Return the latest single-step reward delta.

        Returns 0.0 when fewer than two rewards have been recorded.
        """
        if len(self._rewards) < 2:
            return 0.0
        return self._rewards[-1] - self._rewards[-2]
