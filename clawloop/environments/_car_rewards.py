# clawloop/environments/_car_rewards.py
"""CAR-bench reward mapping — converts CAR metrics to clawloop RewardSignals."""

from __future__ import annotations

import logging
from typing import Any

from clawloop.core.reward import RewardSignal

log = logging.getLogger(__name__)

DEFAULT_CAR_WEIGHTS: dict[str, float] = {
    "r_actions_final": 0.30,
    "r_actions_intermediate": 0.20,
    "r_tool_subset": 0.15,
    "r_tool_execution_errors": 0.15,
    "r_policy_errors": 0.10,
    "r_user_end_conversation": 0.10,
}


def map_car_scores(
    reward_info: dict[str, Any],
    task_reward: float,
    weights: dict[str, float] = DEFAULT_CAR_WEIGHTS,
) -> tuple[dict[str, RewardSignal], dict[str, Any]]:
    """Map CAR-bench metrics to clawloop RewardSignals.

    Parameters
    ----------
    reward_info:
        Per-metric dict from CAR's detailed_results (e.g. r_actions_final: 0/1).
    task_reward:
        Binary task reward (0.0 or 1.0) from CAR's top-level scoring.
    weights:
        Metric weights for composite score. Defaults to DEFAULT_CAR_WEIGHTS.

    Returns
    -------
    tuple of (signals dict, breakdown dict)
        signals: named RewardSignals for the learning loop
        breakdown: raw validated values for score_breakdown
    """
    signals: dict[str, RewardSignal] = {}
    breakdown: dict[str, Any] = {}

    # Primary outcome from task_reward (binary, ground truth)
    clamped_reward = max(0.0, min(1.0, float(task_reward)))
    signals["outcome"] = RewardSignal(
        name="outcome",
        value=clamped_reward * 2.0 - 1.0,
        confidence=1.0,
    )

    # Per-metric signals
    for name in weights:
        raw = reward_info.get(name)
        if raw is None:
            log.debug("Missing CAR metric %s", name)
            continue
        if not isinstance(raw, (int, float)):
            log.warning("Non-numeric CAR metric %s=%r, skipping", name, raw)
            continue
        val = max(0.0, min(1.0, float(raw)))
        conf = 1.0 if val in (0.0, 1.0) else 0.8
        signals[name] = RewardSignal(name=name, value=val * 2.0 - 1.0, confidence=conf)
        breakdown[name] = val

    # Unknown metrics: store but don't map
    for k, v in reward_info.items():
        if k not in weights:
            breakdown[k] = v

    return signals, breakdown
