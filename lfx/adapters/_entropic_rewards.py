# lfx/adapters/_entropic_rewards.py
"""Entropic CRMArenaPro reward mapping — 7-dimension scores to lfx RewardSignals."""

from __future__ import annotations

import logging
from typing import Any

from lfx.core.reward import RewardSignal

log = logging.getLogger(__name__)

# Weights mirror the green-agent scorer (crm/scorer.py SevenDimensionScorer).
DEFAULT_ENTROPIC_WEIGHTS: dict[str, float] = {
    "functional": 0.30,
    "drift_adaptation": 0.20,
    "token_efficiency": 0.12,
    "query_efficiency": 0.12,
    "error_recovery": 0.08,
    "trajectory_efficiency": 0.10,
    "hallucination_rate": 0.08,
}


def map_entropic_scores(
    scores: dict[str, Any],
    task_reward: float,
    weights: dict[str, float] = DEFAULT_ENTROPIC_WEIGHTS,
) -> tuple[dict[str, RewardSignal], dict[str, Any]]:
    """Map Entropic CRMArenaPro 7-dimension scores to lfx RewardSignals.

    Parameters
    ----------
    scores:
        Per-dimension dict from the green agent's scorer (0-100 scale).
    task_reward:
        The ``functional`` score as a 0/1 binary (CRMArena ground truth).
    weights:
        Dimension weights for composite score.

    Returns
    -------
    tuple of (signals dict, breakdown dict)
    """
    signals: dict[str, RewardSignal] = {}
    breakdown: dict[str, Any] = {}

    # Primary outcome from task_reward (binary ground truth)
    clamped_reward = max(0.0, min(1.0, float(task_reward)))
    signals["outcome"] = RewardSignal(
        name="outcome",
        value=clamped_reward * 2.0 - 1.0,
        confidence=1.0,
    )

    # Per-dimension signals (scores arrive on 0-100 scale, normalise to [0,1])
    for name in weights:
        raw = scores.get(name)
        if raw is None:
            log.debug("Missing entropic dimension %s", name)
            continue
        if not isinstance(raw, (int, float)):
            log.warning("Non-numeric entropic dimension %s=%r, skipping", name, raw)
            continue
        val = max(0.0, min(1.0, float(raw) / 100.0))
        conf = 1.0 if val in (0.0, 1.0) else 0.8
        signals[name] = RewardSignal(name=name, value=val * 2.0 - 1.0, confidence=conf)
        breakdown[name] = val

    # Unknown dimensions: store but don't map
    for k, v in scores.items():
        if k not in weights:
            breakdown[k] = v

    return signals, breakdown
