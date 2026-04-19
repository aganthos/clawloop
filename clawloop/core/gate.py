"""Regression gating — deploy-time safety check.

Gating compares metric *distributions* (not point estimates) between the
candidate state and the current production state.  A candidate must demonstrate
non-regression on **all** benchmarks before it is promoted.

This module is deliberately separate from the learning loop: gating only
applies when deploying to production, not every iteration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from clawloop.core.episode import Episode
from clawloop.core.state import StateID

log = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Outcome of a deploy-time regression gate."""

    passed: bool
    candidate_state: str
    production_state: str
    per_bench: dict[str, BenchGateResult]
    summary: str


@dataclass
class BenchGateResult:
    """Per-benchmark gate comparison."""

    bench: str
    candidate_mean: float
    production_mean: float
    delta: float
    passed: bool


def gate_for_deploy(
    candidate_state: StateID,
    production_state: StateID,
    candidate_episodes: list[Episode],
    production_episodes: list[Episode],
    *,
    min_episodes: int = 10,
    regression_threshold: float = 0.0,
) -> GateResult:
    """Check whether ``candidate_state`` is safe to deploy.

    Parameters
    ----------
    candidate_state, production_state:
        The state IDs being compared.
    candidate_episodes, production_episodes:
        Episodes collected under each state.
    min_episodes:
        Minimum episodes per bench required for a valid comparison.
    regression_threshold:
        Maximum allowed reward drop (negative delta) before a bench fails.

    Returns
    -------
    GateResult
    """
    # Group episodes by bench
    cand_by_bench = _group_by_bench(candidate_episodes)
    prod_by_bench = _group_by_bench(production_episodes)

    all_benches = set(cand_by_bench) | set(prod_by_bench)
    per_bench: dict[str, BenchGateResult] = {}
    all_passed = True

    for bench in sorted(all_benches):
        cand_eps = cand_by_bench.get(bench, [])
        prod_eps = prod_by_bench.get(bench, [])

        if len(cand_eps) < min_episodes or len(prod_eps) < min_episodes:
            log.warning(
                "Bench %s: insufficient episodes (candidate=%d, production=%d, min=%d)",
                bench,
                len(cand_eps),
                len(prod_eps),
                min_episodes,
            )
            result = BenchGateResult(
                bench=bench,
                candidate_mean=0.0,
                production_mean=0.0,
                delta=0.0,
                passed=False,
            )
            all_passed = False
        else:
            cand_mean = (
                sum(e.summary.total_reward for e in cand_eps) / len(cand_eps) if cand_eps else 0.0
            )
            prod_mean = (
                sum(e.summary.total_reward for e in prod_eps) / len(prod_eps) if prod_eps else 0.0
            )
            delta = cand_mean - prod_mean
            passed = delta >= regression_threshold
            if not passed:
                all_passed = False
            result = BenchGateResult(
                bench=bench,
                candidate_mean=cand_mean,
                production_mean=prod_mean,
                delta=delta,
                passed=passed,
            )
            log.info(
                "Bench %s: candidate=%.4f production=%.4f delta=%.4f -> %s",
                bench,
                cand_mean,
                prod_mean,
                delta,
                "PASS" if passed else "FAIL",
            )

        per_bench[bench] = result

    summary = "PASS" if all_passed else "FAIL"
    return GateResult(
        passed=all_passed,
        candidate_state=candidate_state.combined_hash,
        production_state=production_state.combined_hash,
        per_bench=per_bench,
        summary=summary,
    )


def _group_by_bench(episodes: list[Episode]) -> dict[str, list[Episode]]:
    groups: dict[str, list[Episode]] = {}
    for ep in episodes:
        groups.setdefault(ep.bench, []).append(ep)
    return groups
