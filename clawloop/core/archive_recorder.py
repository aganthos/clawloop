"""Archive-store recorder extracted from ``learning_loop``.

The recorder encapsulates all writes to an ``ArchiveStore`` during a run
and owns the run-level counters (``run_id``, ``best_reward``,
``initial_reward``, ``total_cost``, ``prev_variant_hash``) that would
otherwise pollute ``learning_loop``'s local scope.

All writes are wrapped in ``try / log.warning(..., exc_info=True)`` —
archive failures are non-fatal by design so that training continues
even if the backing store is misbehaving.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from clawloop.archive.null_store import NullArchiveStore
from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)
from clawloop.archive.store import ArchiveStore
from clawloop.core.episode import Episode
from clawloop.core.state import StateID
from clawloop.core.types import FBResult
from clawloop.learning_layers.harness import Harness
from clawloop.utils.content_hash import canonical_hash

if TYPE_CHECKING:
    from clawloop.core.loop import AgentState

log = logging.getLogger(__name__)


def _build_agent_config(agent_state: "AgentState") -> dict[str, Any]:
    """Serializable snapshot of the agent's config for archive identity."""
    config: dict[str, Any] = {}
    if isinstance(agent_state.harness, Harness):
        config["system_prompts"] = dict(agent_state.harness.system_prompts)
        config["playbook"] = agent_state.harness.playbook.to_dict()
    config["router"] = agent_state.router.to_dict()
    return config


def _variant_from_state(
    agent_state: "AgentState",
    variant_hash: str,
    run_id: str,
    created_at: float,
) -> AgentVariant:
    harness = agent_state.harness
    is_harness = isinstance(harness, Harness)
    return AgentVariant(
        variant_hash=variant_hash,
        system_prompt=(next(iter(harness.system_prompts.values()), "") if is_harness else ""),
        playbook_snapshot=harness.playbook.to_dict() if is_harness else {},
        model="",
        tools=[],
        first_seen_run_id=run_id,
        created_at=created_at,
    )


class ArchiveRecorder:
    """Owns all ``ArchiveStore`` writes for one ``learning_loop`` run.

    Construction writes the initial ``RunRecord`` + ``AgentVariant``. Each
    iteration calls :meth:`record_episodes` and :meth:`record_iteration`.
    :meth:`record_complete` writes the terminal ``log_run_complete`` row
    with accumulated best-reward / improvement-delta / total-cost metrics.
    """

    def __init__(
        self,
        archive: ArchiveStore | None,
        agent_state: "AgentState",
        bench: str,
        domain_tags: list[str] | None,
        n_iterations: int,
    ) -> None:
        self._store: ArchiveStore = archive if archive is not None else NullArchiveStore()
        self._run_id = RunRecord.new_id()
        self._best_reward: float | None = None
        self._initial_reward: float | None = None
        self._total_cost = 0

        initial_config = _build_agent_config(agent_state)
        initial_hash = canonical_hash(initial_config)
        self._prev_variant_hash = initial_hash

        now = time.time()
        try:
            self._store.log_run_start(
                RunRecord(
                    run_id=self._run_id,
                    bench=bench,
                    domain_tags=list(domain_tags or []),
                    agent_config=initial_config,
                    config_hash=initial_hash,
                    n_iterations=n_iterations,
                    best_reward=0.0,
                    improvement_delta=0.0,
                    total_cost_tokens=0,
                    parent_run_id=None,
                    created_at=now,
                    completed_at=None,
                )
            )
        except Exception:
            log.warning("Archive: failed to log run start", exc_info=True)

        try:
            self._store.log_variant(
                _variant_from_state(agent_state, initial_hash, self._run_id, now)
            )
        except Exception:
            log.warning("Archive: failed to log initial variant", exc_info=True)

    @property
    def run_id(self) -> str:
        """Stable run-id used for all records emitted by this recorder."""
        return self._run_id

    def record_episodes(self, iteration: int, episodes: list[Episode]) -> None:
        """Write one ``EpisodeRecord`` per episode. No-op when ``episodes`` is empty."""
        if not episodes:
            return
        records: list[EpisodeRecord] = []
        for ep in episodes:
            tool_call_count = sum(len(m.tool_calls or []) for m in ep.messages)
            token_usage: dict[str, Any]
            if ep.summary.token_usage is not None:
                token_usage = {
                    "prompt_tokens": ep.summary.token_usage.prompt_tokens,
                    "completion_tokens": ep.summary.token_usage.completion_tokens,
                    "total_tokens": ep.summary.token_usage.total_tokens,
                }
            else:
                token_usage = {}
            records.append(
                EpisodeRecord(
                    run_id=self._run_id,
                    iteration_num=iteration,
                    episode_id=ep.id,
                    task_id=ep.task_id,
                    bench=ep.bench,
                    model=ep.model or "",
                    reward=ep.summary.normalized_reward(),
                    signals=(
                        {
                            k: {"value": s.value, "confidence": s.confidence}
                            for k, s in ep.summary.signals.items()
                        }
                        if ep.summary.signals
                        else {}
                    ),
                    n_steps=ep.n_steps(),
                    n_tool_calls=tool_call_count,
                    token_usage=token_usage,
                    latency_ms=int(ep.summary.timing.total_ms if ep.summary.timing else 0),
                    messages_ref=f"traces/{ep.id}.json",
                    created_at=ep.created_at if ep.created_at is not None else time.time(),
                )
            )
        try:
            self._store.log_episodes(records)
        except Exception:
            log.warning("Archive: failed to log episodes", exc_info=True)

    def record_iteration(
        self,
        iteration: int,
        agent_state: "AgentState",
        state_id: StateID,
        fb_results: dict[str, FBResult],
        episodes: list[Episode],
        avg_reward: float,
        prev_avg_reward: float,
    ) -> None:
        """Write the per-iteration ``IterationRecord`` and, if the agent's
        config hash changed, a new ``AgentVariant``. Also updates
        ``best_reward`` / ``initial_reward`` / ``total_cost`` counters used
        by :meth:`record_complete`.
        """
        iter_cost = sum(
            r.metrics.get("tokens_used", 0) for r in fb_results.values() if r.status == "ok"
        )
        self._total_cost += iter_cost

        try:
            cur_config = _build_agent_config(agent_state)
            cur_variant_hash = canonical_hash(cur_config)
            evolver_action: dict[str, Any] = {}
            for name, result in fb_results.items():
                if result.status == "ok":
                    evolver_action[name] = result.metrics
            self._store.log_iteration(
                IterationRecord(
                    run_id=self._run_id,
                    iteration_num=iteration,
                    harness_snapshot_hash=state_id.harness_hash,
                    mean_reward=avg_reward,
                    reward_trajectory=[ep.summary.normalized_reward() for ep in episodes],
                    evolver_action=evolver_action,
                    cost_tokens=iter_cost,
                    parent_variant_hash=self._prev_variant_hash,
                    child_variant_hash=cur_variant_hash,
                    reward_delta=avg_reward - prev_avg_reward,
                    created_at=time.time(),
                )
            )
            if cur_variant_hash != self._prev_variant_hash:
                self._store.log_variant(
                    _variant_from_state(agent_state, cur_variant_hash, self._run_id, time.time())
                )
                self._prev_variant_hash = cur_variant_hash
        except Exception:
            log.warning("Archive: failed to log iteration %d", iteration, exc_info=True)

        if self._best_reward is None or avg_reward > self._best_reward:
            self._best_reward = avg_reward
        if self._initial_reward is None:
            self._initial_reward = avg_reward

    def record_complete(self) -> None:
        """Write the terminal ``log_run_complete`` with accumulated metrics."""
        try:
            best = self._best_reward if self._best_reward is not None else 0.0
            initial = self._initial_reward if self._initial_reward is not None else 0.0
            self._store.log_run_complete(
                self._run_id,
                best,
                best - initial,
                total_cost_tokens=self._total_cost,
            )
        except Exception:
            log.warning("Archive: failed to log run complete", exc_info=True)
