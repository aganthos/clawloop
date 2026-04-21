"""Learning loop: collect episodes -> forward_backward -> optim_step -> repeat.

The loop is benchmark-agnostic. It delegates episode collection to an
``AdapterLike`` and learning to the Layer protocol on each layer.
Gating (regression checks) is intentionally *not* part of the inner
loop -- see ``gate.py``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from clawloop.archive.store import ArchiveStore
from clawloop.core.archive_recorder import ArchiveRecorder
from clawloop.core.episode import Episode
from clawloop.core.evolution_log import EvolutionEntry, EvolutionLog
from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.runner import EpisodeCollectorRunner
from clawloop.core.state import StateID
from clawloop.core.transaction import LayerTransaction
from clawloop.core.types import FBResult
from clawloop.learning_layers.harness import Harness
from clawloop.learning_layers.router import Router
from clawloop.learning_layers.weights import Weights

log = logging.getLogger(__name__)


LAYER_NAMES = ("harness", "router", "weights")


class ExperimentLog:
    """Append-only JSONL experiment logger.

    Writes one JSON line per iteration to ``<output_dir>/experiment.jsonl``.
    Each line contains: iteration, timestamp, rewards, playbook snapshot,
    insights generated, fb/optim results.  Designed to survive crashes
    (flush after each write).
    """

    def __init__(
        self,
        output_dir: str | Path | None = None,
        *,
        wandb_project: str | None = None,
        wandb_name: str | None = None,
    ):
        self._path: Path | None = None
        if output_dir:
            self._path = Path(output_dir) / "experiment.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)

        # Optional cookbook logger: always emits a Rich console table; opt-in
        # for wandb / neptune via setup_logging args + env vars. We keep our
        # own experiment.jsonl as the canonical disk-of-record; the cookbook
        # logger ALSO writes its own metrics.jsonl alongside.
        self._ml_logger: Any | None = None
        if output_dir:
            try:
                from tinker_cookbook.utils import ml_log

                self._ml_logger = ml_log.setup_logging(
                    log_dir=str(Path(output_dir).expanduser()),
                    wandb_project=wandb_project,
                    wandb_name=wandb_name,
                    config=None,
                )
            except Exception as e:
                # Best-effort — never abort training because Rich/wandb failed.
                log.warning("ml_log.setup_logging failed: %s; falling back to JSONL only", e)
                self._ml_logger = None

    def log_iteration(
        self,
        iteration: int,
        episodes: list[Episode],
        fb_results: dict[str, FBResult],
        harness: Harness | None = None,
        backend: Any | None = None,
    ) -> None:
        if self._path is None:
            return
        rewards = [ep.summary.total_reward for ep in episodes]
        # Aggregate across GRPO duplicates (the same task_id can appear K times
        # in one iter — we don't want the dict-comprehension last-wins
        # behaviour to drop K-1 rollouts). Emit summary stats per task_id +
        # the list of rewards so downstream viz can plot per-rollout if it
        # wants.
        by_task: dict[str, list] = {}
        for ep in episodes:
            by_task.setdefault(ep.task_id, []).append(ep)
        per_task: dict[str, dict] = {}
        for tid, eps in by_task.items():
            task_rewards = [e.summary.total_reward for e in eps]
            errors = [
                e.metadata.get("error") for e in eps if e.metadata and e.metadata.get("error")
            ]
            # Latest episode's signals in the original {value, confidence} shape —
            # keeps the existing viewer (`learning_viewer.html` reads
            # `info.signals.<name>.value`) working without a simultaneous viewer
            # rewrite. All K rollouts' signals are preserved in `rollouts[]`
            # for the eventual viewer revamp.
            latest = eps[-1]
            latest_signals: dict[str, dict[str, Any]] = {}
            if latest.summary.signals:
                for k, s in latest.summary.signals.items():
                    latest_signals[k] = {
                        "value": s.value,
                        "confidence": s.confidence,
                    }
            rollouts = [
                {
                    "reward": e.summary.total_reward,
                    "error": (e.metadata.get("error") if e.metadata else None),
                    "signals": {
                        k: {"value": s.value, "confidence": s.confidence}
                        for k, s in (e.summary.signals or {}).items()
                    },
                }
                for e in eps
            ]
            mean_reward = sum(task_rewards) / len(task_rewards)
            per_task[tid] = {
                # Backward-compatible keys (viewer + existing consumers):
                "reward": mean_reward,
                "signals": latest_signals,
                "error": errors[0] if errors else None,
                # New per-rollout detail (issue #66 viewer work will use these):
                "n_rollouts": len(eps),
                "reward_mean": mean_reward,
                "reward_min": min(task_rewards),
                "reward_max": max(task_rewards),
                "rewards": task_rewards,
                "errors": errors,
                "rollouts": rollouts,
            }
        entry: dict[str, Any] = {
            "iteration": iteration,
            "timestamp": time.time(),
            "n_episodes": len(episodes),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "per_task": per_task,
            "fb_results": {
                name: {"status": r.status, "metrics": r.metrics} for name, r in fb_results.items()
            },
        }
        if backend is not None and hasattr(backend, "list_tinker_checkpoints"):
            try:
                entry["tinker_checkpoints"] = backend.list_tinker_checkpoints()
            except Exception as e:  # best-effort — never abort the run
                entry["tinker_checkpoints"] = [{"error": type(e).__name__, "message": str(e)}]
            entry["tinker_model_id"] = getattr(backend, "model_id", None)
            entry["tinker_durable_paths"] = list(getattr(backend, "_durable_paths", []))
        if harness is not None:
            entry["playbook_size"] = len(harness.playbook.entries)
            entry["playbook_entries"] = [
                {
                    "id": e.id,
                    "content": e.content[:200],
                    "helpful": e.helpful,
                    "harmful": e.harmful,
                    "tags": e.tags,
                }
                for e in harness.playbook.entries
            ]
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
        log.info(
            "  [log] iter=%d avg=%.4f min=%.4f max=%.4f playbook=%d",
            iteration,
            entry["avg_reward"],
            entry["min_reward"],
            entry["max_reward"],
            entry.get("playbook_size", 0),
        )
        # Mirror to cookbook ml_log (Rich console + opt-in wandb / neptune).
        if self._ml_logger is not None:
            try:
                # Flatten to scalar metrics — wandb/Rich expect numbers, not nested dicts.
                scalar_metrics: dict[str, Any] = {
                    "n_episodes": entry["n_episodes"],
                    "avg_reward": entry["avg_reward"],
                    "min_reward": entry["min_reward"],
                    "max_reward": entry["max_reward"],
                }
                for name, r in fb_results.items():
                    for mk, mv in (r.metrics or {}).items():
                        if isinstance(mv, (int, float, bool)):
                            scalar_metrics[f"{name}/{mk}"] = mv
                if "playbook_size" in entry:
                    scalar_metrics["playbook/size"] = entry["playbook_size"]
                self._ml_logger.log_metrics(scalar_metrics, step=iteration)
            except Exception as e:
                log.warning("ml_logger.log_metrics failed: %s", e)


@dataclass
class AgentState:
    """Bundle of the three mutable learning layers."""

    harness: Harness = field(default_factory=Harness)
    router: Router = field(default_factory=Router)
    weights: Weights = field(default_factory=Weights)
    inference_url: str | None = None  # vLLM endpoint for Harbor agents
    # Tinker SamplingClient, set per iter by TinkerWeightsBackend; kept Any to avoid tinker import.
    sampling_client: Any = None
    renderer: Any = None  # tinker_cookbook renderer; set per iter by TinkerWeightsBackend.
    tokenizer: Any = (
        None  # Tinker training-client tokenizer; set per iter by TinkerWeightsBackend.
    )
    tried_paradigms: list[str] = field(default_factory=list)  # paradigm contents tried
    _prev_playbook_generation: int = 0  # tracks generation for flush logic

    def state_id(self) -> StateID:
        return StateID.from_layers(self.harness, self.router, self.weights)

    def get_layers(
        self,
        active: list[str] | None = None,
    ) -> list[tuple[str, Any]]:
        """Return (name, layer) pairs, filtered by *active* if given."""
        all_layers = [(name, getattr(self, name)) for name in LAYER_NAMES]
        if active is None:
            return all_layers
        return [(n, layer) for n, layer in all_layers if n in active]


class AdapterLike(Protocol):
    def run_episode(self, task: Any, agent_state: AgentState) -> Episode: ...


def _avg_reward(episodes: list[Episode]) -> float:
    """Mean ``total_reward`` across episodes; 0.0 when empty."""
    if not episodes:
        return 0.0
    return sum(ep.summary.total_reward for ep in episodes) / len(episodes)


def _refresh_tinker_fields(
    agent_state: AgentState,
    active_layers: list[str] | None,
) -> tuple[AgentState, list[tuple[str, Any]]]:
    """Refresh Tinker-driven fields on ``AgentState`` from current backend.

    Gated on ``hasattr`` so non-Tinker backends (e.g. SkyRL) are untouched.
    """
    backend = getattr(agent_state.weights, "_backend", None)
    if backend is None or not hasattr(backend, "current_sampling_client"):
        return agent_state, agent_state.get_layers(active_layers)

    from dataclasses import replace as _replace

    agent_state = _replace(
        agent_state,
        sampling_client=backend.current_sampling_client(),
        renderer=getattr(backend, "renderer", None),
        tokenizer=getattr(backend, "tokenizer", None),
    )
    return agent_state, agent_state.get_layers(active_layers)


def _set_evolver_context(
    agent_state: AgentState,
    intensity: AdaptiveIntensity | None,
    iteration: int,
) -> None:
    """Push reward history + stagnation signal onto the harness's evolver."""
    if not isinstance(agent_state.harness, Harness) or agent_state.harness.evolver is None:
        return
    from clawloop.core.evolver import EvolverContext

    agent_state.harness.set_evolver_context(
        EvolverContext(
            reward_history=list(intensity._rewards) if intensity else [],
            is_stagnating=intensity.is_stagnating() if intensity else False,
            iteration=iteration,
            tried_paradigms=list(agent_state.tried_paradigms),
        )
    )


def _maybe_save_backend_state(
    agent_state: AgentState,
    weights_fb: FBResult | None,
    iteration: int,
) -> None:
    """Tinker save_state hook — persist intermediate weights between iters.

    Skips when the weights step produced zero datums (GRPO filtered every
    group for zero variance) to avoid polluting the durable-path timeline.
    """
    backend = getattr(agent_state.weights, "_backend", None)
    if backend is None or not hasattr(backend, "save_state"):
        return
    n_datums = weights_fb.metrics.get("n_datums", 0) if weights_fb and weights_fb.metrics else 0
    if n_datums > 0:
        try:
            backend.save_state(f"iter_{iteration}").result()
        except Exception:
            log.exception("backend.save_state failed for iter_%d", iteration)
    else:
        log.info(
            "  [save_state] skipped iter_%d — no datums produced (GRPO filtered all groups)",
            iteration,
        )


def _flush_generation_if_advanced(agent_state: AgentState, optim_failed: bool) -> None:
    """Drop stale weights-buffer episodes when the playbook generation advances.

    Prevents RL from learning against pre-adaptation behavior. No-op when
    optim failed (we already rolled back) or harness isn't a ``Harness``.
    """
    if optim_failed or not isinstance(agent_state.harness, Harness):
        return
    current_gen = agent_state.harness.playbook_generation
    prev_gen = agent_state._prev_playbook_generation
    if current_gen > prev_gen:
        stale = agent_state.weights.pending_advantage_count()
        agent_state.weights.clear_pending_state()
        log.info(
            "  Generation %d->%d: flushed %d stale episodes from weights buffer",
            prev_gen,
            current_gen,
            stale,
        )
    agent_state._prev_playbook_generation = current_gen


def _log_evolution_entry(
    evo_log: EvolutionLog,
    agent_state: AgentState,
    fb_results: dict[str, FBResult],
    prev_hash: str,
    new_hash: str,
    prev_avg_reward: float,
    avg_reward: float,
    iteration: int,
) -> None:
    """Append one ``EvolutionEntry`` iff any fb result produced a named action."""
    actions: list[str] = []
    for result in fb_results.values():
        if result.status != "ok":
            continue
        if result.metrics.get("insights_generated"):
            actions.append("reflect")
        if result.metrics.get("candidates_generated"):
            actions.append("mutate")
        if result.metrics.get("paradigm_shifted"):
            actions.append("paradigm_shift")
    if not actions:
        return
    backend = (
        agent_state.harness.evolver.name()
        if isinstance(agent_state.harness, Harness) and agent_state.harness.evolver
        else "none"
    )
    evo_log.append(
        EvolutionEntry(
            iteration=iteration,
            state_hash_before=prev_hash,
            state_hash_after=new_hash,
            actions=actions,
            reward_before=prev_avg_reward,
            reward_after=avg_reward,
            backend=backend,
        )
    )


def _run_after_iteration(
    cb: "Callable[[int, AgentState, list[Episode]], None] | None",
    iteration: int,
    agent_state: AgentState,
    episodes: list[Episode],
) -> None:
    """Invoke the optional after-iteration callback, logging any exception."""
    if cb is None:
        return
    try:
        cb(iteration, agent_state, episodes)
    except Exception:
        log.exception("after_iteration callback failed")


def learning_loop(
    adapter: AdapterLike,
    agent_state: AgentState,
    tasks: list[Any],
    n_episodes: int,
    n_iterations: int,
    *,
    active_layers: list[str] | None = None,
    intensity: AdaptiveIntensity | None = None,
    output_dir: str | Path | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
    after_iteration: "Callable[[int, AgentState, list[Episode]], None] | None" = None,
    archive: ArchiveStore | None = None,
    bench: str = "unknown",
    domain_tags: list[str] | None = None,
) -> tuple[AgentState, StateID]:
    """Run the unified learning loop.

    Parameters
    ----------
    adapter:
        Environment adapter that produces episodes.
    agent_state:
        Initial layer configuration.
    tasks:
        Pool of tasks to sample from.
    n_episodes:
        Number of episodes to collect per iteration.
    n_iterations:
        Number of learning iterations.
    active_layers:
        Which layers to train. None means all three.
    intensity:
        Optional adaptive intensity controller that gates when the
        evolver fires (saves LLM calls).
    after_iteration:
        Optional callback ``f(iteration, agent_state, episodes)`` called
        after each iteration completes (e.g. for eval scoring).

    Returns
    -------
    tuple[AgentState, StateID]
        The final agent state and its content-addressed state ID.
    """
    state_id = agent_state.state_id()
    layers = agent_state.get_layers(active_layers)
    exp_log = ExperimentLog(
        output_dir,
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    evo_log = EvolutionLog(output_dir)
    runner = EpisodeCollectorRunner(adapter)
    recorder = ArchiveRecorder(
        archive,
        agent_state,
        bench=bench,
        domain_tags=domain_tags,
        n_iterations=n_iterations,
    )
    prev_avg_reward = 0.0
    log.info("Starting learning loop — initial state: %s", state_id.combined_hash[:12])

    for iteration in range(n_iterations):
        log.info("Iteration %d/%d", iteration + 1, n_iterations)

        agent_state, layers = _refresh_tinker_fields(agent_state, active_layers)
        episodes: list[Episode] = runner.collect(agent_state, tasks, n_episodes)
        avg_reward = _avg_reward(episodes)
        log.info("  Collected %d episodes, avg reward: %.4f", len(episodes), avg_reward)
        recorder.record_episodes(iteration, episodes)
        if intensity is not None:
            intensity.record_reward(avg_reward)

        _set_evolver_context(agent_state, intensity, iteration)

        tx_result = LayerTransaction(
            layers,
            intensity=intensity,
            episodes=episodes,
            agent_state=agent_state,
        ).run(iteration)
        fb_results = tx_result.fb_results

        _maybe_save_backend_state(agent_state, fb_results.get("weights"), iteration)
        _flush_generation_if_advanced(agent_state, tx_result.optim_failed)

        harness_ref = agent_state.harness if isinstance(agent_state.harness, Harness) else None
        backend = getattr(agent_state.weights, "_backend", None)
        exp_log.log_iteration(iteration, episodes, fb_results, harness_ref, backend=backend)

        prev_hash = state_id.combined_hash
        state_id = agent_state.state_id()
        _log_evolution_entry(
            evo_log,
            agent_state,
            fb_results,
            prev_hash,
            state_id.combined_hash,
            prev_avg_reward,
            avg_reward,
            iteration,
        )

        recorder.record_iteration(
            iteration=iteration,
            agent_state=agent_state,
            state_id=state_id,
            fb_results=fb_results,
            episodes=episodes,
            avg_reward=avg_reward,
            prev_avg_reward=prev_avg_reward,
        )
        prev_avg_reward = avg_reward

        _run_after_iteration(after_iteration, iteration, agent_state, episodes)

    recorder.record_complete()
    log.info("Loop complete — final state: %s", state_id.combined_hash[:12])
    return agent_state, state_id
