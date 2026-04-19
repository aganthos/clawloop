"""Learning loop: collect episodes -> forward_backward -> optim_step -> repeat.

The loop is benchmark-agnostic. It delegates episode collection to an
``AdapterLike`` and learning to the Layer protocol on each layer.
Gating (regression checks) is intentionally *not* part of the inner
loop -- see ``gate.py``.
"""

from __future__ import annotations

import copy
import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from clawloop.archive.null_store import NullArchiveStore
from clawloop.archive.schema import (
    AgentVariant,
    EpisodeRecord,
    IterationRecord,
    RunRecord,
)
from clawloop.archive.store import ArchiveStore
from clawloop.core.episode import Episode
from clawloop.core.evolution_log import EvolutionEntry, EvolutionLog
from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.state import StateID
from clawloop.core.types import Datum, FBResult
from clawloop.learning_layers.harness import Harness
from clawloop.learning_layers.router import Router
from clawloop.learning_layers.weights import Weights
from clawloop.utils.content_hash import canonical_hash

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


def _build_agent_config(agent_state: AgentState) -> dict[str, Any]:
    """Extract serializable agent config snapshot for archive identity."""
    config: dict[str, Any] = {}
    if isinstance(agent_state.harness, Harness):
        config["system_prompts"] = dict(agent_state.harness.system_prompts)
        config["playbook"] = agent_state.harness.playbook.to_dict()
    config["router"] = agent_state.router.to_dict()
    return config


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
    _archive: ArchiveStore = archive if archive is not None else NullArchiveStore()
    _run_id = RunRecord.new_id()
    _initial_config = _build_agent_config(agent_state)
    _initial_variant_hash = canonical_hash(_initial_config)
    _now = time.time()
    try:
        _archive.log_run_start(
            RunRecord(
                run_id=_run_id,
                bench=bench,
                domain_tags=list(domain_tags or []),
                agent_config=_initial_config,
                config_hash=_initial_variant_hash,
                n_iterations=n_iterations,
                best_reward=0.0,
                improvement_delta=0.0,
                total_cost_tokens=0,
                parent_run_id=None,
                created_at=_now,
                completed_at=None,
            )
        )
    except Exception:
        log.warning("Archive: failed to log run start", exc_info=True)
    try:
        _archive.log_variant(
            AgentVariant(
                variant_hash=_initial_variant_hash,
                system_prompt=(
                    next(iter(agent_state.harness.system_prompts.values()), "")
                    if isinstance(agent_state.harness, Harness)
                    else ""
                ),
                playbook_snapshot=(
                    agent_state.harness.playbook.to_dict()
                    if isinstance(agent_state.harness, Harness)
                    else {}
                ),
                model="",
                tools=[],
                first_seen_run_id=_run_id,
                created_at=_now,
            )
        )
    except Exception:
        log.warning("Archive: failed to log initial variant", exc_info=True)
    _prev_variant_hash = _initial_variant_hash
    _best_reward: float | None = None  # None until first iteration — handles negative rewards
    _initial_reward: float | None = None
    _total_cost = 0
    prev_avg_reward = 0.0
    log.info("Starting learning loop — initial state: %s", state_id.combined_hash[:12])

    for iteration in range(n_iterations):
        log.info("Iteration %d/%d", iteration + 1, n_iterations)

        # 0. Refresh Tinker-driven fields on AgentState from current backend.
        # Gated on hasattr so non-Tinker backends (e.g. SkyRL) are untouched.
        backend = getattr(agent_state.weights, "_backend", None)
        if backend is not None and hasattr(backend, "current_sampling_client"):
            from dataclasses import replace as _replace

            agent_state = _replace(
                agent_state,
                sampling_client=backend.current_sampling_client(),
                renderer=getattr(backend, "renderer", None),
                tokenizer=getattr(backend, "tokenizer", None),
            )
            # Refresh the layers view against the (possibly) new agent_state.
            layers = agent_state.get_layers(active_layers)

        # 1. Collect episodes
        if not tasks or n_episodes <= 0:
            episodes: list[Episode] = []
        else:
            if n_episodes <= len(tasks):
                selected_tasks = random.sample(tasks, n_episodes)
            else:
                selected_tasks = random.choices(tasks, k=n_episodes)

            if hasattr(adapter, "run_batch") and callable(getattr(adapter, "run_batch", None)):
                episodes = adapter.run_batch(agent_state, selected_tasks)
            elif hasattr(adapter, "run_episodes_batch") and callable(
                getattr(adapter, "run_episodes_batch", None)
            ):
                # Concurrent rollout path (OpenSpielGameAdapter): all
                # episodes for this iter fan out under one event loop,
                # Tinker queues them as parallel ConcurrentFutures.
                episodes = adapter.run_episodes_batch(selected_tasks, agent_state)
            else:
                episodes = []
                for task in selected_tasks:
                    ep = adapter.run_episode(task, agent_state)
                    episodes.append(ep)

        avg_reward = (
            sum(ep.summary.total_reward for ep in episodes) / len(episodes) if episodes else 0.0
        )
        log.info("  Collected %d episodes, avg reward: %.4f", len(episodes), avg_reward)

        if episodes:
            _ep_records: list[EpisodeRecord] = []
            for ep in episodes:
                tool_call_count = sum(len(m.tool_calls or []) for m in ep.messages)
                _ep_records.append(
                    EpisodeRecord(
                        run_id=_run_id,
                        iteration_num=iteration,
                        episode_id=ep.id,
                        task_id=ep.task_id,
                        bench=ep.bench,
                        model=ep.model or "",
                        reward=ep.summary.normalized_reward(),
                        signals={
                            k: {"value": s.value, "confidence": s.confidence}
                            for k, s in ep.summary.signals.items()
                        }
                        if ep.summary.signals
                        else {},
                        n_steps=ep.n_steps(),
                        n_tool_calls=tool_call_count,
                        token_usage=(
                            {
                                "prompt_tokens": ep.summary.token_usage.prompt_tokens,
                                "completion_tokens": ep.summary.token_usage.completion_tokens,
                                "total_tokens": ep.summary.token_usage.total_tokens,
                            }
                            if ep.summary.token_usage
                            else {}
                        ),
                        latency_ms=int(ep.summary.timing.total_ms if ep.summary.timing else 0),
                        messages_ref=f"traces/{ep.id}.json",
                        created_at=ep.created_at if ep.created_at is not None else time.time(),
                    )
                )
            try:
                _archive.log_episodes(_ep_records)
            except Exception:
                log.warning("Archive: failed to log episodes", exc_info=True)

        # Record reward for adaptive intensity
        if intensity is not None:
            intensity.record_reward(avg_reward)

        # 2. Build per-layer datums
        # NOTE: Support-query split (failures→harness,
        # successes→weights) is disabled. GRPO needs all episodes for
        # advantage variance, and the on-policy vs off-policy boundary
        # after harness updates needs more work. See roadmap Task 2.1.
        layer_datums: dict[str, Datum] = {
            "harness": Datum(episodes=episodes),
            "weights": Datum(episodes=episodes),
            "router": Datum(episodes=episodes),
        }

        # 2b. Set evolver context on harness (for Evolver-based optimization)
        if isinstance(agent_state.harness, Harness) and agent_state.harness.evolver is not None:
            from clawloop.core.evolver import EvolverContext

            ctx = EvolverContext(
                reward_history=list(intensity._rewards) if intensity else [],
                is_stagnating=intensity.is_stagnating() if intensity else False,
                iteration=iteration,
                tried_paradigms=list(agent_state.tried_paradigms),
            )
            agent_state.harness.set_evolver_context(ctx)

        # 3. Phase 1: forward_backward (all active layers)
        fb_results: dict[str, FBResult] = {}
        for name, layer in layers:
            # Skip harness reflection when intensity says not to
            if (
                name == "harness"
                and intensity is not None
                and not intensity.should_reflect(iteration)
            ):
                log.info("  skipping harness fb (adaptive intensity)")
                fb_results[name] = FBResult(status="skipped")
                continue
            if name in layer_datums:
                datum = layer_datums[name]
            else:
                log.warning("  unknown layer %s — using all episodes as fallback", name)
                datum = Datum(episodes=episodes)
            should_clear = False
            try:
                fut = layer.forward_backward(datum)
                fb_result = fut.result()
                fb_results[name] = fb_result
                if fb_result.status in ("error", "skipped"):
                    should_clear = True
            except Exception:
                log.exception("forward_backward failed for %s", name)
                fb_results[name] = FBResult(status="error")
                should_clear = True

            if should_clear:
                try:
                    layer.clear_pending_state()
                except Exception:
                    log.exception("Failed to clear pending for %s", name)

        for name, result in fb_results.items():
            log.info("  fb %s: %s %s", name, result.status, result.metrics)

        # Track paradigm shifts before optim drains _pending
        harness_fb = fb_results.get("harness")
        if (
            harness_fb is not None
            and harness_fb.metrics.get("paradigm_shifted")
            and isinstance(agent_state.harness, Harness)
        ):
            for insight in agent_state.harness._pending.insights:
                if "paradigm" in (insight.tags or []):
                    agent_state.tried_paradigms.append(insight.content)

        # 4. Phase 2: optim_step with cross-layer rollback
        layers_to_optim = [
            (name, layer)
            for name, layer in layers
            if fb_results.get(name, FBResult(status="error")).status not in ("error", "skipped")
        ]

        # Snapshot all layers before optim (for cross-layer rollback)
        snapshots: dict[str, dict[str, Any]] = {}
        try:
            for name, layer in layers_to_optim:
                snapshots[name] = copy.deepcopy(layer.to_dict())
        except Exception:
            log.exception("Snapshot failed — skipping optim this iteration")
            for name, layer in layers_to_optim:
                try:
                    layer.clear_pending_state()
                except Exception:
                    log.exception("Failed to clear pending for %s", name)
            layers_to_optim = []

        optim_failed = False
        for name, layer in layers_to_optim:
            try:
                result = layer.optim_step().result()
                log.info(
                    "  optim %s: %s, %d updates",
                    name,
                    result.status,
                    result.updates_applied,
                )
                if result.status == "error":
                    optim_failed = True
                    log.error(
                        "  optim %s returned error — triggering rollback",
                        name,
                    )
                    break
            except Exception:
                log.exception(
                    "optim_step failed for %s — triggering rollback",
                    name,
                )
                optim_failed = True
                break

        if optim_failed:
            log.warning("  rolling back all layers to pre-optim state")
            for name, layer in layers_to_optim:
                if name in snapshots:
                    try:
                        lr = layer.load_state(snapshots[name]).result()
                        if lr.status != "ok":
                            log.error(
                                "  rollback returned %s for %s",
                                lr.status,
                                name,
                            )
                    except Exception:
                        log.exception("  rollback failed for %s", name)

        # 4b. Tinker save_state hook — persist intermediate weights and swap
        # the backend's internal SamplingClient so the next iter picks up the
        # freshly-trained adapter on its top-of-iter refresh.  Gated on
        # hasattr so SkyRL-style backends are unaffected.
        #
        # Skip the save entirely when the weights step produced zero datums
        # (GRPO filtered every group for zero variance) — checkpointing an
        # unchanged adapter wastes Tinker quota and pollutes the durable-path
        # timeline.
        weights_fb = fb_results.get("weights")
        n_datums = (
            weights_fb.metrics.get("n_datums", 0)
            if weights_fb is not None and weights_fb.metrics
            else 0
        )
        if backend is not None and hasattr(backend, "save_state") and n_datums > 0:
            try:
                backend.save_state(f"iter_{iteration}").result()
            except Exception:
                log.exception("backend.save_state failed for iter_%d", iteration)
        elif backend is not None and hasattr(backend, "save_state"):
            log.info(
                "  [save_state] skipped iter_%d — no datums produced "
                "(GRPO filtered all groups)",
                iteration,
            )

        # Generation flush: when playbook_generation advances, clear stale
        # episodes from weights buffer to prevent RL learning pre-adaptation behavior
        if isinstance(agent_state.harness, Harness) and not optim_failed:
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

        # 5. Log iteration results
        harness_ref = agent_state.harness if isinstance(agent_state.harness, Harness) else None
        exp_log.log_iteration(iteration, episodes, fb_results, harness_ref, backend=backend)

        # 6. Recompute state identity and log evolution entry
        prev_hash = state_id.combined_hash
        state_id = agent_state.state_id()

        # Build actions list from fb results for evolution log
        actions: list[str] = []
        for name, result in fb_results.items():
            if result.status == "ok":
                if result.metrics.get("insights_generated"):
                    actions.append("reflect")
                if result.metrics.get("candidates_generated"):
                    actions.append("mutate")
                if result.metrics.get("paradigm_shifted"):
                    actions.append("paradigm_shift")
        if actions:
            evo_log.append(
                EvolutionEntry(
                    iteration=iteration,
                    state_hash_before=prev_hash,
                    state_hash_after=state_id.combined_hash,
                    actions=actions,
                    reward_before=prev_avg_reward,
                    reward_after=avg_reward,
                    backend=(
                        agent_state.harness.evolver.name()
                        if isinstance(agent_state.harness, Harness) and agent_state.harness.evolver
                        else "none"
                    ),
                )
            )

        try:
            _cur_config = _build_agent_config(agent_state)
            _cur_variant_hash = canonical_hash(_cur_config)
            _evolver_action: dict[str, Any] = {}
            for name, result in fb_results.items():
                if result.status == "ok":
                    _evolver_action[name] = result.metrics
            _iter_cost = sum(
                r.metrics.get("tokens_used", 0) for r in fb_results.values() if r.status == "ok"
            )
            _total_cost += _iter_cost
            _archive.log_iteration(
                IterationRecord(
                    run_id=_run_id,
                    iteration_num=iteration,
                    harness_snapshot_hash=state_id.harness_hash,
                    mean_reward=avg_reward,
                    reward_trajectory=[ep.summary.normalized_reward() for ep in episodes],
                    evolver_action=_evolver_action,
                    cost_tokens=_iter_cost,
                    parent_variant_hash=_prev_variant_hash,
                    child_variant_hash=_cur_variant_hash,
                    reward_delta=avg_reward - prev_avg_reward,
                    created_at=time.time(),
                )
            )
            if _cur_variant_hash != _prev_variant_hash:
                _archive.log_variant(
                    AgentVariant(
                        variant_hash=_cur_variant_hash,
                        system_prompt=(
                            next(iter(agent_state.harness.system_prompts.values()), "")
                            if isinstance(agent_state.harness, Harness)
                            else ""
                        ),
                        playbook_snapshot=(
                            agent_state.harness.playbook.to_dict()
                            if isinstance(agent_state.harness, Harness)
                            else {}
                        ),
                        model="",
                        tools=[],
                        first_seen_run_id=_run_id,
                        created_at=time.time(),
                    )
                )
                _prev_variant_hash = _cur_variant_hash
        except Exception:
            log.warning("Archive: failed to log iteration %d", iteration, exc_info=True)

        if _best_reward is None or avg_reward > _best_reward:
            _best_reward = avg_reward
        if _initial_reward is None:
            _initial_reward = avg_reward

        prev_avg_reward = avg_reward

        # 7. Optional after-iteration callback (e.g. eval scoring)
        if after_iteration is not None:
            try:
                after_iteration(iteration, agent_state, episodes)
            except Exception:
                log.exception("after_iteration callback failed")

    try:
        final_best = _best_reward if _best_reward is not None else 0.0
        final_initial = _initial_reward if _initial_reward is not None else 0.0
        _archive.log_run_complete(
            _run_id,
            final_best,
            final_best - final_initial,
            total_cost_tokens=_total_cost,
        )
    except Exception:
        log.warning("Archive: failed to log run complete", exc_info=True)

    log.info("Loop complete — final state: %s", state_id.combined_hash[:12])
    return agent_state, state_id
