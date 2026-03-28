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
from typing import Any, Protocol

from clawloop.core.episode import Episode
from clawloop.core.evolution_log import EvolutionEntry, EvolutionLog
from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.state import StateID
from clawloop.core.types import Datum, FBResult, Future, OptimResult
from clawloop.layers.harness import Harness
from clawloop.layers.router import Router
from clawloop.layers.weights import Weights

log = logging.getLogger(__name__)


LAYER_NAMES = ("harness", "router", "weights")


class ExperimentLog:
    """Append-only JSONL experiment logger.

    Writes one JSON line per iteration to ``<output_dir>/experiment.jsonl``.
    Each line contains: iteration, timestamp, rewards, playbook snapshot,
    insights generated, fb/optim results.  Designed to survive crashes
    (flush after each write).
    """

    def __init__(self, output_dir: str | Path | None = None):
        self._path: Path | None = None
        if output_dir:
            self._path = Path(output_dir) / "experiment.jsonl"
            self._path.parent.mkdir(parents=True, exist_ok=True)

    def log_iteration(
        self,
        iteration: int,
        episodes: list[Episode],
        fb_results: dict[str, FBResult],
        harness: Harness | None = None,
    ) -> None:
        if self._path is None:
            return
        rewards = [ep.summary.total_reward for ep in episodes]
        per_task = {
            ep.task_id: {
                "reward": ep.summary.total_reward,
                "signals": {
                    k: {"value": s.value, "confidence": s.confidence}
                    for k, s in ep.summary.signals.items()
                } if ep.summary.signals else {},
                "error": ep.metadata.get("error") if ep.metadata else None,
            }
            for ep in episodes
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
                name: {"status": r.status, "metrics": r.metrics}
                for name, r in fb_results.items()
            },
        }
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


@dataclass
class AgentState:
    """Bundle of the three mutable learning layers."""

    harness: Harness = field(default_factory=Harness)
    router: Router = field(default_factory=Router)
    weights: Weights = field(default_factory=Weights)
    inference_url: str | None = None  # vLLM endpoint for Harbor agents
    tried_paradigms: list[str] = field(default_factory=list)  # paradigm contents tried
    _prev_playbook_generation: int = 0  # tracks generation for flush logic

    def state_id(self) -> StateID:
        return StateID.from_layers(self.harness, self.router, self.weights)

    def get_layers(
        self, active: list[str] | None = None,
    ) -> list[tuple[str, Any]]:
        """Return (name, layer) pairs, filtered by *active* if given."""
        all_layers = [(name, getattr(self, name)) for name in LAYER_NAMES]
        if active is None:
            return all_layers
        return [(n, l) for n, l in all_layers if n in active]


class AdapterLike(Protocol):
    def run_episode(self, task: Any, agent_state: AgentState) -> Episode: ...


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

    Returns
    -------
    tuple[AgentState, StateID]
        The final agent state and its content-addressed state ID.
    """
    state_id = agent_state.state_id()
    layers = agent_state.get_layers(active_layers)
    exp_log = ExperimentLog(output_dir)
    evo_log = EvolutionLog(output_dir)
    log.info("Starting learning loop — initial state: %s", state_id.combined_hash[:12])

    for iteration in range(n_iterations):
        log.info("Iteration %d/%d", iteration + 1, n_iterations)

        # 1. Collect episodes
        if not tasks or n_episodes <= 0:
            episodes: list[Episode] = []
        else:
            if n_episodes <= len(tasks):
                selected_tasks = random.sample(tasks, n_episodes)
            else:
                selected_tasks = random.choices(tasks, k=n_episodes)

            if hasattr(adapter, "run_batch") and callable(
                getattr(adapter, "run_batch", None)
            ):
                episodes = adapter.run_batch(agent_state, selected_tasks)
            else:
                episodes = []
                for task in selected_tasks:
                    ep = adapter.run_episode(task, agent_state)
                    episodes.append(ep)

        avg_reward = (
            sum(ep.summary.total_reward for ep in episodes) / len(episodes)
            if episodes
            else 0.0
        )
        log.info("  Collected %d episodes, avg reward: %.4f", len(episodes), avg_reward)

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
            if name == "harness" and intensity is not None and not intensity.should_reflect(iteration):
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
            (name, layer) for name, layer in layers
            if fb_results.get(name, FBResult(status="error")).status
            not in ("error", "skipped")
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
                    name, result.status, result.updates_applied,
                )
                if result.status == "error":
                    optim_failed = True
                    log.error(
                        "  optim %s returned error — triggering rollback", name,
                    )
                    break
            except Exception:
                log.exception(
                    "optim_step failed for %s — triggering rollback", name,
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
                                "  rollback returned %s for %s", lr.status, name,
                            )
                    except Exception:
                        log.exception("  rollback failed for %s", name)

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
                    prev_gen, current_gen, stale,
                )
            agent_state._prev_playbook_generation = current_gen

        # 5. Log iteration results
        harness_ref = agent_state.harness if isinstance(agent_state.harness, Harness) else None
        exp_log.log_iteration(iteration, episodes, fb_results, harness_ref)

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
            evo_log.append(EvolutionEntry(
                iteration=iteration,
                state_hash_before=prev_hash,
                state_hash_after=state_id.combined_hash,
                actions=actions,
                reward_before=avg_reward if iteration == 0 else (
                    intensity._rewards[-2] if intensity and len(intensity._rewards) >= 2 else avg_reward
                ),
                reward_after=avg_reward,
                backend=(
                    agent_state.harness.evolver.name()
                    if isinstance(agent_state.harness, Harness) and agent_state.harness.evolver
                    else "none"
                ),
            ))

    log.info("Loop complete — final state: %s", state_id.combined_hash[:12])
    return agent_state, state_id
