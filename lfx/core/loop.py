"""Learning loop: collect episodes -> forward_backward -> optim_step -> repeat.

The loop is benchmark-agnostic. It delegates episode collection to an
``AdapterLike`` and learning to the Layer protocol on each layer.
Gating (regression checks) is intentionally *not* part of the inner
loop -- see ``gate.py``.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Protocol

from lfx.core.episode import Episode
from lfx.core.intensity import AdaptiveIntensity
from lfx.core.paradigm import ParadigmBreakthrough
from lfx.core.state import StateID
from lfx.core.types import Datum, FBResult, Future, OptimResult
from lfx.layers.harness import Harness
from lfx.layers.router import Router
from lfx.layers.weights import Weights

log = logging.getLogger(__name__)


LAYER_NAMES = ("harness", "router", "weights")


@dataclass
class AgentState:
    """Bundle of the three mutable learning layers."""

    harness: Harness = field(default_factory=Harness)
    router: Router = field(default_factory=Router)
    weights: Weights = field(default_factory=Weights)

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
    paradigm: ParadigmBreakthrough | None = None,
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
        Reflector fires (saves LLM calls).
    paradigm:
        Optional paradigm breakthrough generator that fires when
        learning stagnates.

    Returns
    -------
    tuple[AgentState, StateID]
        The final agent state and its content-addressed state ID.
    """
    state_id = agent_state.state_id()
    layers = agent_state.get_layers(active_layers)
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

        # 2. Build Datum
        datum = Datum(episodes=episodes)

        # 3. Phase 1: forward_backward (all active layers)
        fb_results: dict[str, FBResult] = {}
        for name, layer in layers:
            # Skip harness reflection when intensity says not to
            if name == "harness" and intensity is not None and not intensity.should_reflect(iteration):
                log.info("  skipping harness fb (adaptive intensity)")
                fb_results[name] = FBResult(status="skipped")
                continue
            try:
                fut = layer.forward_backward(datum)
                fb_results[name] = fut.result()
            except Exception:
                log.exception("forward_backward failed for %s", name)
                fb_results[name] = FBResult(status="error")
                # Clear any partially-accumulated pending state so it doesn't
                # leak into a future optim_step.
                layer.clear_pending_state()

        for name, result in fb_results.items():
            log.info("  fb %s: %s %s", name, result.status, result.metrics)

        # 4. Phase 2: optim_step (only layers whose fb succeeded)
        for name, layer in layers:
            if fb_results.get(name, FBResult(status="error")).status in ("error", "skipped"):
                log.warning("  skipping optim_step for %s (fb failed or skipped)", name)
                continue
            try:
                result = layer.optim_step().result()
                log.info(
                    "  optim %s: %s, %d updates",
                    name, result.status, result.updates_applied,
                )
            except Exception:
                log.exception("optim_step failed for %s", name)

        # Paradigm breakthrough on stagnation
        if paradigm is not None and intensity is not None and intensity.is_stagnating():
            log.info("  stagnation detected — triggering paradigm breakthrough")
            try:
                if isinstance(agent_state.harness, Harness):
                    insights = paradigm.generate(
                        playbook=agent_state.harness.playbook,
                        reward_history=intensity._rewards,
                        tried_paradigms=[],  # TODO: track tried paradigms
                    )
                    if insights:
                        agent_state.harness._pending.insights.extend(insights)
                        agent_state.harness.optim_step()
                        log.info("  paradigm: applied %d insights", len(insights))
            except Exception:
                log.exception("paradigm breakthrough failed")

        # 5. Recompute state identity
        state_id = agent_state.state_id()

    log.info("Loop complete — final state: %s", state_id.combined_hash[:12])
    return agent_state, state_id
