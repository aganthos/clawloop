"""Main learning loop: collect episodes -> propose updates -> apply -> repeat.

The loop is benchmark-agnostic.  It delegates episode collection to an
``EnvAdapter`` and update proposals to per-layer proposers.  Gating (regression
checks) is intentionally *not* part of the inner loop — see ``gate.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from lfx.core.episode import Episode, LearningUpdate
from lfx.core.state import StateID
from lfx.layers.harness import Harness
from lfx.layers.router import Router
from lfx.layers.weights import Weights

log = logging.getLogger(__name__)


# -- Agent state bundle --


@dataclass
class AgentState:
    """Bundle of the three mutable learning layers."""

    harness: Harness = field(default_factory=Harness)
    router: Router = field(default_factory=Router)
    weights: Weights = field(default_factory=Weights)

    def state_id(self) -> StateID:
        return StateID.from_layers(self.harness, self.router, self.weights)


# -- Proposer protocol --


class Proposer(Protocol):
    """Callable that inspects episodes and proposes layer mutations."""

    def propose(self, episodes: list[Episode], layer: Any) -> list[LearningUpdate]: ...


# -- Adapter protocol (mirrors adapters/base.py ABC) --


class AdapterLike(Protocol):
    def run_episode(self, task: Any, agent_state: AgentState) -> Episode: ...


# -- Learning loop --


def learning_loop(
    adapter: AdapterLike,
    agent_state: AgentState,
    tasks: list[Any],
    n_episodes: int,
    n_iterations: int,
    *,
    harness_proposer: Proposer | None = None,
    router_proposer: Proposer | None = None,
    weights_proposer: Proposer | None = None,
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
    harness_proposer, router_proposer, weights_proposer:
        Optional proposers for each layer.  ``None`` means that layer is
        not updated.

    Returns
    -------
    tuple[AgentState, StateID]
        The final agent state and its content-addressed state ID.
    """
    state_id = agent_state.state_id()
    log.info("Starting learning loop — initial state: %s", state_id.combined_hash[:12])

    for iteration in range(n_iterations):
        log.info("Iteration %d/%d", iteration + 1, n_iterations)

        # 1. Collect episodes
        selected_tasks = tasks[:n_episodes]
        episodes: list[Episode] = []
        for task in selected_tasks:
            ep = adapter.run_episode(task, agent_state)
            episodes.append(ep)

        avg_reward = (
            sum(ep.summary.total_reward for ep in episodes) / len(episodes)
            if episodes
            else 0.0
        )
        log.info("  Collected %d episodes, avg reward: %.4f", len(episodes), avg_reward)

        # 2. Propose updates
        updates: list[LearningUpdate] = []
        if harness_proposer is not None:
            updates.extend(harness_proposer.propose(episodes, agent_state.harness))
        if router_proposer is not None:
            updates.extend(router_proposer.propose(episodes, agent_state.router))
        if weights_proposer is not None:
            updates.extend(weights_proposer.propose(episodes, agent_state.weights))

        # 3. Apply accepted updates (no gating during iteration)
        for update in updates:
            if update.decision == "accept":
                agent_state = _apply_update(agent_state, update)
                log.info(
                    "  Applied %s update: %s -> %s",
                    update.layer_type,
                    update.state_id_before[:12],
                    update.state_id_after[:12],
                )

        state_id = agent_state.state_id()

    log.info("Loop complete — final state: %s", state_id.combined_hash[:12])
    return agent_state, state_id


def _apply_update(agent_state: AgentState, update: LearningUpdate) -> AgentState:
    """Apply a single accepted update, returning a new AgentState.

    The actual mutation logic lives in the proposers which produce fully-formed
    replacement layers via the ``proposal`` dict.
    """
    proposal = update.proposal
    harness = agent_state.harness
    router = agent_state.router
    weights = agent_state.weights
    if update.layer_type == "harness" and "harness" in proposal:
        harness = proposal["harness"]
    elif update.layer_type == "router" and "router" in proposal:
        router = proposal["router"]
    elif update.layer_type == "weights" and "weights" in proposal:
        weights = proposal["weights"]
    return AgentState(harness=harness, router=router, weights=weights)
