"""Episode-collection helper extracted from ``learning_loop``.

The runner owns the per-iteration task-sampling + adapter-dispatch protocol.
It is deliberately a thin wrapper: the loop still owns iteration control flow,
and the runner just encapsulates the "get me ``n_episodes`` rollouts" step so
``learning_loop`` stays readable end-to-end.
"""

from __future__ import annotations

import random
from typing import Any

from clawloop.core.episode import Episode


class EpisodeCollectorRunner:
    """Samples tasks and dispatches to the adapter to collect a batch of episodes.

    Parameters
    ----------
    adapter:
        Environment adapter. Must expose ``run_episode(task, agent_state)`` and
        may optionally expose ``run_batch(agent_state, tasks)`` for a faster
        path. Typed as ``Any`` to avoid a circular import with the
        ``AdapterLike`` Protocol, which lives in ``loop.py``.
    """

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def collect(
        self,
        agent_state: Any,
        tasks: list[Any],
        n_episodes: int,
    ) -> list[Episode]:
        """Return ``n_episodes`` rollouts against ``tasks``.

        Behavior mirrors the block previously inlined in ``learning_loop``:

        * Empty ``tasks`` or ``n_episodes <= 0`` returns ``[]`` without
          touching the adapter.
        * Samples without replacement when ``n_episodes <= len(tasks)``,
          with replacement otherwise.
        * Prefers ``adapter.run_batch(agent_state, selected_tasks)`` when
          available; otherwise falls back to a per-task ``run_episode`` loop.
        """
        if not tasks or n_episodes <= 0:
            return []

        if n_episodes <= len(tasks):
            selected_tasks = random.sample(tasks, n_episodes)
        else:
            selected_tasks = random.choices(tasks, k=n_episodes)

        run_batch = getattr(self._adapter, "run_batch", None)
        if callable(run_batch):
            return run_batch(agent_state, selected_tasks)

        episodes: list[Episode] = []
        for task in selected_tasks:
            episodes.append(self._adapter.run_episode(task, agent_state))
        return episodes
