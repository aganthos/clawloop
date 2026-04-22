"""Weights & Biases sink — log reward curves, playbook growth, and layer state hashes.

Usage::

    from clawloop.integrations.wandb import WandbSink

    sink = WandbSink(run_id="my-run", project="clawloop-experiments")
    # As an after_iteration callback in the learning loop:
    learning_loop(..., after_iteration=sink.after_iteration)
    # Or log a batch of episodes directly:
    sink.log_episodes(episodes, iteration=0)
    sink.finish()

Requires the ``wandb`` optional extra: ``uv sync --extra wandb``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawloop.core.episode import Episode
    from clawloop.core.state import StateID
    from clawloop.core.types import FBResult
    from clawloop.learning_layers.harness import Harness

_log = logging.getLogger(__name__)


class WandbSink:
    """Log ClawLoop metrics to Weights & Biases.

    Parameters
    ----------
    run_id:
        W&B run ID.  When resuming, pass the same ID to append.
    project:
        W&B project name.
    entity:
        W&B entity (team or user).  ``None`` uses the default entity.
    config:
        Dict merged into ``wandb.config`` at init time.
    log_episodes:
        If ``True`` (default), log a ``wandb.Table`` of per-episode detail
        each iteration.  Set to ``False`` to reduce upload volume.
    """

    def __init__(
        self,
        *,
        run_id: str | None = None,
        project: str | None = None,
        entity: str | None = None,
        config: dict[str, Any] | None = None,
        log_episodes: bool = True,
    ) -> None:
        try:
            import wandb  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "wandb is required for WandbSink. " "Install with: pip install 'clawloop[wandb]'"
            ) from exc

        self._wandb = wandb
        self._log_episodes = log_episodes
        self._step = 0

        init_kwargs: dict[str, Any] = {}
        if run_id is not None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = "allow"
        if project is not None:
            init_kwargs["project"] = project
        if entity is not None:
            init_kwargs["entity"] = entity
        if config is not None:
            init_kwargs["config"] = config

        self._run = wandb.init(**init_kwargs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_episodes(
        self,
        episodes: list[Episode],
        *,
        iteration: int | None = None,
    ) -> None:
        """Log per-episode reward curves and summary metrics for one batch.

        Parameters
        ----------
        episodes:
            Batch of completed episodes.
        iteration:
            Explicit iteration counter.  When ``None``, an internal
            auto-incrementing counter is used.
        """
        if not episodes:
            return

        step = iteration if iteration is not None else self._step
        self._step = step + 1

        try:
            self._log_reward_scalars(episodes, step)
            if self._log_episodes:
                self._log_episode_table(episodes, step)
        except Exception:
            _log.warning("WandbSink.log_episodes failed", exc_info=True)

    def log_iteration(
        self,
        iteration: int,
        episodes: list[Episode],
        fb_results: dict[str, FBResult] | None = None,
        *,
        harness: Harness | None = None,
        state_id: StateID | None = None,
    ) -> None:
        """Log a full iteration: rewards + playbook growth + layer hashes.

        Designed to be called from the learning loop alongside
        :class:`~clawloop.core.loop.ExperimentLog`.
        """
        try:
            self._log_reward_scalars(episodes, iteration)

            if harness is not None:
                self._log_playbook(harness, iteration)

            if state_id is not None:
                self._log_state_hashes(state_id, iteration)

            if fb_results is not None:
                self._log_fb_results(fb_results, iteration)

            if self._log_episodes and episodes:
                self._log_episode_table(episodes, iteration)
        except Exception:
            _log.warning("WandbSink.log_iteration failed", exc_info=True)

    def after_iteration(
        self,
        iteration: int,
        agent_state: Any,
        episodes: list[Episode],
    ) -> None:
        """Drop-in ``after_iteration`` callback for :func:`learning_loop`.

        Extracts harness and state_id from *agent_state* automatically.
        """
        from clawloop.learning_layers.harness import Harness

        harness = agent_state.harness if isinstance(agent_state.harness, Harness) else None
        state_id = agent_state.state_id()

        self.log_iteration(
            iteration,
            episodes,
            harness=harness,
            state_id=state_id,
        )

    def finish(self) -> None:
        """Finalize the W&B run."""
        try:
            self._run.finish()
        except Exception:
            _log.warning("WandbSink.finish failed", exc_info=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log_reward_scalars(self, episodes: list[Episode], step: int) -> None:
        """Log aggregate reward metrics as scalars."""
        if not episodes:
            return

        rewards = [ep.summary.effective_reward() for ep in episodes]
        normalized = [ep.summary.normalized_reward() for ep in episodes]

        metrics: dict[str, Any] = {
            "reward/mean": sum(rewards) / len(rewards),
            "reward/min": min(rewards),
            "reward/max": max(rewards),
            "reward/mean_normalized": sum(normalized) / len(normalized),
            "episodes/count": len(episodes),
        }

        # Per-signal averages across the batch
        signal_sums: dict[str, list[float]] = {}
        for ep in episodes:
            for name, sig in ep.summary.signals.items():
                signal_sums.setdefault(name, []).append(sig.value)
        for name, values in signal_sums.items():
            metrics[f"reward/{name}_mean"] = sum(values) / len(values)

        self._run.log(metrics, step=step)

    def _log_playbook(self, harness: Harness, step: int) -> None:
        """Log playbook growth metrics."""
        entries = harness.playbook.entries
        metrics: dict[str, Any] = {
            "playbook/size": len(entries),
        }

        if entries:
            scores = [e.effective_score() for e in entries]
            metrics["playbook/mean_score"] = sum(scores) / len(scores)
            metrics["playbook/max_score"] = max(scores)
            metrics["playbook/helpful_total"] = sum(e.helpful for e in entries)
            metrics["playbook/harmful_total"] = sum(e.harmful for e in entries)

        self._run.log(metrics, step=step)

    def _log_state_hashes(self, state_id: StateID, step: int) -> None:
        """Log layer state hashes as string summaries."""
        self._run.log(
            {
                "state/combined": state_id.combined_hash[:12],
                "state/harness": state_id.harness_hash[:12],
                "state/router": state_id.router_hash[:12],
                "state/weights": state_id.weights_hash[:12],
            },
            step=step,
        )

    def _log_fb_results(self, fb_results: dict[str, FBResult], step: int) -> None:
        """Log per-layer forward_backward result metrics."""
        metrics: dict[str, Any] = {}
        for name, result in fb_results.items():
            metrics[f"fb/{name}/status"] = result.status
            for mk, mv in (result.metrics or {}).items():
                if isinstance(mv, (int, float, bool)):
                    metrics[f"fb/{name}/{mk}"] = mv
        if metrics:
            self._run.log(metrics, step=step)

    def _log_episode_table(self, episodes: list[Episode], step: int) -> None:
        """Log a W&B Table with per-episode detail."""
        wandb = self._wandb

        columns = [
            "episode_id",
            "task_id",
            "bench",
            "state_id",
            "model",
            "effective_reward",
            "normalized_reward",
            "n_steps",
            "n_messages",
            "filtered",
        ]
        data = []
        for ep in episodes:
            data.append(
                [
                    ep.id,
                    ep.task_id,
                    ep.bench,
                    ep.state_id[:12],
                    ep.model or "",
                    ep.summary.effective_reward(),
                    ep.summary.normalized_reward(),
                    ep.n_steps(),
                    len(ep.messages),
                    ep.summary.filtered,
                ]
            )

        table = wandb.Table(columns=columns, data=data)
        self._run.log({"episodes/table": table}, step=step)
