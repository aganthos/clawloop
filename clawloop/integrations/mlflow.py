"""MLflow tracking sink for ClawLoop episode and iteration metrics.

Usage::

    from clawloop.integrations.mlflow import MlflowSink

    sink = MlflowSink(experiment_name="clawloop-experiments")
    learning_loop(..., after_iteration=sink.after_iteration)
    sink.finish()

Requires the ``mlflow`` optional extra: ``uv sync --extra mlflow``.
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


class MlflowSink:
    """Log ClawLoop metrics and artifacts to MLflow.

    Parameters
    ----------
    experiment_name:
        Optional MLflow experiment name.
    run_name:
        Optional MLflow run name.
    tracking_uri:
        Optional tracking URI passed to ``mlflow.set_tracking_uri``.
    tags:
        Optional tags attached to the run.
    log_episodes:
        If ``True`` (default), log a JSON artifact with per-episode summaries.
    """

    def __init__(
        self,
        *,
        experiment_name: str | None = None,
        run_name: str | None = None,
        tracking_uri: str | None = None,
        tags: dict[str, Any] | None = None,
        log_episodes: bool = True,
    ) -> None:
        try:
            import mlflow  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "mlflow is required for MlflowSink. Install with: pip install 'clawloop[mlflow]'"
            ) from exc

        self._mlflow = mlflow
        self._log_episodes = log_episodes
        self._step = 0
        self._run = None

        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            mlflow.set_experiment(experiment_name)

        self._run = mlflow.start_run(run_name=run_name, tags=tags)

    def log_episodes(
        self,
        episodes: list[Episode],
        *,
        iteration: int | None = None,
    ) -> None:
        """Log aggregate reward metrics and optional episode summaries."""
        if not episodes:
            return

        step = iteration if iteration is not None else self._step
        self._step = step + 1

        try:
            self._log_reward_scalars(episodes, step)
            if self._log_episodes:
                self._log_episode_summaries(episodes, step)
        except Exception:
            _log.warning("MlflowSink.log_episodes failed", exc_info=True)

    def log_iteration(
        self,
        iteration: int,
        episodes: list[Episode],
        fb_results: dict[str, FBResult] | None = None,
        *,
        harness: Harness | None = None,
        state_id: StateID | None = None,
    ) -> None:
        """Log a full iteration: rewards, playbook growth, state hashes, and feedback."""
        self._step = iteration + 1

        try:
            self._log_reward_scalars(episodes, iteration)

            if harness is not None:
                self._log_playbook(harness, iteration)

            if state_id is not None:
                self._log_state_hashes(state_id, iteration)

            if fb_results is not None:
                self._log_fb_results(fb_results, iteration)

            if self._log_episodes and episodes:
                self._log_episode_summaries(episodes, iteration)
        except Exception:
            _log.warning("MlflowSink.log_iteration failed", exc_info=True)

    def after_iteration(
        self,
        iteration: int,
        agent_state: Any,
        episodes: list[Episode],
    ) -> None:
        """Drop-in ``after_iteration`` callback for :func:`learning_loop`."""
        from clawloop.learning_layers.harness import Harness

        harness = agent_state.harness if isinstance(agent_state.harness, Harness) else None
        state_id = agent_state.state_id()

        self.log_iteration(iteration, episodes, harness=harness, state_id=state_id)

    def finish(self) -> None:
        """Finalize the MLflow run."""
        try:
            self._mlflow.end_run()
        except Exception:
            _log.warning("MlflowSink.finish failed", exc_info=True)

    def _log_reward_scalars(self, episodes: list[Episode], step: int) -> None:
        if not episodes:
            return

        rewards = [ep.summary.effective_reward() for ep in episodes]
        normalized = [ep.summary.normalized_reward() for ep in episodes]

        metrics: dict[str, float] = {
            "reward.mean": sum(rewards) / len(rewards),
            "reward.min": min(rewards),
            "reward.max": max(rewards),
            "reward.mean_normalized": sum(normalized) / len(normalized),
            "episodes.count": float(len(episodes)),
        }

        signal_sums: dict[str, list[float]] = {}
        for ep in episodes:
            for name, sig in ep.summary.signals.items():
                signal_sums.setdefault(name, []).append(sig.value)
        for name, values in signal_sums.items():
            metrics[f"reward.{name}.mean"] = sum(values) / len(values)

        self._mlflow.log_metrics(metrics, step=step)

    def _log_playbook(self, harness: Harness, step: int) -> None:
        entries = harness.playbook.entries
        metrics: dict[str, float] = {
            "playbook.size": float(len(entries)),
        }

        if entries:
            scores = [e.effective_score() for e in entries]
            metrics["playbook.mean_score"] = sum(scores) / len(scores)
            metrics["playbook.max_score"] = max(scores)
            metrics["playbook.helpful_total"] = float(sum(e.helpful for e in entries))
            metrics["playbook.harmful_total"] = float(sum(e.harmful for e in entries))
            self._mlflow.log_dict(
                [self._playbook_entry_dict(entry) for entry in entries],
                artifact_file=f"iterations/{step}/playbook_entries.json",
            )

        self._mlflow.log_metrics(metrics, step=step)

    def _log_state_hashes(self, state_id: StateID, step: int) -> None:
        self._mlflow.log_dict(
            {
                "combined": state_id.combined_hash,
                "harness": state_id.harness_hash,
                "router": state_id.router_hash,
                "weights": state_id.weights_hash,
            },
            artifact_file=f"iterations/{step}/state_hashes.json",
        )

    def _log_fb_results(self, fb_results: dict[str, FBResult], step: int) -> None:
        metrics: dict[str, float] = {}
        artifacts: dict[str, Any] = {}
        for name, result in fb_results.items():
            artifacts[name] = {
                "status": result.status,
                "metrics": result.metrics or {},
            }
            for key, value in (result.metrics or {}).items():
                if isinstance(value, (int, float, bool)):
                    metrics[f"fb.{name}.{key}"] = float(value)

        if metrics:
            self._mlflow.log_metrics(metrics, step=step)
        if artifacts:
            self._mlflow.log_dict(artifacts, artifact_file=f"iterations/{step}/fb_results.json")

    def _log_episode_summaries(self, episodes: list[Episode], step: int) -> None:
        self._mlflow.log_dict(
            [self._episode_dict(ep) for ep in episodes],
            artifact_file=f"iterations/{step}/episodes.json",
        )

    @staticmethod
    def _episode_dict(ep: Episode) -> dict[str, Any]:
        return {
            "episode_id": ep.id,
            "task_id": ep.task_id,
            "bench": ep.bench,
            "state_id": ep.state_id,
            "model": ep.model,
            "effective_reward": ep.summary.effective_reward(),
            "normalized_reward": ep.summary.normalized_reward(),
            "n_steps": ep.n_steps(),
            "n_messages": len(ep.messages),
            "filtered": ep.summary.filtered,
        }

    @staticmethod
    def _playbook_entry_dict(entry: Any) -> dict[str, Any]:
        return {
            "score": entry.effective_score(),
            "helpful": entry.helpful,
            "harmful": entry.harmful,
        }
