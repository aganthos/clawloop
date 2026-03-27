"""Weights layer — model fine-tuning via SkyRL GRPO.

This layer owns the model reference, LoRA adapters, and GRPO training config.
It delegates the actual training to SkyRL's ``RayPPOTrainer`` by converting
ClawLoop Episodes into ``GeneratorOutput`` via the SkyRL exporter.

The training loop:
  1. Collect episodes via the learning loop.
  2. Serialize to ``GeneratorOutput`` (prompt_token_ids, response_ids,
     loss_masks, rewards, trajectory_ids).
  3. SkyRL computes GRPO advantages (reward - mean_reward across rollouts
     of the same task) and runs forward-backward.
  4. Updated adapter checkpoints are stored as new ``adapter_refs``.

GRPO is completion-level, NOT step-level.  Multi-turn support works via
prefix-sharing with loss masks (1=assistant tokens, 0=env tokens).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from clawloop.backends.base import ClawLoopBackend

from clawloop.core.types import (
    Datum, FBResult, Future, LoadResult, OptimResult,
    SampleContext, SampleResult, SaveResult,
)


@dataclass
class GRPOConfig:
    """Configuration for SkyRL GRPO training.

    Maps to SkyRL's trainer hyperparameters.
    """

    n_samples_per_prompt: int = 4  # GRPO group size (N completions per prompt)
    learning_rate: float = 1e-5
    kl_coeff: float = 0.05  # KL penalty coefficient (policy vs reference)
    clip_ratio: float = 0.2  # PPO clipping ratio
    epochs_per_batch: int = 1
    max_grad_norm: float = 1.0
    use_advantage_normalization: bool = True
    min_group_variance: float = 1e-6  # GRPO zero-variance filter threshold


@dataclass
class _WeightsPending:
    """Accumulator for GRPO advantages. Drained by optim_step."""
    advantages: list[tuple[str, float]] = field(default_factory=list)
    # (episode_id, advantage)


@dataclass
class Weights:
    """Model fine-tuning state and SkyRL GRPO interface.

    Tracks the base model, active adapters, and training configuration.
    The ``propose`` step (in the learning loop) uses the SkyRL exporter
    to convert episodes -> GeneratorOutput and triggers a GRPO training step.
    """

    model_ref: str = ""  # base model identifier (e.g. "meta-llama/Llama-3-8B")
    adapter_refs: list[str] = field(default_factory=list)  # LoRA checkpoint paths
    grpo_config: GRPOConfig = field(default_factory=GRPOConfig)
    training_history: list[dict[str, Any]] = field(default_factory=list)
    _pending: _WeightsPending = field(default_factory=_WeightsPending)
    _backend: ClawLoopBackend | None = None

    @property
    def active_adapter(self) -> str | None:
        """The most recently trained adapter, if any."""
        return self.adapter_refs[-1] if self.adapter_refs else None

    def record_training_step(
        self,
        adapter_path: str,
        metrics: dict[str, float],
    ) -> None:
        """Record a completed GRPO training step."""
        self.adapter_refs.append(adapter_path)
        self.training_history.append({
            "adapter_path": adapter_path,
            "metrics": metrics,
        })

    def to_dict(self) -> dict[str, Any]:
        if self._backend:
            return self._backend.to_dict()
        return {
            "model_ref": self.model_ref,
            "adapter_refs": self.adapter_refs,
            "grpo_config": {
                "n_samples_per_prompt": self.grpo_config.n_samples_per_prompt,
                "learning_rate": self.grpo_config.learning_rate,
                "kl_coeff": self.grpo_config.kl_coeff,
                "clip_ratio": self.grpo_config.clip_ratio,
                "epochs_per_batch": self.grpo_config.epochs_per_batch,
                "max_grad_norm": self.grpo_config.max_grad_norm,
                "use_advantage_normalization": self.grpo_config.use_advantage_normalization,
                "min_group_variance": self.grpo_config.min_group_variance,
            },
        }

    # -- Layer protocol methods --

    def pending_advantage_count(self) -> int:
        """Return the number of pending advantages (for logging/flush)."""
        return len(self._pending.advantages)

    def clear_pending_state(self) -> None:
        """Reset the internal pending accumulator."""
        if self._backend:
            self._backend.clear_pending_state()
        self._pending = _WeightsPending()

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Compute GRPO advantages without mutating observable state."""
        if self._backend:
            return self._backend.forward_backward(data)
        return self._stub_forward_backward(data)

    def _stub_forward_backward(self, data: Datum) -> Future[FBResult]:
        """Compute GRPO advantages without mutating observable state (stub)."""
        # Group episodes by task_id
        by_task: dict[str, list[Any]] = defaultdict(list)
        for ep in data.episodes:
            by_task[ep.task_id].append(ep)

        advantages: list[tuple[str, float]] = []
        for task_id, episodes in by_task.items():
            mean_reward = sum(ep.summary.total_reward for ep in episodes) / len(episodes)
            for ep in episodes:
                advantage = ep.summary.total_reward - mean_reward
                advantages.append((ep.id, advantage))

        self._pending.advantages.extend(advantages)
        return Future.immediate(FBResult(status="ok", metrics={"n_advantages": len(advantages)}))

    def optim_step(self) -> Future[OptimResult]:
        """Record a deferred training step (actual SkyRL training deferred)."""
        if self._backend:
            return self._backend.optim_step()
        return self._stub_optim_step()

    def _stub_optim_step(self) -> Future[OptimResult]:
        """Record a deferred training step (stub)."""
        n = len(self._pending.advantages)
        if n == 0:
            return Future.immediate(OptimResult(status="skipped", updates_applied=0))

        # Snapshot-rollback: snapshot training_history before applying
        snapshot = list(self.training_history)
        try:
            self.training_history.append({
                "status": "deferred",
                "advantages_computed": n,
            })
            # Drain pending on success
            self._pending.advantages.clear()
        except Exception:
            # Rollback on failure
            self.training_history = snapshot
            raise

        return Future.immediate(OptimResult(
            status="skipped",
            updates_applied=0,
            metrics={"advantages_computed": n},
        ))

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Return the current model reference."""
        if self._backend:
            return self._backend.sample(ctx)
        return Future.immediate(SampleResult(
            output=self.model_ref,
            metadata={"active_adapter": self.active_adapter},
        ))

    def save_state(self, name: str) -> Future[SaveResult]:
        """Save current state."""
        if self._backend:
            return self._backend.save_state(name)
        return Future.immediate(SaveResult(name=name, status="ok"))

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        """Restore state from a dict. Clears training_history and _pending."""
        if self._backend:
            return self._backend.load_state(state)
        self.model_ref = state.get("model_ref", "")
        self.adapter_refs = list(state.get("adapter_refs", []))
        grpo_dict = state.get("grpo_config", {})
        if grpo_dict:
            self.grpo_config = GRPOConfig(**grpo_dict)
        else:
            self.grpo_config = GRPOConfig()
        self.training_history = []
        self._pending = _WeightsPending()
        return Future.immediate(LoadResult(status="ok"))
