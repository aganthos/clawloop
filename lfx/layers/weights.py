"""Weights layer — model fine-tuning via SkyRL GRPO.

This layer owns the model reference, LoRA adapters, and GRPO training config.
It delegates the actual training to SkyRL's ``RayPPOTrainer`` by converting
LfX Episodes into ``GeneratorOutput`` via the SkyRL exporter.

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

from dataclasses import dataclass, field
from typing import Any


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
