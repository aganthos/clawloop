"""SkyRL exporter — Episode -> GeneratorOutput serialization.

This is the Weights layer training interface.  It converts LfX Episodes into the
``GeneratorOutput`` TypedDict consumed by SkyRL's GRPO trainer.

The conversion requires:
  1. Tokenizing messages using the target model's tokenizer.
  2. Reconstructing prompt/response boundaries from ``step_boundaries``.
  3. Building ``loss_masks`` (1 for assistant tokens, 0 for everything else).
  4. Sparse rewards — only the terminal step carries the actual reward.
  5. ``trajectory_ids`` for GRPO advantage estimation across rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from lfx.core.episode import Episode
from lfx.exporters.base import TraceExporter


class Tokenizer(Protocol):
    """Minimal tokenizer interface (compatible with HuggingFace tokenizers)."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...
    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
    ) -> list[int] | str: ...


@dataclass
class TrajectoryID:
    """Mirrors SkyRL's TrajectoryID for GRPO grouping."""

    instance_id: str
    repetition_id: int

    def to_string(self) -> str:
        return f"{self.instance_id}_{self.repetition_id}"


@dataclass
class SkyRLExporter(TraceExporter):
    """Converts Episodes to SkyRL ``GeneratorOutput`` dicts.

    Parameters
    ----------
    tokenizer:
        A tokenizer that implements ``encode()`` and ``apply_chat_template()``.
    """

    tokenizer: Any = None  # set at init; typed loosely to avoid HF dependency

    def export(self, episodes: list[Episode]) -> dict[str, Any]:
        """Export a batch of episodes as a single GeneratorOutput."""
        prompt_token_ids: list[list[int]] = []
        response_ids: list[list[int]] = []
        rewards: list[float] = []
        loss_masks: list[list[int]] = []
        trajectory_ids: list[TrajectoryID] = []
        is_last_step: list[bool] = []

        for ep in episodes:
            result = self._episode_to_transitions(ep)
            prompt_token_ids.extend(result["prompt_token_ids"])
            response_ids.extend(result["response_ids"])
            rewards.extend(result["rewards"])
            loss_masks.extend(result["loss_masks"])
            trajectory_ids.extend(result["trajectory_ids"])
            is_last_step.extend(result["is_last_step"])

        return {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": None,
            "rollout_metrics": None,
            "rollout_logprobs": None,
            "trajectory_ids": trajectory_ids,
            "is_last_step": is_last_step,
        }

    def export_one(self, episode: Episode) -> dict[str, Any]:
        return self.export([episode])

    def _episode_to_transitions(self, episode: Episode) -> dict[str, Any]:
        """Convert a single episode into per-transition SkyRL arrays.

        For each step *t*, the *prompt* is ``messages[:step_boundaries[t]]``
        and the *response* is
        ``messages[step_boundaries[t]:step_boundaries[t+1]]`` (or end).

        Only the terminal step carries the episode reward; all others get 0.0.
        """
        prompt_token_ids: list[list[int]] = []
        response_ids: list[list[int]] = []
        rewards: list[float] = []
        loss_masks: list[list[int]] = []
        trajectory_ids: list[TrajectoryID] = []
        is_last_step_flags: list[bool] = []

        n_steps = len(episode.steps)

        for t in range(n_steps):
            # Prompt = all messages before this step
            prompt_end = episode.step_boundaries[t]
            prompt_msgs = [m.to_openai_dict() for m in episode.messages[:prompt_end]]

            # Response = messages in this step
            resp_start = episode.step_boundaries[t]
            if t + 1 < len(episode.step_boundaries):
                resp_end = episode.step_boundaries[t + 1]
            else:
                resp_end = len(episode.messages)
            resp_msgs = [m.to_openai_dict() for m in episode.messages[resp_start:resp_end]]

            # Tokenize prompt
            if prompt_msgs:
                p_ids = self.tokenizer.apply_chat_template(
                    prompt_msgs,
                    tokenize=True,
                    add_generation_prompt=True,
                )
                if isinstance(p_ids, str):
                    p_ids = self.tokenizer.encode(p_ids, add_special_tokens=False)
            else:
                p_ids = []

            # Tokenize response messages and build loss mask
            r_ids: list[int] = []
            l_mask: list[int] = []
            for msg in resp_msgs:
                text = msg.get("content", "") or ""
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                r_ids.extend(tokens)
                if msg["role"] == "assistant":
                    l_mask.extend([1] * len(tokens))
                else:
                    # user / system / tool tokens are masked out
                    l_mask.extend([0] * len(tokens))

            prompt_token_ids.append(list(p_ids))
            response_ids.append(r_ids)
            loss_masks.append(l_mask)

            # Sparse reward: only terminal step
            is_terminal = episode.steps[t].done
            rewards.append(episode.terminal_reward() if is_terminal else 0.0)
            is_last_step_flags.append(is_terminal)

            trajectory_ids.append(
                TrajectoryID(instance_id=episode.task_id, repetition_id=0)
            )

        return {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "trajectory_ids": trajectory_ids,
            "is_last_step": is_last_step_flags,
        }
