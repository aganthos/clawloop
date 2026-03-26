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

from clawloop.core.episode import Episode
from clawloop.exporters.base import TraceExporter


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

    def export(self, episodes: list[Episode], repetition_offset: int = 0) -> dict[str, Any]:
        """Export a batch of episodes as a single GeneratorOutput.

        Parameters
        ----------
        repetition_offset:
            Starting repetition index.  When exporting multiple batches for
            the same task pool, increment this to avoid trajectory ID collisions.
        """
        prompt_token_ids: list[list[int]] = []
        response_ids: list[list[int]] = []
        rewards: list[float] = []
        loss_masks: list[list[int]] = []
        trajectory_ids: list[TrajectoryID] = []
        is_last_step: list[bool] = []
        rollout_logprobs: list[list[float] | None] = []

        rep_idx = 0
        for ep in episodes:
            if not ep.steps:
                continue
            result = self._episode_to_transitions(ep, repetition_id=repetition_offset + rep_idx)
            rep_idx += 1
            prompt_token_ids.extend(result["prompt_token_ids"])
            response_ids.extend(result["response_ids"])
            rewards.extend(result["rewards"])
            loss_masks.extend(result["loss_masks"])
            trajectory_ids.extend(result["trajectory_ids"])
            is_last_step.extend(result["is_last_step"])
            rollout_logprobs.extend(result["rollout_logprobs"])

        return {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": None,
            "rollout_metrics": None,
            "rollout_logprobs": rollout_logprobs if any(lp is not None for lp in rollout_logprobs) else None,
            "trajectory_ids": trajectory_ids,
            "is_last_step": is_last_step,
        }

    def export_one(self, episode: Episode) -> dict[str, Any]:
        return self.export([episode])

    def _episode_to_transitions(self, episode: Episode, repetition_id: int = 0) -> dict[str, Any]:
        """Convert a single episode into per-transition SkyRL arrays.

        For each step *t*, ``step_boundaries[t]`` marks the start of the user
        turn.  The **prompt** includes everything up to and including the user
        message(s) — i.e. the context the model sees at inference time.  The
        **response** is only the assistant (and tool-result) messages that
        follow, which is what the policy actually generates.

        Only the terminal step carries the episode reward; all others get 0.0.
        """
        prompt_token_ids: list[list[int]] = []
        response_ids: list[list[int]] = []
        rewards: list[float] = []
        loss_masks: list[list[int]] = []
        trajectory_ids: list[TrajectoryID] = []
        is_last_step_flags: list[bool] = []
        rollout_logprobs: list[list[float] | None] = []

        n_steps = len(episode.steps)

        for t in range(n_steps):
            # Determine the end of this step's messages
            if t + 1 < len(episode.step_boundaries):
                step_end = episode.step_boundaries[t + 1]
            else:
                step_end = len(episode.messages)

            # Find the first assistant message in this step — that's where
            # the response begins.  Everything before it is prompt context.
            step_start = episode.step_boundaries[t]
            resp_start = step_start
            for idx in range(step_start, step_end):
                if episode.messages[idx].role == "assistant":
                    resp_start = idx
                    break

            # Prompt = all messages up to (but not including) the first
            # assistant message in this step.  This includes the system
            # prompt, prior turns, AND the current user message — matching
            # what the model sees at inference time.
            prompt_msgs = [m.to_openai_dict() for m in episode.messages[:resp_start]]

            # Response = assistant (and tool-result) messages only
            resp_msgs = [m.to_openai_dict() for m in episode.messages[resp_start:step_end]]

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
                # For assistant messages with tool_calls, serialize the
                # tool call payload so the model learns to emit them.
                if msg["role"] == "assistant" and msg.get("tool_calls"):
                    tc_parts = []
                    if text:
                        tc_parts.append(text)
                    for tc in msg["tool_calls"]:
                        fn = tc.get("function", {})
                        tc_parts.append(
                            f'{{"name":"{fn.get("name", "")}",'
                            f'"arguments":{fn.get("arguments", "{}")}}}'
                        )
                    text = " ".join(tc_parts)
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                r_ids.extend(tokens)
                if msg["role"] == "assistant":
                    l_mask.extend([1] * len(tokens))
                else:
                    # tool-result tokens are masked out
                    l_mask.extend([0] * len(tokens))

            # Collect logprobs from assistant messages in this step.
            # Only usable when logprobs cover ALL response tokens (len matches
            # response_ids); otherwise set to None to avoid SkyRL assertion.
            step_lps: list[float] = []
            for msg in episode.messages[resp_start:step_end]:
                if msg.role == "assistant" and msg.logprobs:
                    step_lps.extend(lp.logprob for lp in msg.logprobs)
            if step_lps and len(step_lps) == len(r_ids):
                rollout_logprobs.append(step_lps)
            else:
                rollout_logprobs.append(None)

            prompt_token_ids.append(list(p_ids))
            response_ids.append(r_ids)
            loss_masks.append(l_mask)

            # Sparse reward: only terminal step
            is_terminal = episode.steps[t].done
            rewards.append(episode.terminal_reward() if is_terminal else 0.0)
            is_last_step_flags.append(is_terminal)

            trajectory_ids.append(
                TrajectoryID(instance_id=episode.task_id, repetition_id=repetition_id)
            )

        return {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": response_ids,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "trajectory_ids": trajectory_ids,
            "is_last_step": is_last_step_flags,
            "rollout_logprobs": rollout_logprobs,
        }
