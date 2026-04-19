"""Episode -> Tinker ``Datum`` exporter with episode-level GRPO advantages.

Algorithm
---------
1. Group input episodes by ``Episode.task_id``.
2. For each group with N >= 2 and non-zero reward variance, compute
   ``advantage[ep] = reward(ep) - mean(rewards in group)``.  Singleton
   groups and zero-variance groups are dropped (no gradient signal).
3. For each surviving episode, emit one :class:`tinker.Datum` per LLM
   turn.  An LLM turn is a step whose ``StepMeta.info`` carries the
   exact-token alignment payload written by the SamplingClient at
   sampling time:

   - ``prompt_tokens``: ``list[int]`` — exact prompt tokens the
     SamplingClient saw.
   - ``sampled_tokens``: ``list[int]`` — exact tokens it emitted.
   - ``sampling_logprobs``: ``list[float]`` — per-token logprobs,
     aligned 1:1 with ``sampled_tokens``.

   Steps lacking ``prompt_tokens`` are non-LLM (CHANCE / opponent) and
   are skipped.

The exporter does **not** re-tokenize assistant text — chat-template
markers, role headers, and whitespace do not round-trip deterministically
through tokenizers, so we rely on the persisted token IDs as ground truth.

Reward attribute note
---------------------
``EpisodeSummary`` does not expose a ``.reward`` attribute.  The canonical
internal reward space is [-1, 1], reachable via
:meth:`EpisodeSummary.effective_reward`.  We use that here (matches the
project's "canonical [-1, 1] internal" convention).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from tinker import types as tinker_types

from clawloop.core.episode import Episode

__all__ = ["episodes_to_tinker_datums"]


def _episode_return(ep: Episode) -> float:
    """Total return for one episode in canonical [-1, 1] space."""
    return ep.summary.effective_reward()


def episodes_to_tinker_datums(
    episodes: list[Episode],
    *,
    loss_fn: str,  # noqa: ARG001 — accepted for future use (e.g., losses
    # that ignore logprobs); v1 does not branch on it.
) -> list[tinker_types.Datum]:
    """Convert episodes into Tinker training data with GRPO advantages.

    See module docstring for the full algorithm.  Returns an empty list
    when every group is filtered (singleton or zero-variance).
    """
    # 1. Group by task_id — but exclude episodes carrying a rollout_error
    # signal (from OpenSpielGameAdapter's per-episode failure isolation).
    # Their outcome value is a neutral 0.0 and would taint the group baseline
    # if we let them in. Their `.steps` is empty too, so they'd emit zero
    # datums anyway — filtering here just keeps the baseline clean.
    by_task: dict[str, list[Episode]] = defaultdict(list)
    for ep in episodes:
        if ep.summary.signals and ep.summary.signals.get("rollout_error"):
            continue
        by_task[ep.task_id].append(ep)

    # 2. Compute episode-level advantages for surviving groups.
    advantage_by_id: dict[int, float] = {}
    for group in by_task.values():
        if len(group) < 2:
            continue
        returns = [_episode_return(ep) for ep in group]
        if max(returns) == min(returns):
            continue  # zero variance — no gradient signal
        mean_return = sum(returns) / len(returns)
        for ep, r in zip(group, returns, strict=True):
            advantage_by_id[id(ep)] = r - mean_return

    # 3. Emit one Datum per LLM turn.
    datums: list[tinker_types.Datum] = []
    for ep in episodes:
        adv = advantage_by_id.get(id(ep))
        if adv is None:
            continue
        for step in ep.steps:
            info = step.info
            if "prompt_tokens" not in info:
                continue  # non-LLM step (CHANCE / opponent) — skip

            prompt_tokens = list(info["prompt_tokens"])
            sampled_tokens = list(info["sampled_tokens"])
            sampling_logprobs = list(info["sampling_logprobs"])

            if len(sampled_tokens) != len(sampling_logprobs):
                raise ValueError(
                    "alignment invariant violated: "
                    f"len(sampled_tokens)={len(sampled_tokens)} != "
                    f"len(sampling_logprobs)={len(sampling_logprobs)} "
                    f"on episode {ep.id} step {step.t}"
                )
            if len(sampled_tokens) == 0:
                continue  # empty completion — nothing to learn from

            n_prompt = len(prompt_tokens)
            n_comp = len(sampled_tokens)
            full_tokens = prompt_tokens + sampled_tokens
            target_tokens = [0] * n_prompt + sampled_tokens
            logprobs_arr = [0.0] * n_prompt + sampling_logprobs
            advantages_arr = [0.0] * n_prompt + [adv] * n_comp

            datums.append(
                tinker_types.Datum(
                    model_input=tinker_types.ModelInput(
                        chunks=[
                            tinker_types.EncodedTextChunk(tokens=full_tokens),
                        ],
                    ),
                    loss_fn_inputs={
                        "target_tokens": np.asarray(target_tokens, dtype=np.int64),
                        "logprobs": np.asarray(logprobs_arr, dtype=np.float32),
                        "advantages": np.asarray(advantages_arr, dtype=np.float32),
                    },
                )
            )
    return datums
