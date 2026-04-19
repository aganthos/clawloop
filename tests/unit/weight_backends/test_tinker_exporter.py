"""Unit tests for ``_tinker_exporter.episodes_to_tinker_datums``.

Tests verify episode-level GRPO grouping, exact-token alignment via
``StepMeta.info`` keys (``prompt_tokens``/``sampled_tokens``/
``sampling_logprobs``), prompt-position zero-padding, dtype contracts,
and skip rules for non-LLM steps / singleton groups / zero-variance groups.

No network access — Tinker types are local pydantic models.
"""

from __future__ import annotations

from typing import Any

import pytest

# Optional extras (in pyproject.toml [games]); skip the whole module if
# any are missing.  Order: check BEFORE the actual imports below.
np = pytest.importorskip("numpy")
pytest.importorskip("tinker")

from clawloop.core.episode import (
    Episode,
    EpisodeSummary,
    Message,
    StepMeta,
)
from clawloop.weight_backends._tinker_exporter import (
    episodes_to_tinker_datums,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# A "turn" tuple is (prompt_tokens, sampled_tokens, sampling_logprobs).
# A turn entry of ``None`` denotes a non-LLM step (CHANCE / opponent), which
# yields a StepMeta with no prompt_tokens/sampled_tokens/sampling_logprobs in
# its ``info`` dict — the exporter must skip such steps.
TurnSpec = tuple[list[int], list[int], list[float]] | None


def _make_episode(
    task_id: str,
    terminal_reward: float,
    turns: list[TurnSpec],
) -> Episode:
    """Build an Episode with one StepMeta per turn entry.

    ``terminal_reward`` is in [-1, 1] (canonical internal range).  It is set
    via ``EpisodeSummary.total_reward`` (which expects [0, 1] and stores as
    an outcome signal mapped to [-1, 1]); we therefore pre-convert.
    """
    steps: list[StepMeta] = []
    for t, turn in enumerate(turns):
        info: dict[str, Any] = {}
        if turn is not None:
            prompt_tokens, sampled_tokens, sampling_logprobs = turn
            info["prompt_tokens"] = list(prompt_tokens)
            info["sampled_tokens"] = list(sampled_tokens)
            info["sampling_logprobs"] = list(sampling_logprobs)
        steps.append(StepMeta(t=t, reward=0.0, done=False, timing_ms=0.0, info=info))
    if steps:
        steps[-1].done = True
        steps[-1].reward = terminal_reward

    # ``EpisodeSummary.total_reward`` accepts [0, 1] and stores as outcome in
    # [-1, 1].  Convert the canonical [-1, 1] terminal_reward back to [0, 1].
    summary = EpisodeSummary(total_reward=(terminal_reward + 1.0) / 2.0)

    return Episode(
        id=Episode.new_id(),
        state_id="state-x",
        task_id=task_id,
        bench="test",
        messages=[Message(role="user", content="x")],
        step_boundaries=[0],
        steps=steps,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_grpo_groups_by_task_id_and_broadcasts_advantage() -> None:
    ep_a = _make_episode(
        task_id="t1",
        terminal_reward=1.0,
        turns=[([10, 11, 12], [20, 21], [-0.1, -0.2])],
    )
    ep_b = _make_episode(
        task_id="t1",
        terminal_reward=-1.0,
        turns=[([30, 31], [40], [-0.5])],
    )

    datums = episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling")

    assert len(datums) == 2
    # ep_a -> +1, ep_b -> -1 (mean=0)
    advs_a = list(datums[0].loss_fn_inputs["advantages"].data)
    advs_b = list(datums[1].loss_fn_inputs["advantages"].data)
    # ep_a: 3 prompt zeros + 2 completion +1's
    assert advs_a == [0.0, 0.0, 0.0, 1.0, 1.0]
    # ep_b: 2 prompt zeros + 1 completion -1
    assert advs_b == [0.0, 0.0, -1.0]


def test_grpo_skips_singleton_groups() -> None:
    ep = _make_episode(
        task_id="solo",
        terminal_reward=0.5,
        turns=[([1, 2], [3], [-0.3])],
    )
    assert episodes_to_tinker_datums([ep], loss_fn="importance_sampling") == []


def test_grpo_skips_zero_variance_groups() -> None:
    ep_a = _make_episode("t1", 0.5, [([1, 2], [3], [-0.1])])
    ep_b = _make_episode("t1", 0.5, [([4, 5], [6], [-0.1])])
    assert episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling") == []


def test_one_datum_per_llm_turn() -> None:
    # Two episodes, same task, each: LLM turn, CHANCE step, LLM turn.
    turns_a: list[TurnSpec] = [
        ([1, 2], [10, 11], [-0.1, -0.2]),
        None,  # CHANCE
        ([3, 4], [12], [-0.3]),
    ]
    turns_b: list[TurnSpec] = [
        ([5, 6], [20], [-0.4]),
        None,
        ([7, 8], [21, 22], [-0.5, -0.6]),
    ]
    ep_a = _make_episode("t1", 1.0, turns_a)
    ep_b = _make_episode("t1", -1.0, turns_b)

    datums = episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling")
    assert len(datums) == 4  # 2 episodes * 2 LLM turns each


def test_token_alignment_invariant() -> None:
    ep_a = _make_episode("t1", 1.0, [([1, 2, 3], [9, 8], [-0.1, -0.2])])
    ep_b = _make_episode("t1", -1.0, [([4], [7, 6, 5], [-0.3, -0.4, -0.5])])
    datums = episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling")
    for d in datums:
        # full token length from the model_input chunk
        full_len = len(d.model_input.chunks[0].tokens)
        assert len(d.loss_fn_inputs["target_tokens"].data) == full_len
        assert len(d.loss_fn_inputs["logprobs"].data) == full_len
        assert len(d.loss_fn_inputs["advantages"].data) == full_len


def test_prompt_positions_zero_padded() -> None:
    # n_prompt=3, n_comp=1
    ep_a = _make_episode("t1", 1.0, [([1, 2, 3], [9], [-0.7])])
    ep_b = _make_episode("t1", -1.0, [([4, 5, 6], [8], [-0.8])])
    datums = episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling")
    d = datums[0]
    target = list(d.loss_fn_inputs["target_tokens"].data)
    lp = list(d.loss_fn_inputs["logprobs"].data)
    advs = list(d.loss_fn_inputs["advantages"].data)
    assert target[:3] == [0, 0, 0]
    assert lp[:3] == [0.0, 0.0, 0.0]
    assert advs[:3] == [0.0, 0.0, 0.0]
    # ep_a advantage = 1 - mean(1, -1) = 1.0
    assert advs[3] == 1.0
    # last completion token == sampled token
    assert target[3] == 9


def test_dtypes_int64_and_float32() -> None:
    ep_a = _make_episode("t1", 1.0, [([1, 2], [3], [-0.1])])
    ep_b = _make_episode("t1", -1.0, [([4, 5], [6], [-0.2])])
    datums = episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling")
    for d in datums:
        # Tinker stores as TensorData with literal dtype strings; verify the
        # serialized dtype labels match the int64/float32 contract.
        assert d.loss_fn_inputs["target_tokens"].dtype == "int64"
        assert d.loss_fn_inputs["logprobs"].dtype == "float32"
        assert d.loss_fn_inputs["advantages"].dtype == "float32"


def test_misaligned_logprobs_raises() -> None:
    # sampled_tokens has 2 entries but sampling_logprobs has 3
    ep_a = _make_episode("t1", 1.0, [([1], [2, 3], [-0.1, -0.2, -0.3])])
    ep_b = _make_episode("t1", -1.0, [([4], [5], [-0.4])])
    with pytest.raises(ValueError, match="alignment invariant"):
        episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling")


def test_empty_completion_is_skipped_but_other_turns_emitted() -> None:
    # ep_a has one empty-completion turn and one normal turn; ep_b has two normal turns.
    ep_a = _make_episode(
        "t1",
        1.0,
        [
            ([1, 2], [], []),  # empty -> skipped
            ([3, 4], [99], [-0.7]),  # normal
        ],
    )
    ep_b = _make_episode(
        "t1",
        -1.0,
        [
            ([5, 6], [88], [-0.5]),
            ([7, 8], [77], [-0.6]),
        ],
    )
    datums = episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling")
    # ep_a contributes 1 datum, ep_b contributes 2 -> 3 total
    assert len(datums) == 3


def test_skips_non_llm_steps_without_prompt_tokens_key() -> None:
    # Both episodes are entirely non-LLM (CHANCE) — even with reward variance
    # there are no LLM turns, so zero datums.
    ep_a = _make_episode("t1", 1.0, [None, None])
    ep_b = _make_episode("t1", -1.0, [None, None])
    datums = episodes_to_tinker_datums([ep_a, ep_b], loss_fn="importance_sampling")
    assert datums == []
