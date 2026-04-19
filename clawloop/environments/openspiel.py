"""OpenSpiel environment adapter for ClawLoop.

Wraps OpenSpiel games + game_arena prompt/parse primitives. Each
`OpenSpielTaskEnvironment` is one (game, seed) pair; running it produces
a ClawLoop `Episode`. The adapter dispatches by task_id.

Design invariants for ``run_episode``:

- CHANCE nodes are advanced by sampling from ``state.chance_outcomes()`` with
  a seeded ``numpy.random.default_rng``. No LLM call, no message appended,
  no StepMeta recorded.
- LLM turns persist exact tokens: ``prompt_tokens`` go on the ``StepMeta.info``
  bag and ``sampled_tokens`` + ``sampling_logprobs`` pair up in
  ``Message.logprobs``. The downstream exporter never re-tokenizes.
- The terminal reward lives on ``summary.signals["outcome"]`` (name-keyed)
  so :meth:`EpisodeSummary.effective_reward` returns the canonical [-1, 1]
  value for the LLM player.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from dataclasses import replace as dc_replace
from typing import Any, Literal, Protocol

# The LLM always plays as seat 0 across every game in this env. Opponents
# (including self-play in a future release) cover seats 1..N-1.
LLM_PID: int = 0


class OpponentPolicy(Protocol):
    """Scripted policy for non-LLM players (2P+ games only)."""

    def act(self, state: Any) -> int: ...


class RandomPolicy:
    """Uniformly sample a legal action for the current player.

    Works for every OpenSpiel turn-based game. For simultaneous-move games the
    driver loop queries ``act(state, player_id)`` via the compatible
    :meth:`act_for_player` shim — the Protocol stays minimal.
    """

    def __init__(self, seed: int | None = None) -> None:
        import numpy as _np

        self._rng = _np.random.default_rng(seed)

    def act(self, state: Any) -> int:
        legal = state.legal_actions()
        return int(self._rng.choice(legal))

    def act_for_player(self, state: Any, player: int) -> int:
        legal = state.legal_actions(player)
        return int(self._rng.choice(legal))


def _resolve_opponent(spec: Any) -> OpponentPolicy | None:
    """Translate a YAML ``opponent`` spec into an :class:`OpponentPolicy`.

    Accepted forms:
        None / missing            -> no opponent (1P / chance-only game)
        "random"                  -> :class:`RandomPolicy`
        {"type": "random", "seed": 7}
        an already-instantiated OpponentPolicy               -> passed through
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        if spec == "random":
            return RandomPolicy()
        raise ValueError(f"unknown opponent spec: {spec!r}")
    if isinstance(spec, dict):
        kind = spec.get("type")
        if kind == "random":
            return RandomPolicy(seed=spec.get("seed"))
        raise ValueError(f"unknown opponent spec: {spec!r}")
    # Assume caller gave us an OpponentPolicy-like object.
    return spec


@dataclass
class OpenSpielTaskConfig:
    game_name: str  # e.g. "blackjack"
    seeds: list[int]  # scenario pool
    prompt_style: Literal["canonical", "ascii"] = "canonical"
    rethink_k: int = 3
    max_turns: int = 50
    opponent: OpponentPolicy | None = None  # None for 1P games
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 128


def _build_generation_prompt_tokens(
    renderer: Any,
    tokenizer: Any,
    messages: list,
) -> list[int]:
    """Render messages into prompt tokens the SamplingClient will consume.

    Preferred: ``renderer.build_generation_prompt(messages) -> list[int]``.
    Fallback: serialize to OpenAI-style dicts and use
    ``tokenizer.apply_chat_template(..., tokenize=True, add_generation_prompt=True)``.
    The fallback lets us support renderer API drift across tinker_cookbook
    versions without breaking the env.
    """
    # tinker_cookbook renderers expect dicts with ["role"]/["content"] keys,
    # not ClawLoop Message dataclasses. Serialize first.
    openai_msgs = [m.to_openai_dict() for m in messages]
    if hasattr(renderer, "build_generation_prompt"):
        out = renderer.build_generation_prompt(openai_msgs)
        # tinker_cookbook's renderer returns a ModelInput (multi-chunk). Flatten
        # to a single list[int] — the exporter relies on this being contiguous.
        if hasattr(out, "chunks"):
            tokens: list[int] = []
            for chunk in out.chunks:
                chunk_tokens = getattr(chunk, "tokens", None)
                if chunk_tokens is not None:
                    tokens.extend(int(t) for t in chunk_tokens)
            return tokens
        return [int(t) for t in out]
    return [
        int(t)
        for t in tokenizer.apply_chat_template(
            openai_msgs,
            tokenize=True,
            add_generation_prompt=True,
        )
    ]


async def _sample_one_llm_attempt(
    *,
    sampling_client: Any,
    renderer: Any,
    tokenizer: Any,
    cfg: "OpenSpielTaskConfig",
    turn_messages: list,
    state: Any,
    player: int,
) -> tuple[int | None, list[int], list[int], list[float], str, float]:
    """One sample -> parse attempt.

    Shared by the regular sequential-turn loop and the simultaneous-move loop
    so the two branches don't duplicate the build-prompt/submit/await/decode/
    parse scaffolding (~50 LoC each).

    Returns ``(action_or_None, prompt_tokens, sampled_tokens,
    sampling_logprobs, response_text, timing_ms)``. ``action`` is validated
    against the correct player's legal_actions (simultaneous games need the
    player arg; sequential games default to the current player).
    """
    import asyncio as _aio

    # Local import: the SDK adapter lives in weight_backends/; avoid paying
    # the import cost at module load for non-Tinker callers.
    from clawloop.weight_backends import _tinker_sdk

    prompt_tokens = _build_generation_prompt_tokens(
        renderer,
        tokenizer,
        turn_messages,
    )
    t0 = time.perf_counter()
    fut = _tinker_sdk.async_sample(
        sampling_client,
        prompt_tokens=prompt_tokens,
        num_samples=1,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    resp = await _aio.wrap_future(fut)
    timing_ms = (time.perf_counter() - t0) * 1000.0
    seq = resp.sequences[0]
    sampled_tokens = list(seq.tokens)
    # Logprobs are REQUIRED for the loss functions we use
    # (``importance_sampling``, ``ppo``, ``cispo``, ``dro``).  A 0.0 fallback
    # would mean log(1) = probability 1.0 → IS ratios of 1.0 → mathematically
    # bogus training signal.  If Tinker's SamplingClient ever returns None
    # we raise hard so the caller sees it rather than silent corruption.
    if seq.logprobs is None:
        raise RuntimeError(
            "SamplingClient returned None for .logprobs; Tinker RL losses "
            "(importance_sampling / ppo / cispo / dro) require real per-token "
            "logprobs. If your workflow uses loss_fn='cross_entropy' (which "
            "ignores logprobs), bypass this helper and build the Datum "
            "without logprobs in loss_fn_inputs."
        )
    sampling_logprobs = list(seq.logprobs)
    if len(sampled_tokens) != len(sampling_logprobs):
        raise RuntimeError(
            f"sampled_tokens/logprobs length mismatch: "
            f"{len(sampled_tokens)} != {len(sampling_logprobs)}"
        )
    response_text = tokenizer.decode(sampled_tokens)
    action = parse_move(response_text, state)
    # Validate legality against the correct player (simultaneous vs sequential).
    if action is not None:
        legal = (
            state.legal_actions(player) if state.is_simultaneous_node() else state.legal_actions()
        )
        if action not in legal:
            action = None
    return action, prompt_tokens, sampled_tokens, sampling_logprobs, response_text, timing_ms


def _build_retry_hint(state: Any) -> str:
    """Build a user-message hint shown after an illegal-move parse failure."""
    # Simultaneous nodes return a sentinel (<0) from current_player(); the LLM
    # always drives seat 0 in those cases. Also, simultaneous games need the
    # per-player legal_actions(player) API — passing no arg raises on some
    # games (matrix_mp).
    raw = state.current_player()
    player = raw if raw >= 0 else LLM_PID
    legal = state.legal_actions(player) if state.is_simultaneous_node() else state.legal_actions()
    legal_strs = [state.action_to_string(player, a) for a in legal]
    return (
        "Your previous response did not contain a legal move. "
        f"Legal moves are: {', '.join(legal_strs)}. "
        "Respond with exactly `Final Answer: <move>` where <move> is one of "
        "the listed legal moves."
    )


class OpenSpielTaskEnvironment:
    """One game + one seed. Produces one Episode per run_episode() call."""

    def __init__(self, config: OpenSpielTaskConfig, seed: int) -> None:
        self._config = config
        self._seed = seed

    @property
    def task_id(self) -> str:
        return f"{self._config.game_name}_seed_{self._seed}"

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def config(self) -> OpenSpielTaskConfig:
        return self._config

    async def run_episode(
        self,
        agent_state: Any,
        rollout_idx: int | None = None,
    ):
        """Roll out one OpenSpiel game, producing a ClawLoop Episode.

        ``rollout_idx`` (optional) gives callers a reproducibility knob: when
        supplied, each rollout in a GRPO group gets a deterministic-but-unique
        RNG seed derived from ``(self._seed, rollout_idx)`` so chance outcomes
        still diverge across the group (→ variance for GRPO) but the run as a
        whole is reproducible. When ``None`` (the default), RNG is unseeded —
        matches previous behavior for ad-hoc single-episode callers.

        See module docstring for other design invariants.
        """
        import numpy as np
        import pyspiel

        from clawloop.core.episode import (
            Episode,
            EpisodeSummary,
            Message,
            StepMeta,
            TokenLogProb,
        )
        from clawloop.core.reward import RewardSignal

        cfg = self._config
        if rollout_idx is not None:
            # Reproducible path: combine scenario seed and rollout index with
            # a large multiplier so adjacent (seed, idx) pairs don't collide.
            # Different idx → different chance outcomes → GRPO variance.
            rng = np.random.default_rng(self._seed * 2654435761 + rollout_idx)
        else:
            # Unseeded path — keeps the previous fresh-RNG behavior for callers
            # that don't care about reproducibility.
            rng = np.random.default_rng()
        game = pyspiel.load_game(cfg.game_name)
        state = game.new_initial_state()

        sampling_client = getattr(agent_state, "sampling_client", None)
        if sampling_client is None:
            raise RuntimeError(
                "agent_state.sampling_client is None — wire TinkerWeightsBackend first"
            )
        renderer = getattr(agent_state, "renderer", None)
        tokenizer = getattr(agent_state, "tokenizer", None)
        if renderer is None or tokenizer is None:
            raise RuntimeError(
                "agent_state.renderer / tokenizer not set — "
                "learning_loop refresh missing (Task 15)"
            )

        llm_pid = LLM_PID
        messages: list[Message] = []
        steps: list[StepMeta] = []
        illegal_parse = False
        turn_idx = 0

        while not state.is_terminal() and turn_idx < cfg.max_turns:
            if state.is_chance_node():
                outcomes = state.chance_outcomes()
                actions = [a for a, _ in outcomes]
                probs = np.array([p for _, p in outcomes], dtype=np.float64)
                total = float(probs.sum())
                probs = probs / total if total > 0 else probs
                action = int(rng.choice(actions, p=probs))
                state.apply_action(action)
                continue

            # Simultaneous-move games (matrix_mp, matching_pennies_3p).
            # current_player == SIMULTANEOUS; every seat picks at once. The
            # LLM picks for seat 0 via the usual prompt/sample/parse loop; all
            # other seats use the opponent policy. Then state.apply_actions
            # with the list of all players' actions advances the step.
            if state.is_simultaneous_node():
                # Sample LLM action for seat 0 below via the same retry loop,
                # then collect random actions for seats 1..N-1.
                n_players = state.num_players()
                # Build actions[] for each seat; seat 0 comes from LLM.
                seat_actions: list[int | None] = [None] * n_players
                # Fill opponent seats first so we know if we're stuck.
                for p in range(n_players):
                    if p == llm_pid:
                        continue
                    assert (
                        cfg.opponent is not None
                    ), f"{cfg.game_name} simultaneous node has seat {p} but no opponent"
                    # Some opponent implementations expose act_for_player;
                    # fall back to act() for single-seat pollicies.
                    if hasattr(cfg.opponent, "act_for_player"):
                        seat_actions[p] = int(cfg.opponent.act_for_player(state, p))
                    else:
                        seat_actions[p] = int(cfg.opponent.act(state))
                # Now LLM action for seat 0 — reuse the sampling loop below
                # by faking current_player for the next block; handled
                # specially by jumping into the shared LLM-turn branch.
                # For clarity we inline a minimal version here.
                prompt_str = build_prompt(state, messages, cfg.prompt_style)
                turn_start = len(messages)
                messages.append(Message(role="user", content=prompt_str))
                resolved = False
                llm_action: int | None = None
                for attempt in range(cfg.rethink_k + 1):
                    (
                        action,
                        prompt_tokens,
                        sampled_tokens,
                        sampling_logprobs,
                        response_text,
                        timing_ms,
                    ) = await _sample_one_llm_attempt(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        tokenizer=tokenizer,
                        cfg=cfg,
                        turn_messages=messages[turn_start:],
                        state=state,
                        player=llm_pid,
                    )
                    assistant_msg = Message(
                        role="assistant",
                        content=response_text,
                        logprobs=[
                            TokenLogProb(token=str(t), logprob=float(lp))
                            for t, lp in zip(sampled_tokens, sampling_logprobs)
                        ],
                    )
                    if action is not None:
                        messages.append(assistant_msg)
                        steps.append(
                            StepMeta(
                                t=turn_idx,
                                reward=0.0,
                                done=False,
                                timing_ms=timing_ms,
                                info={
                                    "prompt_tokens": prompt_tokens,
                                    "sampled_tokens": sampled_tokens,
                                    "sampling_logprobs": sampling_logprobs,
                                    "legal_actions": list(state.legal_actions(llm_pid)),
                                    "chosen_action": int(action),
                                    "rethinks": attempt,
                                    "simultaneous": True,
                                },
                            )
                        )
                        llm_action = int(action)
                        resolved = True
                        break
                    messages.append(assistant_msg)
                    messages.append(
                        Message(
                            role="user",
                            content=_build_retry_hint(state),
                        )
                    )
                if not resolved:
                    illegal_parse = True
                    steps.append(
                        StepMeta(
                            t=turn_idx,
                            reward=0.0,
                            done=True,
                            timing_ms=0.0,
                            info={"illegal_after_retries": True, "simultaneous": True},
                        )
                    )
                    break
                seat_actions[llm_pid] = llm_action
                state.apply_actions([int(a) for a in seat_actions])
                turn_idx += 1
                continue

            current = state.current_player()
            if current == llm_pid:
                prompt_str = build_prompt(state, messages, cfg.prompt_style)
                # Track the start of THIS turn's messages. For state-based
                # OpenSpiel games the observation is Markovian, so we pass
                # only the current turn (+ any retry pairs) to the renderer.
                # Otherwise long games (2048 at 200 turns) blow past the
                # model's context window.
                turn_start = len(messages)
                messages.append(Message(role="user", content=prompt_str))
                resolved = False
                for attempt in range(cfg.rethink_k + 1):
                    (
                        action,
                        prompt_tokens,
                        sampled_tokens,
                        sampling_logprobs,
                        response_text,
                        timing_ms,
                    ) = await _sample_one_llm_attempt(
                        sampling_client=sampling_client,
                        renderer=renderer,
                        tokenizer=tokenizer,
                        cfg=cfg,
                        turn_messages=messages[turn_start:],
                        state=state,
                        player=llm_pid,
                    )
                    assistant_msg = Message(
                        role="assistant",
                        content=response_text,
                        logprobs=[
                            TokenLogProb(token=str(t), logprob=float(lp))
                            for t, lp in zip(sampled_tokens, sampling_logprobs)
                        ],
                    )
                    if action is not None:
                        messages.append(assistant_msg)
                        steps.append(
                            StepMeta(
                                t=turn_idx,
                                reward=0.0,
                                done=False,
                                timing_ms=timing_ms,
                                info={
                                    "prompt_tokens": prompt_tokens,
                                    "sampled_tokens": sampled_tokens,
                                    "sampling_logprobs": sampling_logprobs,
                                    "legal_actions": list(state.legal_actions()),
                                    "chosen_action": int(action),
                                    "rethinks": attempt,
                                },
                            )
                        )
                        state.apply_action(int(action))
                        resolved = True
                        break
                    # Illegal: append rejected response + retry hint and try again.
                    messages.append(assistant_msg)
                    messages.append(
                        Message(
                            role="user",
                            content=_build_retry_hint(state),
                        )
                    )

                if not resolved:
                    illegal_parse = True
                    steps.append(
                        StepMeta(
                            t=turn_idx,
                            reward=0.0,
                            done=True,
                            timing_ms=0.0,
                            info={"illegal_after_retries": True},
                        )
                    )
                    break
            else:
                assert (
                    cfg.opponent is not None
                ), f"{cfg.game_name} has non-LLM player {current} but no opponent configured"
                action = int(cfg.opponent.act(state))
                state.apply_action(action)
            turn_idx += 1

        # Termination — attach final reward to the last StepMeta.
        if state.is_terminal():
            final_reward = float(state.returns()[llm_pid])
        else:
            final_reward = 0.0
        if steps:
            last = steps[-1]
            steps[-1] = dc_replace(last, reward=final_reward, done=True)

        signals = {
            "outcome": RewardSignal(
                name="outcome",
                value=final_reward,
                confidence=1.0,
            ),
        }
        if illegal_parse:
            signals["illegal_parse"] = RewardSignal(
                name="illegal_parse",
                value=1.0,
                confidence=1.0,
            )

        summary = EpisodeSummary(signals=signals)

        return Episode(
            id=Episode.new_id(),
            state_id=agent_state.state_id().combined_hash
            if hasattr(agent_state, "state_id")
            else "",
            task_id=self.task_id,
            bench="openspiel",
            messages=messages,
            step_boundaries=[],
            steps=steps,
            summary=summary,
        )


class OpenSpielGameAdapter:
    """Dispatches run_episode(task_id, agent_state) to the per-seed env.

    Implements the AdapterLike protocol with a SYNC run_episode; the per-seed
    env's async run_episode is executed under an event loop via
    :func:`clawloop.utils.async_bridge.run_async` (same pattern as
    :class:`~clawloop.environments.harbor.HarborAdapter`).
    """

    def __init__(self, envs_by_task_id: dict[str, OpenSpielTaskEnvironment]) -> None:
        self._envs_by_task_id = envs_by_task_id

    def run_episode(self, task_id: str, agent_state: Any):
        from clawloop.utils.async_bridge import run_async

        env = self._envs_by_task_id[task_id]
        return run_async(env.run_episode(agent_state))

    @staticmethod
    def _make_error_episode(task_id: str, exc: BaseException):
        """Stub Episode for a failed rollout — preserves batch cardinality
        so GRPO grouping stays sane; carries the error on signals / metadata
        so the logger can surface per-iter failure rates.
        """
        from clawloop.core.episode import Episode, EpisodeSummary
        from clawloop.core.reward import RewardSignal

        summary = EpisodeSummary(
            signals={
                "outcome": RewardSignal(name="outcome", value=0.0, confidence=1.0),
                "rollout_error": RewardSignal(name="rollout_error", value=1.0, confidence=1.0),
            }
        )
        return Episode(
            id=Episode.new_id(),
            state_id="",
            task_id=task_id,
            bench="openspiel",
            messages=[],
            step_boundaries=[],
            steps=[],
            summary=summary,
            metadata={"error": f"{type(exc).__name__}: {exc}"},
        )

    async def run_episodes_batch_async(
        self,
        task_ids: list[str],
        agent_state: Any,
    ) -> list:
        """Async rollout of many episodes concurrently.

        Call this from an existing event loop (Streamlit, FastAPI, another
        asyncio coroutine). For the common synchronous driver (ClawLoop's
        :func:`~clawloop.core.loop.learning_loop`) use :meth:`run_episodes_batch`
        which wraps this one.

        Tinker's :class:`SamplingClient.sample` returns a ``ConcurrentFuture``
        so N in-flight requests run in parallel up to the account's quota.
        Our per-episode loop is sequential (turns are causally dependent)
        but across episodes we gain an N× speedup — this turns a ~1h
        serial mixed-game iter into ~10 min.

        Each ``task_id`` entry maps to one episode. Duplicates are allowed
        (GRPO needs K rollouts per same scenario) — each rollout gets the
        enumeration index as its ``rollout_idx``, so chance outcomes
        diverge while remaining reproducible across runs.
        """
        import asyncio

        coros = [
            self._envs_by_task_id[tid].run_episode(
                agent_state,
                rollout_idx=i,
            )
            for i, tid in enumerate(task_ids)
        ]
        # return_exceptions=True isolates per-episode failures — one
        # transient Tinker sampling error shouldn't abort a 32-episode
        # iter. Failing episodes are replaced with a stub Episode that
        # carries the error on its summary signals so downstream logging
        # can surface the rate without breaking the training loop.
        results = await asyncio.gather(*coros, return_exceptions=True)
        out: list = []
        for task_id, res in zip(task_ids, results):
            if isinstance(res, BaseException):
                out.append(self._make_error_episode(task_id, res))
            else:
                out.append(res)
        return out

    def run_episodes_batch(
        self,
        task_ids: list[str],
        agent_state: Any,
    ) -> list:
        """Synchronous wrapper around :meth:`run_episodes_batch_async`.

        Spins up a fresh event loop via ``asyncio.run``. Refuses to be
        called from an already-running event loop — use
        :meth:`run_episodes_batch_async` instead in that case.
        """
        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_episodes_batch_async(task_ids, agent_state))
        raise RuntimeError(
            "run_episodes_batch cannot be called from inside a running event "
            "loop — use run_episodes_batch_async instead."
        )


# ---------------------------------------------------------------------------
# Prompt-building / move-parsing helpers (Task 12).
#
# Pure functions: no network, no SDK state. Task 13's per-turn loop uses these.
# Strategy: prefer game_arena's harness primitives when available; fall back
# to OpenSpiel-native state/legal-action strings otherwise. Both paths share
# the same input/output shapes so callers stay agnostic.
# ---------------------------------------------------------------------------


def _state_observation(state: Any, player: int) -> str:
    """Return a human-readable state string, trying the methods OpenSpiel
    games implement in varying combinations. sheriff lacks
    ``observation_string`` (has ``information_state_string``); hanabi / maedn
    lack ``information_state_string`` (have ``observation_string``). Every
    game implements ``str(state)`` as a safe last resort."""
    for fn_name in ("observation_string", "information_state_string"):
        fn = getattr(state, fn_name, None)
        if fn is None:
            continue
        try:
            return fn(player)
        except Exception:
            continue
    return str(state)


def _prompt_fallback(state: Any, history: list, style: str) -> str:
    """OpenSpiel-native prompt. Works for 1P / chance / 2P / multi-player alike."""
    # For simultaneous nodes, current_player() returns a sentinel — the LLM
    # always takes seat 0 in that case (we drive all non-LLM seats via
    # opponent policy, so seat 0's observation is what we render).
    raw = state.current_player()
    player = raw if raw >= 0 else 0
    observation = _state_observation(state, player)
    legal = state.legal_actions(player) if state.is_simultaneous_node() else state.legal_actions()
    legal_strs = [state.action_to_string(player, a) for a in legal]
    lines = [
        f"You are player {player}.",
        "Current game state:",
        observation,
        "",
        f"Your legal moves: {', '.join(legal_strs)}",
        "",
        "Respond with your chosen move wrapped as `Final Answer: <move>`.",
    ]
    return "\n".join(lines)


def _parse_move_fallback(response: str, state: Any) -> int | None:
    """Extract a legal action from free-form text. Returns action int or None.

    Strategy:
    1. Prefer `Final Answer: <move>` form.
    2. Longest-match the candidate against legal action strings (case-insensitive).
    3. If still no match, scan the whole response for any legal action string.
    """
    raw = state.current_player()
    player = raw if raw >= 0 else 0
    legal = state.legal_actions(player) if state.is_simultaneous_node() else state.legal_actions()
    legal_strs = [(a, state.action_to_string(player, a)) for a in legal]

    m = re.search(
        r"Final Answer\s*[:\-]\s*(.+?)(?:$|\n|[.!,])",
        response,
        flags=re.IGNORECASE,
    )
    candidate = m.group(1).strip() if m else response
    candidate_lower = candidate.lower()

    # Longest-match among legal action strings within the candidate segment.
    best: tuple[int, int] | None = None  # (length, action)
    for a, s in legal_strs:
        s_lower = s.lower()
        if s_lower and s_lower in candidate_lower:
            if best is None or len(s_lower) > best[0]:
                best = (len(s_lower), a)
    if best is not None:
        return best[1]

    # Whole-response fallback.
    response_lower = response.lower()
    for a, s in legal_strs:
        if s.lower() and s.lower() in response_lower:
            return a
    return None


def _prompt_via_game_arena(state: Any, history: list, style: str) -> str | None:
    """Try game_arena's prompt_generation. Return None if unsupported for this game."""
    try:
        from game_arena.harness import prompt_generation  # type: ignore[import-not-found]
    except Exception:
        return None
    try:
        # game_arena 0.x: build(state=..., style=..., history=...) — signature may
        # vary across pin; wrap in try/except to gracefully fall back.
        return prompt_generation.build(state=state, style=style, history=history)
    except Exception:
        return None


def build_prompt(state: Any, history: list, style: str) -> str:
    """Prefer game_arena; fall back to OpenSpiel-native."""
    return _prompt_via_game_arena(state, history, style) or _prompt_fallback(state, history, style)


def parse_move(response: str, state: Any) -> int | None:
    """Prefer game_arena parser; fall back to regex-over-legal-strings."""
    try:
        from game_arena.harness import parsers  # type: ignore[import-not-found]
    except Exception:
        return _parse_move_fallback(response, state)

    try:
        action = parsers.parse_move(response, state.legal_actions())
        if action is not None:
            return int(action)
    except Exception:
        pass
    return _parse_move_fallback(response, state)
