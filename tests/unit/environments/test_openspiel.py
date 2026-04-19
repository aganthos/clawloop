"""Unit tests for the OpenSpiel env skeleton.

`run_episode` is implemented in Task 13 — not tested here.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# The OpenSpiel env has three optional runtime deps:
#   - pyspiel (open_spiel): for `pyspiel.load_game()` in run_episode
#   - pytest-asyncio: for @pytest.mark.asyncio support
#   - tinker: for `_tinker_sdk.async_sample` inside run_episode
# All three live in the `games` extra; CI without that extra skips.
pytest.importorskip("pyspiel")
pytest.importorskip("pytest_asyncio")
pytest.importorskip("tinker")


@pytest.mark.asyncio
async def test_sample_one_llm_attempt_raises_on_none_logprobs():
    """_sample_one_llm_attempt must hard-fail when the SamplingClient returns
    None for .logprobs — silently falling back to 0.0 would give log(1)=prob 1.0
    IS ratios for importance_sampling loss (mathematically bogus)."""
    import concurrent.futures
    from clawloop.environments import openspiel as osp

    # Build a fake sampling_client.sample() that returns sequences[0].logprobs=None
    fake_seq = MagicMock()
    fake_seq.tokens = [10, 11, 12]
    fake_seq.logprobs = None  # <-- the failure case
    fake_response = MagicMock()
    fake_response.sequences = [fake_seq]
    fut = concurrent.futures.Future()
    fut.set_result(fake_response)
    sampling_client = MagicMock()
    sampling_client.sample.return_value = fut

    fake_tokenizer = MagicMock()
    fake_renderer = MagicMock(spec=[])   # no build_generation_prompt
    fake_tokenizer.apply_chat_template.return_value = [1, 2, 3]
    fake_tokenizer.decode.return_value = "<text>"

    fake_state = MagicMock()
    fake_state.current_player.return_value = 0
    fake_state.legal_actions.return_value = [0, 1]
    fake_state.action_to_string.side_effect = lambda p, a: {0: "A", 1: "B"}[a]
    fake_state.is_simultaneous_node.return_value = False

    cfg = osp.OpenSpielTaskConfig(game_name="blackjack", seeds=[0])

    with pytest.raises(RuntimeError, match="None for .logprobs"):
        await osp._sample_one_llm_attempt(
            sampling_client=sampling_client,
            renderer=fake_renderer,
            tokenizer=fake_tokenizer,
            cfg=cfg,
            turn_messages=[],
            state=fake_state,
            player=0,
        )


def test_run_episodes_batch_concurrent():
    """run_episodes_batch gathers async episodes — episodes execute concurrently."""
    import asyncio
    from clawloop.environments.openspiel import (
        OpenSpielGameAdapter, OpenSpielTaskConfig, OpenSpielTaskEnvironment,
    )

    # Stub envs whose async run_episode sleeps briefly + records start/end times.
    timings: list[tuple[str, float, float]] = []
    import time as _time

    class _StubEnv:
        def __init__(self, name):
            self._name = name
        async def run_episode(self, agent_state, rollout_idx=None):
            t0 = _time.perf_counter()
            await asyncio.sleep(0.05)  # simulate sampling
            t1 = _time.perf_counter()
            timings.append((self._name, t0, t1))
            return f"ep_{self._name}"

    envs = {f"t_{i}": _StubEnv(f"t_{i}") for i in range(8)}
    adapter = OpenSpielGameAdapter(envs)
    task_ids = [f"t_{i}" for i in range(8)]

    t0 = _time.perf_counter()
    eps = adapter.run_episodes_batch(task_ids, agent_state=object())
    wall = _time.perf_counter() - t0

    assert eps == [f"ep_t_{i}" for i in range(8)]
    # Concurrent: wall < sum of individual waits. 8 * 0.05 = 0.4s serial;
    # expect wall < 0.2s with any real async speedup.
    assert wall < 0.25, f"expected concurrent speedup, got wall={wall:.3f}s"


def test_config_defaults():
    from clawloop.environments.openspiel import OpenSpielTaskConfig
    cfg = OpenSpielTaskConfig(game_name="blackjack", seeds=[0, 1])
    assert cfg.prompt_style == "canonical"
    assert cfg.rethink_k == 3
    assert cfg.max_turns == 50
    assert cfg.temperature == 1.0
    assert cfg.top_p == 0.95
    assert cfg.max_tokens == 128
    assert cfg.opponent is None


def test_task_env_task_id_format():
    from clawloop.environments.openspiel import (
        OpenSpielTaskConfig, OpenSpielTaskEnvironment,
    )
    cfg = OpenSpielTaskConfig(game_name="blackjack", seeds=[0])
    env = OpenSpielTaskEnvironment(cfg, seed=7)
    assert env.task_id == "blackjack_seed_7"


def test_task_env_exposes_seed_and_config():
    from clawloop.environments.openspiel import (
        OpenSpielTaskConfig, OpenSpielTaskEnvironment,
    )
    cfg = OpenSpielTaskConfig(game_name="chess", seeds=[3])
    env = OpenSpielTaskEnvironment(cfg, seed=3)
    assert env.seed == 3
    assert env.config is cfg


def test_adapter_stores_envs_keyed_by_task_id():
    from clawloop.environments.openspiel import (
        OpenSpielGameAdapter, OpenSpielTaskConfig, OpenSpielTaskEnvironment,
    )
    cfg = OpenSpielTaskConfig(game_name="blackjack", seeds=[0, 1])
    envs = {
        "blackjack_seed_0": OpenSpielTaskEnvironment(cfg, seed=0),
        "blackjack_seed_1": OpenSpielTaskEnvironment(cfg, seed=1),
    }
    adapter = OpenSpielGameAdapter(envs)
    assert "blackjack_seed_0" in adapter._envs_by_task_id
    assert "blackjack_seed_1" in adapter._envs_by_task_id


# ---------------------------------------------------------------------------
# Task 12: prompt-building / move-parsing helpers.
# ---------------------------------------------------------------------------


def _fake_blackjack_state():
    s = MagicMock()
    s.current_player.return_value = 0
    s.observation_string.return_value = "Hand: 10, 7. Dealer shows: 5."
    s.legal_actions.return_value = [0, 1]
    s.action_to_string.side_effect = lambda p, a: {0: "Hit", 1: "Stand"}[a]
    return s


def test_prompt_fallback_includes_observation_and_legal_actions():
    from clawloop.environments.openspiel import _prompt_fallback
    state = _fake_blackjack_state()
    prompt = _prompt_fallback(state, history=[], style="canonical")
    assert "Hand: 10, 7" in prompt
    assert "Hit" in prompt
    assert "Stand" in prompt
    assert "Final Answer" in prompt


def test_parse_move_fallback_final_answer_form():
    from clawloop.environments.openspiel import _parse_move_fallback
    state = _fake_blackjack_state()
    assert _parse_move_fallback("Final Answer: Hit", state) == 0
    assert _parse_move_fallback("final answer: stand", state) == 1


def test_parse_move_fallback_free_form_match():
    from clawloop.environments.openspiel import _parse_move_fallback
    state = _fake_blackjack_state()
    assert _parse_move_fallback("I think I'll Hit now.", state) == 0
    assert _parse_move_fallback("Better to stand.", state) == 1


def test_parse_move_fallback_returns_none_on_gibberish():
    from clawloop.environments.openspiel import _parse_move_fallback
    state = _fake_blackjack_state()
    assert _parse_move_fallback("some unrelated text xyzzy", state) is None


def test_parse_move_fallback_longest_match_preferred():
    """If a shorter legal string is a substring of a longer one, prefer the longest."""
    from clawloop.environments.openspiel import _parse_move_fallback
    state = MagicMock()
    state.current_player.return_value = 0
    state.legal_actions.return_value = [0, 1]
    state.action_to_string.side_effect = lambda p, a: {0: "Call", 1: "Call Bluff"}[a]
    # "Call Bluff" contains "Call" — longest-match must win.
    assert _parse_move_fallback("Final Answer: Call Bluff", state) == 1


def test_build_prompt_uses_fallback_when_game_arena_unavailable(monkeypatch):
    """When game_arena raises on import, build_prompt must return the fallback prompt."""
    from clawloop.environments.openspiel import build_prompt
    # game_arena IS installed, but we simulate it failing by patching _prompt_via_game_arena to None.
    import clawloop.environments.openspiel as osp
    monkeypatch.setattr(osp, "_prompt_via_game_arena", lambda *a, **kw: None)
    state = _fake_blackjack_state()
    prompt = build_prompt(state, history=[], style="canonical")
    assert "Hand: 10, 7" in prompt   # fallback content


def test_parse_move_uses_fallback_when_game_arena_unavailable(monkeypatch):
    from clawloop.environments.openspiel import parse_move
    # Force fallback by patching game_arena parser call to raise.
    import clawloop.environments.openspiel as osp
    def _raise(*a, **kw):
        raise ImportError("no game_arena")
    # The function tries to `from game_arena.harness import parsers` inside.
    # Patch sys.modules so the import fails.
    import sys
    monkeypatch.setitem(sys.modules, "game_arena.harness", None)
    state = _fake_blackjack_state()
    assert parse_move("Final Answer: Hit", state) == 0


# ---------------------------------------------------------------------------
# Task 13: run_episode.
# ---------------------------------------------------------------------------


def _make_fake_agent_state(
    *, sampling_client, renderer, tokenizer,
):
    """Build a minimal stand-in for AgentState.

    Using a plain object instead of the real AgentState avoids coupling this
    test to the loop module while still exposing the attributes run_episode
    reads.
    """
    from types import SimpleNamespace
    return SimpleNamespace(
        sampling_client=sampling_client,
        renderer=renderer,
        tokenizer=tokenizer,
    )


def _make_fake_sampling(tokens, logprobs):
    """Build a sampling client whose .sample() returns a REAL concurrent.futures.Future.

    Real future is required because run_episode now uses ``asyncio.wrap_future``
    which asserts ``isinstance(f, concurrent.futures.Future)``.
    """
    import concurrent.futures
    fake_seq = MagicMock()
    fake_seq.tokens = tokens
    fake_seq.logprobs = logprobs
    fake_seq.stop_reason = "stop"
    fake_response = MagicMock()
    fake_response.sequences = [fake_seq]

    fake_sampling = MagicMock()

    def _sample(**_kwargs):
        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.set_result(fake_response)
        return fut

    fake_sampling.sample.side_effect = _sample
    return fake_sampling


@pytest.mark.asyncio
async def test_run_episode_blackjack_terminates_and_captures_reward():
    """Run a full Blackjack episode with a mocked sampler that always says 'Stand'."""
    from clawloop.environments import openspiel as osp

    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = [1, 2, 3]
    fake_tokenizer.decode.return_value = "Final Answer: Stand"
    # Renderer without build_generation_prompt -> force the fallback via tokenizer.
    fake_renderer = MagicMock(spec=[])

    fake_sampling = _make_fake_sampling(
        tokens=[10, 11, 12], logprobs=[-0.1, -0.2, -0.3],
    )

    cfg = osp.OpenSpielTaskConfig(
        game_name="blackjack", seeds=[0], max_turns=10, max_tokens=8,
    )
    env = osp.OpenSpielTaskEnvironment(cfg, seed=0)
    agent_state = _make_fake_agent_state(
        sampling_client=fake_sampling,
        renderer=fake_renderer,
        tokenizer=fake_tokenizer,
    )
    episode = await env.run_episode(agent_state)

    # Standing with the seed=0 initial hand ends the episode; effective reward
    # in Blackjack is in {-1, 0, 1}.
    assert episode.task_id == "blackjack_seed_0"
    assert episode.summary.effective_reward() in (-1.0, 0.0, 1.0)
    llm_steps = [s for s in episode.steps if "prompt_tokens" in s.info]
    assert len(llm_steps) >= 1
    for s in llm_steps:
        assert len(s.info["sampled_tokens"]) == len(s.info["sampling_logprobs"])
    # Illegal parse was not triggered.
    ip = episode.summary.signals.get("illegal_parse")
    if ip is not None:
        assert ip.value == 0.0 or ip == 0.0
    # Terminal StepMeta carries the final reward and done=True.
    assert episode.steps[-1].done is True


@pytest.mark.asyncio
async def test_run_episode_illegal_parse_terminates_with_zero_reward():
    """Unparseable responses across all retries -> reward 0 + illegal_parse signal."""
    from clawloop.environments import openspiel as osp

    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = [1]
    fake_tokenizer.decode.return_value = "totally unparseable gibberish xyzzy"
    fake_renderer = MagicMock(spec=[])

    fake_sampling = _make_fake_sampling(tokens=[99], logprobs=[-0.5])

    cfg = osp.OpenSpielTaskConfig(
        game_name="blackjack", seeds=[0], max_turns=10, max_tokens=8, rethink_k=1,
    )
    env = osp.OpenSpielTaskEnvironment(cfg, seed=0)
    agent_state = _make_fake_agent_state(
        sampling_client=fake_sampling,
        renderer=fake_renderer,
        tokenizer=fake_tokenizer,
    )
    episode = await env.run_episode(agent_state)

    assert episode.summary.signals.get("illegal_parse") is not None
    # Final reward is 0.0 — neither illegal penalty nor actual game return.
    assert episode.summary.effective_reward() == 0.0
    assert episode.steps[-1].done is True
    assert episode.steps[-1].info.get("illegal_after_retries") is True


@pytest.mark.asyncio
async def test_run_episode_requires_sampling_client():
    from clawloop.environments import openspiel as osp

    cfg = osp.OpenSpielTaskConfig(game_name="blackjack", seeds=[0])
    env = osp.OpenSpielTaskEnvironment(cfg, seed=0)
    agent_state = _make_fake_agent_state(
        sampling_client=None, renderer=MagicMock(), tokenizer=MagicMock(),
    )
    with pytest.raises(RuntimeError, match="sampling_client"):
        await env.run_episode(agent_state)


@pytest.mark.asyncio
async def test_run_episode_requires_renderer_and_tokenizer():
    from clawloop.environments import openspiel as osp

    cfg = osp.OpenSpielTaskConfig(game_name="blackjack", seeds=[0])
    env = osp.OpenSpielTaskEnvironment(cfg, seed=0)
    agent_state = _make_fake_agent_state(
        sampling_client=MagicMock(), renderer=None, tokenizer=None,
    )
    with pytest.raises(RuntimeError, match="renderer"):
        await env.run_episode(agent_state)
