"""Tests for LocalEvolver — wraps Reflector + GEPA + Paradigm."""

from unittest.mock import MagicMock

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.evolver import EvolverContext, EvolverResult, HarnessSnapshot
from clawloop.core.reward import RewardSignal
from clawloop.evolvers.local import LocalEvolver
from clawloop.layers.harness import Insight, Playbook, PromptCandidate


def _make_episode(reward: float = 0.5, bench: str = "test") -> Episode:
    return Episode(
        id="ep-1",
        state_id="s1",
        task_id="t1",
        bench=bench,
        messages=[Message(role="user", content="hi"), Message(role="assistant", content="hello")],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100)],
        summary=EpisodeSummary(total_reward=reward),
    )


def _make_snapshot() -> HarnessSnapshot:
    return HarnessSnapshot(
        system_prompts={"test": "You are helpful."},
        playbook_entries=[],
        pareto_fronts={"test": [{"text": "You are helpful.", "scores": {"t1": 0.8}}]},
        playbook_generation=0,
        playbook_version=0,
    )


def test_local_evolver_name():
    evolver = LocalEvolver()
    assert evolver.name() == "local"


def test_local_evolver_no_components_returns_empty():
    evolver = LocalEvolver()
    result = evolver.evolve(
        episodes=[_make_episode()],
        harness_state=_make_snapshot(),
        context=EvolverContext(),
    )
    assert isinstance(result, EvolverResult)
    assert result.insights == []
    assert result.candidates == {}
    assert result.paradigm_shift is False
    assert result.run_id == ""


def test_local_evolver_with_reflector():
    reflector = MagicMock()
    reflector.config = MagicMock()
    reflector.config.reflection_batch_size = 10
    reflector.reflect.return_value = [
        Insight(action="add", content="be concise", tags=["test"]),
    ]

    playbook = Playbook()
    evolver = LocalEvolver(reflector=reflector)
    result = evolver.evolve(
        episodes=[_make_episode()],
        harness_state=_make_snapshot(),
        context=EvolverContext(),
    )
    assert len(result.insights) == 1
    assert result.insights[0].content == "be concise"
    reflector.reflect.assert_called_once()


def test_local_evolver_with_paradigm_stagnating():
    paradigm = MagicMock()
    paradigm.generate.return_value = [
        Insight(action="add", content="try MCTS", tags=["paradigm"]),
    ]

    evolver = LocalEvolver(paradigm=paradigm)
    result = evolver.evolve(
        episodes=[_make_episode()],
        harness_state=_make_snapshot(),
        context=EvolverContext(
            is_stagnating=True,
            reward_history=[0.5, 0.5, 0.5, 0.5, 0.5],
            tried_paradigms=[],
        ),
    )
    assert result.paradigm_shift is True
    assert len(result.insights) >= 1
    assert any(i.content == "try MCTS" for i in result.insights)
    paradigm.generate.assert_called_once()


def test_local_evolver_paradigm_not_called_when_not_stagnating():
    paradigm = MagicMock()

    evolver = LocalEvolver(paradigm=paradigm)
    result = evolver.evolve(
        episodes=[_make_episode()],
        harness_state=_make_snapshot(),
        context=EvolverContext(is_stagnating=False),
    )
    assert result.paradigm_shift is False
    paradigm.generate.assert_not_called()


def test_local_evolver_with_gepa():
    gepa = MagicMock()
    child = PromptCandidate(id="pc-child", text="improved prompt", generation=1, parent_id="pc-1")
    gepa.config = MagicMock()
    gepa.config.max_mutations_per_step = 1
    gepa.config.max_crossovers_per_step = 0
    gepa.mutate.return_value = child

    evolver = LocalEvolver(prompt_evolver=gepa)

    snap = _make_snapshot()
    result = evolver.evolve(
        episodes=[_make_episode(reward=0.2, bench="test")],
        harness_state=snap,
        context=EvolverContext(),
    )
    assert "test" in result.candidates
    assert len(result.candidates["test"]) >= 1
    assert result.candidates["test"][0].text == "improved prompt"
