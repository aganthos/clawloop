"""Tests for PromptEvolver — mutation and crossover operators."""

from __future__ import annotations

import json

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.evolution import PromptEvolver
from clawloop.learning_layers.harness import PromptCandidate
from clawloop.llm import MockLLMClient

# -- Factories ----------------------------------------------------------------


def _make_episode(task_id: str = "t1", reward: float = 0.2) -> Episode:
    return Episode(
        id=Episode.new_id(),
        state_id="deadbeef",
        task_id=task_id,
        bench="test",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


def _make_parent(
    text: str = "You are helpful.",
    generation: int = 0,
) -> PromptCandidate:
    return PromptCandidate(id="pc-test001", text=text, generation=generation)


# -- Mutation tests -----------------------------------------------------------


def test_mutate_produces_valid_candidate():
    """Mock LLM returns valid JSON -> child has parent_id, generation+1, new text."""
    mock_response = json.dumps({"revised_prompt": "Improved system prompt"})
    llm = MockLLMClient(responses=[mock_response])
    evolver = PromptEvolver(llm=llm)
    parent = _make_parent()
    episodes = [_make_episode()]

    child = evolver.mutate(parent, episodes)

    assert child is not None
    assert child.text == "Improved system prompt"
    assert child.generation == parent.generation + 1
    assert child.parent_id == parent.id
    assert child.id != parent.id


def test_mutate_preserves_lineage():
    """parent_id on the child must equal parent.id."""
    mock_response = json.dumps({"revised_prompt": "Better prompt"})
    llm = MockLLMClient(responses=[mock_response])
    evolver = PromptEvolver(llm=llm)
    parent = _make_parent()

    child = evolver.mutate(parent, [_make_episode()])

    assert child is not None
    assert child.parent_id == parent.id


def test_mutate_returns_none_on_invalid_json():
    """Garbage LLM output -> None, no crash."""
    llm = MockLLMClient(responses=["this is not json at all {{{"])
    evolver = PromptEvolver(llm=llm)
    parent = _make_parent()

    result = evolver.mutate(parent, [_make_episode()])

    assert result is None


def test_mutate_returns_none_on_missing_key():
    """Valid JSON but missing 'revised_prompt' key -> None."""
    llm = MockLLMClient(responses=[json.dumps({"wrong_key": "oops"})])
    evolver = PromptEvolver(llm=llm)

    result = evolver.mutate(_make_parent(), [_make_episode()])

    assert result is None


def test_mutate_returns_none_on_empty_revised_prompt():
    """revised_prompt is empty string -> None."""
    llm = MockLLMClient(responses=[json.dumps({"revised_prompt": ""})])
    evolver = PromptEvolver(llm=llm)

    result = evolver.mutate(_make_parent(), [_make_episode()])

    assert result is None


def test_mutate_includes_episode_context():
    """The LLM prompt must contain episode details (task_id, reward, messages)."""
    mock_response = json.dumps({"revised_prompt": "Fixed prompt"})
    llm = MockLLMClient(responses=[mock_response])
    evolver = PromptEvolver(llm=llm)
    parent = _make_parent(text="Original system prompt")
    episode = _make_episode(task_id="special-task-42", reward=0.1)

    evolver.mutate(parent, [episode])

    # MockLLMClient records every call in call_log
    assert len(llm.call_log) == 1
    messages, _kwargs = llm.call_log[0]
    user_msg = messages[1]["content"]
    assert "Original system prompt" in user_msg
    assert "special-task-42" in user_msg
    assert "Hello" in user_msg  # episode message content


# -- Crossover tests ----------------------------------------------------------


def test_crossover_produces_valid_candidate():
    """Mock LLM returns valid JSON -> generation = max(a, b) + 1."""
    mock_response = json.dumps({"revised_prompt": "Hybrid prompt"})
    llm = MockLLMClient(responses=[mock_response])
    evolver = PromptEvolver(llm=llm)
    a = PromptCandidate(id="pc-a", text="Prompt A", generation=2)
    b = PromptCandidate(id="pc-b", text="Prompt B", generation=5)

    child = evolver.crossover(a, b)

    assert child is not None
    assert child.text == "Hybrid prompt"
    assert child.generation == max(a.generation, b.generation) + 1
    assert child.generation == 6
    assert child.parent_id == a.id  # primary parent is a


def test_crossover_returns_none_on_invalid_json():
    """Garbage LLM output -> None, no crash."""
    llm = MockLLMClient(responses=["not json"])
    evolver = PromptEvolver(llm=llm)
    a = PromptCandidate(id="pc-a", text="Prompt A", generation=0)
    b = PromptCandidate(id="pc-b", text="Prompt B", generation=0)

    result = evolver.crossover(a, b)

    assert result is None


def test_crossover_returns_none_on_missing_key():
    """Valid JSON but missing 'revised_prompt' -> None."""
    llm = MockLLMClient(responses=[json.dumps({"other": "data"})])
    evolver = PromptEvolver(llm=llm)
    a = PromptCandidate(id="pc-a", text="A", generation=1)
    b = PromptCandidate(id="pc-b", text="B", generation=3)

    result = evolver.crossover(a, b)

    assert result is None


def test_crossover_includes_both_candidates_in_prompt():
    """The LLM prompt must contain both candidate texts."""
    mock_response = json.dumps({"revised_prompt": "Combined"})
    llm = MockLLMClient(responses=[mock_response])
    evolver = PromptEvolver(llm=llm)
    a = PromptCandidate(id="pc-a", text="Alpha prompt text", generation=0)
    b = PromptCandidate(id="pc-b", text="Beta prompt text", generation=0)

    evolver.crossover(a, b)

    assert len(llm.call_log) == 1
    messages, _kwargs = llm.call_log[0]
    user_msg = messages[1]["content"]
    assert "Alpha prompt text" in user_msg
    assert "Beta prompt text" in user_msg


def test_crossover_equal_generations():
    """When both parents have the same generation, child = that + 1."""
    mock_response = json.dumps({"revised_prompt": "Same gen child"})
    llm = MockLLMClient(responses=[mock_response])
    evolver = PromptEvolver(llm=llm)
    a = PromptCandidate(id="pc-a", text="A", generation=3)
    b = PromptCandidate(id="pc-b", text="B", generation=3)

    child = evolver.crossover(a, b)

    assert child is not None
    assert child.generation == 4
