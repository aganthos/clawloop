"""Tests for the ClawLoopAgent convenience wrapper."""

import json
import os

import pytest

from clawloop.agent import ClawLoopAgent
from clawloop.core.env import EvalResult, Sample, StaticTaskEnvironment
from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.layers.harness import Insight, PlaybookEntry
from clawloop.llm import MockLLMClient


def _make_env(n_tasks: int = 3, score: float = 0.8) -> StaticTaskEnvironment:
    """Create a simple static environment with fixed-score evaluation."""
    tasks = [
        Sample(question=f"Question {i}", ground_truth=f"Answer {i}")
        for i in range(n_tasks)
    ]
    return StaticTaskEnvironment(
        tasks=tasks,
        evaluate_fn=lambda s, r: EvalResult(score=score, feedback="ok"),
    )


def _reflector_response_add() -> str:
    """A canned reflector JSON response that adds a strategy."""
    return json.dumps([
        {
            "action": "add",
            "content": "Always break the problem into smaller steps",
            "target_entry_id": None,
            "tags": ["strategy"],
            "source_episode_ids": [],
        }
    ])


def _make_episode(reward: float = 0.8) -> Episode:
    """Build a minimal Episode for ingest tests."""
    return Episode(
        id=Episode.new_id(),
        state_id="deadbeef",
        task_id="t1",
        bench="default",
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="4"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=50.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class TestLearnRunsLoop:
    """MockLLMClient for both task and reflector, run 1 iteration with 2 episodes.
    Check results dict has 'rewards' with 1 entry.
    """

    def test_learn_runs_loop(self) -> None:
        task_client = MockLLMClient(responses=["The answer is 42"])
        reflector_client = MockLLMClient(responses=[_reflector_response_add()])
        env = _make_env(n_tasks=3, score=0.7)

        agent = ClawLoopAgent(
            task_client=task_client,
            reflector_client=reflector_client,
            bench="default",
            base_system_prompt="You are a helpful assistant.",
        )

        results = agent.learn(env, iterations=1, episodes_per_iter=2)

        assert "rewards" in results
        assert len(results["rewards"]) == 1
        assert isinstance(results["rewards"][0], float)
        assert "playbook" in results
        assert "n_entries" in results


class TestGetSystemPromptEmptyInitially:
    """No learning done, returns a string (may be empty or base prompt)."""

    def test_get_system_prompt_empty_initially(self) -> None:
        task_client = MockLLMClient()
        reflector_client = MockLLMClient()

        agent = ClawLoopAgent(
            task_client=task_client,
            reflector_client=reflector_client,
        )

        prompt = agent.get_system_prompt()
        assert isinstance(prompt, str)


class TestGetSystemPromptAfterLearning:
    """After 1 iteration with a reflector that returns an 'add' insight,
    prompt includes the learned strategy.
    """

    def test_get_system_prompt_after_learning(self) -> None:
        task_client = MockLLMClient(responses=["The answer is 42"])
        reflector_client = MockLLMClient(responses=[_reflector_response_add()])
        env = _make_env(n_tasks=3, score=0.8)

        agent = ClawLoopAgent(
            task_client=task_client,
            reflector_client=reflector_client,
            bench="default",
            base_system_prompt="You are a helpful assistant.",
        )

        agent.learn(env, iterations=1, episodes_per_iter=2)
        prompt = agent.get_system_prompt()

        assert "break the problem into smaller steps" in prompt


class TestIngestEpisodes:
    """Build an Episode externally, call ingest(), verify playbook gets the insight."""

    def test_ingest_episodes(self) -> None:
        task_client = MockLLMClient()
        reflector_client = MockLLMClient(responses=[_reflector_response_add()])

        agent = ClawLoopAgent(
            task_client=task_client,
            reflector_client=reflector_client,
        )

        episode = _make_episode(reward=0.9)
        agent.ingest([episode])

        # After ingest, the playbook should contain the insight from the reflector
        assert len(agent._harness.playbook.entries) > 0
        contents = [e.content for e in agent._harness.playbook.entries]
        assert any("break the problem into smaller steps" in c for c in contents)


class TestSaveLoadPlaybook:
    """Learn, save to file, create new agent, load, verify prompt includes
    the learned strategy.
    """

    def test_save_load_playbook(self, tmp_path) -> None:
        task_client = MockLLMClient(responses=["The answer is 42"])
        reflector_client = MockLLMClient(responses=[_reflector_response_add()])
        env = _make_env(n_tasks=3, score=0.8)

        agent = ClawLoopAgent(
            task_client=task_client,
            reflector_client=reflector_client,
            bench="default",
            base_system_prompt="You are a helpful assistant.",
        )

        agent.learn(env, iterations=1, episodes_per_iter=2)

        # Save
        save_path = str(tmp_path / "playbook.json")
        agent.save_playbook(save_path)
        assert os.path.exists(save_path)

        # Create a new agent and load
        agent2 = ClawLoopAgent(
            task_client=MockLLMClient(),
            reflector_client=MockLLMClient(),
            bench="default",
            base_system_prompt="You are a helpful assistant.",
        )

        agent2.load_playbook(save_path)
        prompt = agent2.get_system_prompt()

        assert "break the problem into smaller steps" in prompt
