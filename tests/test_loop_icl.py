"""Tests for adaptive intensity and paradigm breakthrough wiring in the learning loop."""

import json

import pytest

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.intensity import AdaptiveIntensity
from lfx.core.loop import AgentState, learning_loop
from lfx.core.paradigm import ParadigmBreakthrough
from lfx.core.reflector import Reflector, ReflectorConfig
from lfx.core.types import Datum
from lfx.layers.harness import Harness, Insight, PlaybookEntry


def _make_episode(
    bench: str = "test", task_id: str = "t1", reward: float = 0.8, model: str = "haiku",
) -> Episode:
    return Episode(
        id=Episode.new_id(), state_id="deadbeef", task_id=task_id, bench=bench,
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!", model=model),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class _MockAdapter:
    """Adapter that returns canned episodes."""

    def __init__(self, reward: float = 0.8) -> None:
        self.reward = reward
        self.call_count = 0

    def run_episode(self, task, agent_state) -> Episode:
        self.call_count += 1
        return _make_episode(reward=self.reward, task_id=str(task))


class _MockLLMClient:
    """LLM client that returns a canned JSON response (one insight)."""

    def __init__(self, response: str | None = None) -> None:
        self.call_log: list[dict] = []
        self._response = response or json.dumps([
            {
                "action": "add",
                "content": "Use chain-of-thought for math problems",
                "target_entry_id": None,
                "tags": ["strategy"],
                "source_episode_ids": [],
            }
        ])

    def complete(self, messages, **kwargs) -> str:
        self.call_log.append({"messages": messages, **kwargs})
        return self._response


class TestLoopCallsReflectorViaHarness:
    """Set up Harness with a Reflector, run loop for 1 iteration.
    Verify reflector was called and playbook has entries.
    """

    def test_loop_calls_reflector_via_harness(self) -> None:
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )
        adapter = _MockAdapter(reward=0.8)
        state = AgentState(harness=harness)

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2"],
            n_episodes=2,
            n_iterations=1,
        )

        # Reflector was called at least once
        assert len(client.call_log) > 0, "Reflector LLM client was never called"
        # Playbook should have at least one entry from the insight
        assert len(state.harness.playbook.entries) > 0, "No playbook entries after reflector"


class TestLoopWithAdaptiveIntensity:
    """Set up with intensity(reflect_every_n=2), run 4 iterations.
    Verify reflector was called fewer than 4 times.
    """

    def test_loop_with_adaptive_intensity(self) -> None:
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )
        intensity = AdaptiveIntensity(reflect_every_n=2)
        adapter = _MockAdapter(reward=0.8)
        state = AgentState(harness=harness)

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2"],
            n_episodes=2,
            n_iterations=4,
            intensity=intensity,
        )

        # With reflect_every_n=2:
        # iter 0: should_reflect(0) -> True (first iteration)
        # iter 1: should_reflect(1) -> True (fewer than 2 rewards recorded when checked,
        #         but after record_reward we have 2)
        # iter 2: should_reflect(2) -> True (2 % 2 == 0)
        # iter 3: should_reflect(3) -> depends on stagnation
        # In any case, with gating some iterations should skip the harness fb.
        # The reflector should have been called fewer than 4 times.
        reflector_calls = len(client.call_log)
        assert reflector_calls < 4, (
            f"Expected fewer than 4 reflector calls with intensity gating, got {reflector_calls}"
        )
        assert reflector_calls > 0, "Reflector should have been called at least once"


class TestLoopWithoutReflectorStillWorks:
    """No reflector, no intensity. Loop runs normally (backward compat)."""

    def test_loop_without_reflector_still_works(self) -> None:
        adapter = _MockAdapter(reward=0.7)
        state = AgentState()

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2", "t3"],
            n_episodes=3,
            n_iterations=2,
        )

        assert adapter.call_count == 6
        assert sid.combined_hash
