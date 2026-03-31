"""End-to-end test with REAL LLM calls via Gemini.

Exercises the full pipeline: learning loop → support-query split →
Reflector (real LLM) → Harness optim → PromptEvolver mutation (real LLM) →
Pareto front update. No mocks except the adapter that provides episodes.

Requires GOOGLE_API_KEY environment variable.
Skipped automatically if the key is not set.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import pytest

from clawloop.core.env import Sample, TaskEnvironment
from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.evolution import EvolverConfig, PromptEvolver
from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.loop import AgentState, learning_loop
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.harness_backends.local import LocalEvolver
from clawloop.learning_layers.harness import Harness, PromptCandidate, ParetoFront
from clawloop.llm import LiteLLMClient

log = logging.getLogger(__name__)

_MODEL = os.environ.get("CLAWLOOP_MODEL", "gemini/gemini-2.0-flash-lite")

skip_no_key = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping real LLM test",
)


def _make_episode(task_id: str, reward: float, question: str, answer: str) -> Episode:
    return Episode(
        id=Episode.new_id(), state_id="real-llm-test", task_id=task_id, bench="math",
        messages=[
            Message(role="system", content="You are a math problem solver."),
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=200.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class _FixedAdapter:
    """Adapter that returns pre-built episodes."""
    def __init__(self, episodes: list[Episode]) -> None:
        self._episodes = episodes
        self._idx = 0

    def run_episode(self, task, agent_state) -> Episode:
        ep = self._episodes[self._idx % len(self._episodes)]
        self._idx += 1
        return ep


@skip_no_key
class TestRealLLMEndToEnd:
    """Full end-to-end with real LLM: reflector + evolver + learning loop."""

    def test_real_reflector_produces_insights(self) -> None:
        """Real LLM reflector analyzes failure episodes and produces
        playbook insights that actually appear in the system prompt."""
        llm = LiteLLMClient(model=_MODEL)
        reflector = Reflector(client=llm, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"math": "You are a math problem solver."},
            evolver=LocalEvolver(reflector=reflector),
        )
        state = AgentState(harness=harness)

        episodes = [
            _make_episode("q1", reward=0.1, question="What is 17 + 28?", answer="The answer is 43"),
            _make_episode("q2", reward=0.1, question="What is 15 * 13?", answer="The answer is 165"),
            _make_episode("q3", reward=0.0, question="What is 144 / 12?", answer="The answer is 14"),
        ]
        adapter = _FixedAdapter(episodes)

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2", "t3"],
            n_episodes=3,
            n_iterations=1,
            active_layers=["harness"],
        )

        entries = state.harness.playbook.entries
        log.info("Playbook entries after real reflector: %d", len(entries))
        for e in entries:
            log.info("  - %s: %s", e.id, e.content[:80])

        assert len(entries) >= 1, (
            "Real LLM reflector should produce at least one insight from failure episodes"
        )

        prompt = state.harness.system_prompt("math")
        assert len(prompt) > len("You are a math problem solver."), (
            "System prompt should be enriched with playbook entries"
        )

    def test_real_evolver_mutates_prompt(self) -> None:
        """Real LLM evolver reads failing episodes and produces a mutated
        prompt candidate with actual targeted improvements."""
        llm = LiteLLMClient(model=_MODEL)
        evolver = PromptEvolver(llm=llm, config=EvolverConfig())

        parent = PromptCandidate(
            id="pc-parent",
            text="You are a math problem solver. Answer concisely.",
            per_task_scores={"q1": 0.3, "q2": 0.2},
            generation=0,
        )

        failures = [
            _make_episode("q1", reward=0.1, question="What is 17 + 28?", answer="43"),
            _make_episode("q2", reward=0.0, question="Solve: 3x + 7 = 22", answer="x = 3"),
        ]

        child = evolver.mutate(parent, failures)

        assert child is not None, "Real LLM evolver should produce a mutation"
        assert child.parent_id == parent.id
        assert child.generation == parent.generation + 1
        assert child.text != parent.text, "Mutated prompt should differ from parent"
        assert len(child.text) > 10, "Mutated prompt should be substantive"

        log.info("Mutation result: %s", child.text[:120])

    def test_real_evolver_crossover(self) -> None:
        """Real LLM evolver combines two candidates into a hybrid."""
        llm = LiteLLMClient(model=_MODEL)
        evolver = PromptEvolver(llm=llm, config=EvolverConfig())

        a = PromptCandidate(
            id="pc-a",
            text="You are a math solver. Always show step-by-step work.",
            per_task_scores={"arithmetic": 0.9, "algebra": 0.3},
            generation=1,
        )
        b = PromptCandidate(
            id="pc-b",
            text="You are a math solver. For algebra, isolate the variable first.",
            per_task_scores={"arithmetic": 0.4, "algebra": 0.8},
            generation=2,
        )

        child = evolver.crossover(a, b)

        assert child is not None, "Real LLM evolver should produce a crossover"
        assert child.generation == 3
        assert child.parent_id == a.id
        assert len(child.text) > 10

        log.info("Crossover result: %s", child.text[:120])

    def test_full_loop_with_real_reflector_and_evolver(self) -> None:
        """Complete loop: real reflector produces insights from failures,
        real evolver mutates Pareto front candidates, all through learning_loop."""
        llm = LiteLLMClient(model=_MODEL)
        reflector = Reflector(client=llm, config=ReflectorConfig())

        evolver = PromptEvolver(llm=llm, config=EvolverConfig(
            max_mutations_per_step=1,
            max_crossovers_per_step=0,
        ))

        harness = Harness(
            system_prompts={"math": "You are a math problem solver."},
            evolver=LocalEvolver(reflector=reflector, prompt_evolver=evolver),
        )

        seed = PromptCandidate(
            id="pc-seed",
            text="You are a math problem solver.",
            per_task_scores={"q1": 0.5},
            generation=0,
        )
        harness.pareto_fronts["math"] = ParetoFront(candidates=[seed])

        state = AgentState(harness=harness)

        episodes = [
            _make_episode("q1", reward=0.1, question="What is 17 + 28?", answer="43"),
            _make_episode("q2", reward=0.0, question="What is 15 * 13?", answer="165"),
            _make_episode("q3", reward=0.9, question="What is 2 + 2?", answer="4"),
        ]
        adapter = _FixedAdapter(episodes)

        initial_entries = len(harness.playbook.entries)
        initial_candidates = len(harness.pareto_fronts["math"].candidates)

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2", "t3"],
            n_episodes=3,
            n_iterations=1,
        )

        final_entries = len(state.harness.playbook.entries)
        log.info("Playbook: %d -> %d entries", initial_entries, final_entries)
        assert final_entries > initial_entries, (
            "Real reflector should produce insights from failures"
        )

        front = state.harness.pareto_fronts["math"]
        log.info("Pareto front: %d -> %d candidates", initial_candidates, len(front.candidates))
        assert len(front.candidates) >= 1

        prompt = state.harness.system_prompt("math")
        assert len(prompt) > len("You are a math problem solver.")


@skip_no_key
class TestFullyRealE2E:
    """Zero mocks. Real LLM solves math problems, real LLM reflects on
    failures, real MathEnvironment scores answers. Nothing is canned."""

    def test_agent_learn_real_llm_real_env(self) -> None:
        """ClawLoopAgent.learn() with real LiteLLMClient for both task and
        reflector, real MathEnvironment for scoring."""
        from clawloop.agent import ClawLoopAgent
        from clawloop.environments.math import MathEnvironment

        task_llm = LiteLLMClient(model=_MODEL)
        reflector_llm = LiteLLMClient(model=_MODEL)

        agent = ClawLoopAgent(
            task_client=task_llm,
            reflector_client=reflector_llm,
            bench="math",
            base_system_prompt="You are a math problem solver. Answer with just the number.",
        )

        env = MathEnvironment()

        prompt_before = agent.get_system_prompt()
        results = agent.learn(env, iterations=2, episodes_per_iter=3)

        log.info("Iteration rewards: %s", results["rewards"])
        log.info("Playbook entries: %d", results["n_entries"])

        assert len(results["rewards"]) == 2
        for r in results["rewards"]:
            assert isinstance(r, float)
            assert 0.0 <= r <= 1.0

        prompt_after = agent.get_system_prompt()
        if results["n_entries"] > 0:
            assert len(prompt_after) > len(prompt_before), (
                "System prompt should grow when playbook entries are added"
            )
            log.info("Agent learned %d strategies from real math episodes", results["n_entries"])
        else:
            log.info("Agent aced all problems — no reflection needed (valid but rare)")


class _RealMathAdapter:
    """Adapter that calls a real LLM to solve math problems and scores
    via MathEnvironment. No mocks anywhere."""

    def __init__(self, llm: Any, env: Any, bench: str = "math") -> None:
        self._llm = llm
        self._env = env
        self._bench = bench

    def run_episode(self, task: Sample, agent_state: Any) -> Episode:
        system_prompt = agent_state.harness.system_prompt(self._bench)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task.question},
        ]

        response = self._llm.complete(messages)
        response_text = response.text if hasattr(response, "text") else str(response)
        eval_result = self._env.evaluate(task, response_text)

        task_id = hashlib.sha256(
            f"{self._bench}:{task.question}".encode()
        ).hexdigest()[:16]

        ep_messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=task.question),
            Message(role="assistant", content=response_text),
        ]

        return Episode(
            id=Episode.new_id(),
            state_id="real-e2e",
            task_id=task_id,
            bench=self._bench,
            messages=ep_messages,
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=eval_result.score, done=True, timing_ms=0.0)],
            summary=EpisodeSummary(total_reward=eval_result.score),
        )


@skip_no_key
class TestFullPipelineRealLLM:
    """Full learning_loop() with real LLM, real MathEnvironment, real
    Reflector, real PromptEvolver, support-query separation, and GEPA.
    Zero mocks. Everything goes through the actual code paths."""

    def test_full_learning_loop_real_everything(self) -> None:
        from clawloop.environments.math import MathEnvironment

        llm = LiteLLMClient(model=_MODEL)
        env = MathEnvironment()
        tasks = env.get_tasks()

        reflector = Reflector(client=llm, config=ReflectorConfig())

        evolver = PromptEvolver(llm=llm, config=EvolverConfig(
            max_mutations_per_step=1,
            max_crossovers_per_step=0,
        ))

        harness = Harness(
            system_prompts={"math": "You are a math problem solver. Answer with just the number."},
            evolver=LocalEvolver(reflector=reflector, prompt_evolver=evolver),
        )
        seed = PromptCandidate(
            id="pc-seed",
            text="You are a math problem solver. Answer with just the number.",
            per_task_scores={},
            generation=0,
        )
        harness.pareto_fronts["math"] = ParetoFront(candidates=[seed])

        intensity = AdaptiveIntensity(cooldown_after_request=0.0)

        adapter = _RealMathAdapter(llm=llm, env=env, bench="math")

        state = AgentState(harness=harness)

        initial_entries = len(harness.playbook.entries)

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=tasks,
            n_episodes=4,
            n_iterations=2,
            intensity=intensity,
        )

        log.info("State ID: %s", sid.combined_hash[:12])
        log.info("Playbook entries: %d", len(state.harness.playbook.entries))
        log.info("Pareto candidates: %d", len(state.harness.pareto_fronts["math"].candidates))
        log.info("Weights history: %d", len(state.weights.training_history))

        prompt = state.harness.system_prompt("math")
        log.info("Final prompt:\n%s", prompt[:400])

        assert sid.combined_hash, "Should produce a valid state ID"

        front = state.harness.pareto_fronts["math"]
        assert len(front.candidates) >= 1

        final_entries = len(state.harness.playbook.entries)
        weights_history = len(state.weights.training_history)

        log.info(
            "Results: entries=%d->%d, weights_steps=%d, pareto=%d",
            initial_entries, final_entries,
            weights_history, len(front.candidates),
        )

        assert final_entries > initial_entries or weights_history > 0, (
            "Either harness should learn from failures or weights from successes"
        )
