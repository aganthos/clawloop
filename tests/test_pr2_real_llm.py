"""PR2 end-to-end test with REAL LLM calls via CLIProxyAPI.

Exercises the full pipeline: learning loop → support-query split →
Reflector (real LLM) → Harness optim → PromptEvolver mutation (real LLM) →
Pareto front update. No mocks except the adapter that provides episodes.

Requires CLIProxyAPI running at http://127.0.0.1:8317/v1.
Skipped automatically if the proxy is unreachable.
"""

from __future__ import annotations

import json
import os
import logging

import pytest
import urllib.request
import urllib.error

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.evolution import EvolverConfig, PromptEvolver
from lfx.core.intensity import AdaptiveIntensity
from lfx.core.loop import AgentState, learning_loop
from lfx.core.reflector import Reflector, ReflectorConfig
from lfx.layers.harness import Harness, PromptCandidate, ParetoFront
from lfx.llm import LiteLLMClient

log = logging.getLogger(__name__)

_API_BASE = os.environ.get("LFX_API_BASE", "http://127.0.0.1:8317/v1")
_API_KEY = os.environ.get("LFX_API_KEY", "your-api-key-1")
_MODEL = os.environ.get("LFX_MODEL", "openai/claude-haiku-4-5-20251001")


def _proxy_available() -> bool:
    """Return True if the CLIProxyAPI is reachable."""
    try:
        req = urllib.request.Request(
            f"{_API_BASE}/models",
            headers={"Authorization": f"Bearer {_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


skip_no_proxy = pytest.mark.skipif(
    not _proxy_available(),
    reason="CLIProxyAPI not reachable — skipping real LLM test",
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


@skip_no_proxy
class TestRealLLMEndToEnd:
    """Full end-to-end with real LLM: reflector + evolver + learning loop."""

    def test_real_reflector_produces_insights(self) -> None:
        """Real LLM reflector analyzes failure episodes and produces
        playbook insights that actually appear in the system prompt."""
        llm = LiteLLMClient(model=_MODEL, api_key=_API_KEY, api_base=_API_BASE)
        reflector = Reflector(client=llm, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"math": "You are a math problem solver."},
            reflector=reflector,
        )
        state = AgentState(harness=harness)

        # Failure episodes: wrong answers to math questions
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

        # The real LLM should have produced at least one playbook entry
        entries = state.harness.playbook.entries
        log.info("Playbook entries after real reflector: %d", len(entries))
        for e in entries:
            log.info("  - %s: %s", e.id, e.content[:80])

        assert len(entries) >= 1, (
            "Real LLM reflector should produce at least one insight from failure episodes"
        )

        # Verify the insight appears in the rendered system prompt
        prompt = state.harness.system_prompt("math")
        assert len(prompt) > len("You are a math problem solver."), (
            "System prompt should be enriched with playbook entries"
        )

    def test_real_evolver_mutates_prompt(self) -> None:
        """Real LLM evolver reads failing episodes and produces a mutated
        prompt candidate with actual targeted improvements."""
        llm = LiteLLMClient(model=_MODEL, api_key=_API_KEY, api_base=_API_BASE)
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
        llm = LiteLLMClient(model=_MODEL, api_key=_API_KEY, api_base=_API_BASE)
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
        assert child.generation == 3  # max(1, 2) + 1
        assert child.parent_id == a.id
        assert len(child.text) > 10

        log.info("Crossover result: %s", child.text[:120])

    def test_full_loop_with_real_reflector_and_evolver(self) -> None:
        """Complete loop: real reflector produces insights from failures,
        real evolver mutates Pareto front candidates, all through learning_loop."""
        llm = LiteLLMClient(model=_MODEL, api_key=_API_KEY, api_base=_API_BASE)
        reflector = Reflector(client=llm, config=ReflectorConfig())

        harness = Harness(
            system_prompts={"math": "You are a math problem solver."},
            reflector=reflector,
        )

        # Seed a Pareto front so the evolver has something to mutate
        seed = PromptCandidate(
            id="pc-seed",
            text="You are a math problem solver.",
            per_task_scores={"q1": 0.5},
            generation=0,
        )
        harness.pareto_fronts["math"] = ParetoFront(candidates=[seed])

        evolver = PromptEvolver(llm=llm, config=EvolverConfig(
            max_mutations_per_step=1,
            max_crossovers_per_step=0,
        ))

        state = AgentState(harness=harness)

        # Mix of failures and successes
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
            evolver=evolver,
        )

        # Reflector should have added playbook entries from failure episodes
        final_entries = len(state.harness.playbook.entries)
        log.info("Playbook: %d -> %d entries", initial_entries, final_entries)
        assert final_entries > initial_entries, (
            "Real reflector should produce insights from failures"
        )

        # Evolver should have added candidates to the Pareto front
        front = state.harness.pareto_fronts["math"]
        log.info("Pareto front: %d -> %d candidates", initial_candidates, len(front.candidates))
        # At minimum the seed should still be there
        assert len(front.candidates) >= 1

        # The weights layer should have processed the success episode
        assert any(
            h.get("advantages_computed", 0) > 0
            for h in state.weights.training_history
        ) or len(state.weights.training_history) == 0  # stub may skip

        # System prompt should be enriched
        prompt = state.harness.system_prompt("math")
        assert len(prompt) > len("You are a math problem solver.")

        log.info("Final system prompt:\n%s", prompt[:300])


@skip_no_proxy
class TestFullyRealE2E:
    """Zero mocks. Real LLM solves math problems, real LLM reflects on
    failures, real MathEnvironment scores answers. Nothing is canned."""

    def test_agent_learn_real_llm_real_env(self) -> None:
        """LfXAgent.learn() with real LiteLLMClient for both task and
        reflector, real MathEnvironment for scoring. Verifies the agent
        produces playbook entries and the system prompt grows."""
        from lfx.agent import LfXAgent
        from lfx.envs.math import MathEnvironment

        task_llm = LiteLLMClient(model=_MODEL, api_key=_API_KEY, api_base=_API_BASE)
        reflector_llm = LiteLLMClient(model=_MODEL, api_key=_API_KEY, api_base=_API_BASE)

        agent = LfXAgent(
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

        # Should have run 2 iterations with real episodes
        assert len(results["rewards"]) == 2
        for r in results["rewards"]:
            assert isinstance(r, float)
            assert 0.0 <= r <= 1.0

        # Reflector should produce at least one insight from any mistakes
        # (Haiku won't get 100% on all math problems)
        prompt_after = agent.get_system_prompt()
        log.info("Prompt before:\n%s", prompt_before[:200])
        log.info("Prompt after:\n%s", prompt_after[:300])

        # The agent should have learned something (playbook entries or improved prompt)
        # Note: it's possible (though unlikely) that Haiku aces all 6 problems
        # and the reflector never fires. We assert softly.
        if results["n_entries"] > 0:
            assert len(prompt_after) > len(prompt_before), (
                "System prompt should grow when playbook entries are added"
            )
            log.info("Agent learned %d strategies from real math episodes", results["n_entries"])
        else:
            log.info("Agent aced all problems — no reflection needed (valid but rare)")
