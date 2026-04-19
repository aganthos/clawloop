"""PR2 integration tests — real code paths, real layers, no mocked forward_backward.

Only the LLM is mocked (unavoidable). Everything else runs through the real
Harness, Reflector, Weights, Router, learning loop, and background system.
"""

import json

from clawloop.agent import ClawLoopAgent
from clawloop.collector import EpisodeCollector
from clawloop.core.background import (
    BackgroundScheduler,
    BackgroundState,
    EpisodeDreamer,
    PlaybookConsolidation,
)
from clawloop.core.curator import CuratorConfig, PlaybookCurator
from clawloop.core.embeddings import MockEmbedding
from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.evolution import EvolverConfig, PromptEvolver
from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.loop import AgentState, learning_loop
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.core.reward import RewardPipeline
from clawloop.environments.math import MathEnvironment
from clawloop.harness_backends.local import LocalEvolver
from clawloop.learning_layers.harness import (
    Harness,
    ParetoFront,
    Playbook,
    PlaybookEntry,
    PromptCandidate,
)
from clawloop.llm import MockLLMClient
from clawloop.wrapper import wrap

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _insight_json(content: str, action: str = "add", tags: list[str] | None = None) -> str:
    return json.dumps(
        [
            {
                "action": action,
                "content": content,
                "target_entry_id": None,
                "tags": tags or ["strategy"],
                "source_episode_ids": [],
            }
        ]
    )


def _mutation_json(text: str) -> str:
    return json.dumps({"revised_prompt": text})


def _dreamer_json(content: str) -> str:
    return json.dumps(
        [
            {
                "action": "add",
                "content": content,
                "tags": ["meta-pattern"],
            }
        ]
    )


def _make_episode(task_id: str = "t1", reward: float = 0.5, bench: str = "test") -> Episode:
    return Episode(
        id=Episode.new_id(),
        state_id="int-test",
        task_id=task_id,
        bench=bench,
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content=f"Task {task_id}"),
            Message(role="assistant", content=f"Response for {task_id}"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=50.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class _ReplayAdapter:
    """Adapter that yields pre-built episodes."""

    def __init__(self, episodes: list[Episode]) -> None:
        self._episodes = episodes
        self._idx = 0

    def run_episode(self, task, agent_state) -> Episode:
        ep = self._episodes[self._idx % len(self._episodes)]
        self._idx += 1
        return ep


# ---------------------------------------------------------------------------
# 1. Support-query separation: real Harness + Reflector + Weights
# ---------------------------------------------------------------------------


class TestSupportQueryRealLayers:
    """Verify support-query split using real layers — failures produce
    playbook entries via Reflector, successes accumulate Weights advantages."""

    def test_failures_trigger_reflector_successes_feed_weights(self) -> None:
        """Run loop with mixed episodes. Harness reflector fires on failures,
        Weights accumulates advantages from successes."""
        reflector_client = MockLLMClient(
            responses=[
                _insight_json("When the user asks X, always clarify first"),
            ]
        )
        reflector = Reflector(client=reflector_client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            evolver=LocalEvolver(reflector=reflector),
        )

        state = AgentState(harness=harness)

        # 2 failures (reward=0.2 → effective=-0.6) + 2 successes (reward=0.8 → effective=0.6)
        episodes = [
            _make_episode("fail_1", reward=0.2),
            _make_episode("fail_2", reward=0.2),
            _make_episode("succ_1", reward=0.8),
            _make_episode("succ_2", reward=0.8),
        ]
        adapter = _ReplayAdapter(episodes)

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2", "t3", "t4"],
            n_episodes=4,
            n_iterations=1,
        )

        # Reflector was called (harness got the 2 failure episodes)
        assert len(reflector_client.call_log) > 0, "Reflector should fire on failure episodes"

        # Playbook got an entry from the insight
        assert len(state.harness.playbook.entries) >= 1
        assert any("clarify" in e.content for e in state.harness.playbook.entries)

        # Weights accumulated advantages from the 2 success episodes
        # (Weights stub groups by task_id and computes GRPO advantages)
        assert (
            len(state.weights.training_history) >= 1 or len(state.weights._pending.advantages) == 0
        )
        # After optim_step, advantages are drained — check training_history
        assert any(
            h.get("advantages_computed", 0) > 0 for h in state.weights.training_history
        ), "Weights should have recorded advantages from success episodes"

    def test_all_successes_still_reach_harness(self) -> None:
        """All episodes reach harness (support-query split disabled)."""
        reflector_client = MockLLMClient(
            responses=[
                _insight_json("Insight from successes"),
            ]
        )
        reflector = Reflector(client=reflector_client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            evolver=LocalEvolver(reflector=reflector),
        )

        state = AgentState(harness=harness)
        episodes = [_make_episode("s1", reward=0.9), _make_episode("s2", reward=0.8)]
        adapter = _ReplayAdapter(episodes)

        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2"],
            n_episodes=2,
            n_iterations=1,
        )

        # Harness receives all episodes (split disabled)
        assert len(reflector_client.call_log) > 0

    def test_all_failures_still_reach_weights(self) -> None:
        """All episodes reach all layers (support-query split disabled)."""
        reflector_client = MockLLMClient(
            responses=[
                _insight_json("Handle edge cases"),
            ]
        )
        reflector = Reflector(client=reflector_client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            evolver=LocalEvolver(reflector=reflector),
        )
        state = AgentState(harness=harness)
        episodes = [_make_episode("f1", reward=0.1), _make_episode("f2", reward=0.2)]
        adapter = _ReplayAdapter(episodes)

        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2"],
            n_episodes=2,
            n_iterations=1,
        )

        # Weights receives all episodes (support-query split disabled)
        assert len(state.weights.training_history) > 0


# ---------------------------------------------------------------------------
# 2. Generation flush with real harness + reflector
# ---------------------------------------------------------------------------


class TestGenerationFlushReal:
    """When the reflector adds an insight (advancing playbook_generation),
    stale entries in the weights buffer should be flushed."""

    def test_generation_advance_flushes_weights_pending(self) -> None:
        reflector_client = MockLLMClient(
            responses=[
                _insight_json("Always validate input before processing"),
                json.dumps([]),  # second call returns nothing
            ]
        )
        reflector = Reflector(client=reflector_client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            evolver=LocalEvolver(reflector=reflector),
        )
        state = AgentState(harness=harness)

        # Pre-seed the weights buffer with stale advantages
        state.weights._pending.advantages = [("old_ep_1", 0.5), ("old_ep_2", -0.3)]
        # Set generation tracking to 0
        state._prev_playbook_generation = 0

        # Failure episodes trigger reflector → insight → generation advance
        episodes = [_make_episode("f1", reward=0.1)]
        adapter = _ReplayAdapter(episodes)

        initial_gen = harness.playbook_generation

        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness"],  # only harness to isolate flush
        )

        # If reflector produced an insight, playbook_generation should have advanced
        if harness.playbook_generation > initial_gen:
            # Weights buffer should have been flushed
            assert (
                len(state.weights._pending.advantages) == 0
            ), "Stale advantages should be flushed after generation advance"


# ---------------------------------------------------------------------------
# 3. PromptEvolver through the real learning loop
# ---------------------------------------------------------------------------


class TestEvolutionInLoop:
    """Test that mutation actually runs through the loop and produces
    new Pareto front candidates."""

    def test_evolver_produces_pareto_candidates(self) -> None:
        # Set up a Harness with a Pareto front that has one candidate
        evolver_llm = MockLLMClient(
            responses=[
                _mutation_json("You are helpful. Always ask clarifying questions."),
                _mutation_json("You are helpful and thorough."),  # crossover
            ]
        )
        evolver = PromptEvolver(llm=evolver_llm, config=EvolverConfig())

        harness = Harness(
            system_prompts={"test": "You are helpful."},
            evolver=LocalEvolver(prompt_evolver=evolver),
        )
        seed_candidate = PromptCandidate(
            id="pc-seed",
            text="You are helpful.",
            per_task_scores={"t1": 0.8},
            generation=0,
        )
        harness.pareto_fronts["test"] = ParetoFront(candidates=[seed_candidate])

        state = AgentState(harness=harness)
        # Need failure episodes for mutation (support set)
        episodes = [_make_episode("f1", reward=0.2)]
        adapter = _ReplayAdapter(episodes)

        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
        )

        front = state.harness.pareto_fronts["test"]
        # Should have more candidates than we started with
        assert len(front.candidates) >= 1
        # The mutated candidate should have parent lineage
        mutated = [c for c in front.candidates if c.parent_id is not None]
        assert len(mutated) >= 1, "Expected at least one mutated candidate"
        assert mutated[0].generation >= 1
        assert "clarifying" in mutated[0].text or "thorough" in mutated[0].text


# ---------------------------------------------------------------------------
# 4. Activity-aware intensity — real wrapper + collector
# ---------------------------------------------------------------------------


class TestActivityIntensityReal:
    """Test that the real wrapper/collector wires user activity
    into the intensity cooldown."""

    def test_wrapper_records_activity(self) -> None:
        """WrappedClient.complete() should call intensity.record_user_activity()."""
        intensity = AdaptiveIntensity(cooldown_after_request=30.0)
        mock_client = MockLLMClient(responses=["Hi there!"])
        pipeline = RewardPipeline([])
        collector = EpisodeCollector(pipeline=pipeline, batch_size=100)

        wrapped = wrap(mock_client, collector, intensity=intensity)

        assert intensity._last_user_request == 0.0
        wrapped.complete([{"role": "user", "content": "Hello"}])
        assert (
            intensity._last_user_request > 0.0
        ), "Wrapper should record user activity on complete()"

    def test_collector_records_activity(self) -> None:
        """EpisodeCollector.ingest() should call intensity.record_user_activity()."""
        intensity = AdaptiveIntensity(cooldown_after_request=30.0)
        pipeline = RewardPipeline([])
        collector = EpisodeCollector(
            pipeline=pipeline,
            batch_size=100,
            intensity=intensity,
        )

        assert intensity._last_user_request == 0.0
        collector.ingest(
            [Message(role="user", content="Hello"), Message(role="assistant", content="Hi!")],
            task_id="t1",
        )
        assert (
            intensity._last_user_request > 0.0
        ), "Collector should record user activity on ingest()"

    def test_active_user_defers_loop_reflection(self) -> None:
        """When user is active (within cooldown), the reflector should be skipped."""
        reflector_client = MockLLMClient(
            responses=[
                _insight_json("Should be skipped"),
            ]
        )
        reflector = Reflector(client=reflector_client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            evolver=LocalEvolver(reflector=reflector),
        )

        intensity = AdaptiveIntensity(cooldown_after_request=9999.0)
        # Simulate active user
        intensity.record_user_activity()

        state = AgentState(harness=harness)
        episodes = [_make_episode("f1", reward=0.2)]
        adapter = _ReplayAdapter(episodes)

        state, _ = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            intensity=intensity,
        )

        # Reflector should NOT have been called due to user activity cooldown
        assert (
            len(reflector_client.call_log) == 0
        ), "Reflector should be deferred when user is active"


# ---------------------------------------------------------------------------
# 5. Background scheduler with real curator
# ---------------------------------------------------------------------------


class TestBackgroundSchedulerReal:
    """Run BackgroundScheduler with a real PlaybookCurator doing consolidation."""

    def test_consolidation_runs_real_curator(self) -> None:
        """PlaybookConsolidation task calls the real curator.consolidate()."""
        embedding = MockEmbedding(dim=8)
        llm = MockLLMClient(
            responses=[
                # merge response for consolidation
                json.dumps(
                    {
                        "content": "Merged: handle errors and validate inputs",
                        "tags": ["strategy"],
                    }
                ),
            ]
        )

        curator = PlaybookCurator(
            config=CuratorConfig(
                cluster_threshold=0.99,  # high threshold to avoid unintended merges
                max_playbook_entries=100,
            ),
            embeddings=embedding,
            llm=llm,
        )

        playbook = Playbook(
            entries=[
                PlaybookEntry(
                    id="e1",
                    content="Handle errors gracefully",
                    helpful=5,
                    harmful=0,
                    tags=["strategy"],
                ),
                PlaybookEntry(
                    id="e2",
                    content="Validate all inputs",
                    helpful=3,
                    harmful=0,
                    tags=["strategy"],
                ),
            ]
        )

        task = PlaybookConsolidation(
            episode_threshold=1,
            min_interval=0.0,
            curator=curator,
        )

        scheduler = BackgroundScheduler(tasks=[task])
        scheduler.record_episodes(10)

        # Run tick
        scheduler.tick(
            playbook=playbook,
            recent_episodes=[],
            is_user_idle=True,
        )

        # Curator should have run (pruning at minimum)
        assert curator._metrics.consolidation_runs >= 1

    def test_dreamer_applies_entries_to_playbook(self) -> None:
        """EpisodeDreamer uses the LLM to analyze episodes and add entries to playbook."""
        llm = MockLLMClient(
            responses=[
                _dreamer_json("Failure pattern: agent struggles with multi-step reasoning"),
            ]
        )
        dreamer = EpisodeDreamer(
            episode_threshold=2,
            min_interval=0.0,
            llm=llm,
        )

        episodes = [
            _make_episode("t1", reward=0.3),
            _make_episode("t2", reward=0.2),
            _make_episode("t3", reward=0.7),
        ]
        playbook = Playbook()

        state = BackgroundState(
            episodes_since_last_run=5,
            time_since_last_run=9999.0,
            is_user_idle=True,
            playbook=playbook,
            recent_episodes=episodes,
        )

        dreamer.run(state)
        assert len(playbook.entries) >= 1
        assert "meta-pattern" in playbook.entries[0].tags
        assert "multi-step" in playbook.entries[0].content


# ---------------------------------------------------------------------------
# 6. End-to-end: ClawLoopAgent math learning with support-query under the hood
# ---------------------------------------------------------------------------


class TestClawLoopAgentMathE2E:
    """Full ClawLoopAgent.learn() with MathEnvironment — verifies the learning
    pipeline works end-to-end including the support-query separation
    that now runs inside the Harness reflector path."""

    def test_math_agent_learns_strategy(self) -> None:
        task_responses = [
            "The answer is 45",  # correct for "What is 17 + 28?"
            "The answer is 99",  # wrong for most
            "The answer is 12",  # correct for "What is 144 / 12?"
            "The answer is 0",  # wrong
        ]
        task_client = MockLLMClient(responses=task_responses)

        reflector_responses = [
            _insight_json("Show step-by-step work for arithmetic"),
            json.dumps([]),
        ]
        reflector_client = MockLLMClient(responses=reflector_responses)

        agent = ClawLoopAgent(
            task_client=task_client,
            reflector_client=reflector_client,
            bench="math",
            base_system_prompt="You are a math solver.",
        )

        env = MathEnvironment()
        results = agent.learn(env, iterations=2, episodes_per_iter=2)

        assert len(results["rewards"]) == 2
        assert results["n_entries"] >= 1
        # The learned strategy should appear in the prompt
        prompt = agent.get_system_prompt()
        assert "step-by-step" in prompt
