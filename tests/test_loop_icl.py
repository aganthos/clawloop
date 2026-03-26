"""Tests for adaptive intensity and paradigm breakthrough wiring in the learning loop."""

import json

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.intensity import AdaptiveIntensity
from clawloop.core.loop import AgentState, learning_loop
from clawloop.core.paradigm import ParadigmBreakthrough
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.core.types import Datum, FBResult, Future, OptimResult
from clawloop.layers.harness import Harness, Insight, PlaybookEntry


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
        # Use failure reward so episodes go to harness (support-query separation)
        adapter = _MockAdapter(reward=0.2)
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


class TestPerSampleReflection:
    """Per-sample reflection (batch_size=1) calls reflector once per episode."""

    def test_per_sample_calls_reflector_per_episode(self) -> None:
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig(reflection_batch_size=1))
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )
        # Use failure reward so episodes go to harness (support)
        adapter = _MockAdapter(reward=0.2)
        state = AgentState(harness=harness)

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2", "t3"],
            n_episodes=3,
            n_iterations=1,
        )

        # With batch_size=1, reflector should be called once per support episode
        assert len(client.call_log) == 3, (
            f"Expected 3 reflector calls (one per episode), got {len(client.call_log)}"
        )

    def test_batch_reflection_calls_reflector_once(self) -> None:
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig(reflection_batch_size=5))
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )
        adapter = _MockAdapter(reward=0.2)
        state = AgentState(harness=harness)

        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2", "t3"],
            n_episodes=3,
            n_iterations=1,
        )

        # With batch_size=5, all 3 episodes fit in one batch
        assert len(client.call_log) == 1, (
            f"Expected 1 reflector call (one batch), got {len(client.call_log)}"
        )

    def test_per_sample_auto_tags_insights(self) -> None:
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig(reflection_batch_size=1))
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )

        ep = _make_episode(bench="entropic", task_id="t1", reward=-0.5)
        ep.metadata = {"entropic_category": "knowledge_qa"}

        from clawloop.core.types import Datum
        harness.forward_backward(Datum(episodes=[ep]))

        # Insights should be auto-tagged with bench + category
        for insight in harness._pending.insights:
            assert "entropic" in insight.tags, f"Missing bench tag: {insight.tags}"
            assert "knowledge_qa" in insight.tags, f"Missing category tag: {insight.tags}"


class TestSelectivePlaybookRetrieval:
    """Playbook.render(tags=...) filters entries by tag (ACE/DC-RS style)."""

    def test_render_filters_by_tag(self):
        from clawloop.layers.harness import Playbook, PlaybookEntry
        pb = Playbook(entries=[
            PlaybookEntry(id="e1", content="Refuse confidential info", tags=["confidential_company_knowledge"]),
            PlaybookEntry(id="e2", content="Check data access", tags=["handle_time"]),
            PlaybookEntry(id="e3", content="General strategy", tags=["general"]),
        ])
        rendered = pb.render(tags={"handle_time"})
        assert "Check data access" in rendered
        assert "Refuse confidential" not in rendered
        assert "General strategy" not in rendered

    def test_render_no_match_falls_back_to_all(self):
        from clawloop.layers.harness import Playbook, PlaybookEntry
        pb = Playbook(entries=[
            PlaybookEntry(id="e1", content="Entry one", tags=["alpha"]),
            PlaybookEntry(id="e2", content="Entry two", tags=["beta"]),
        ])
        rendered = pb.render(tags={"nonexistent"})
        assert "Entry one" in rendered
        assert "Entry two" in rendered

    def test_render_no_tags_returns_all(self):
        from clawloop.layers.harness import Playbook, PlaybookEntry
        pb = Playbook(entries=[
            PlaybookEntry(id="e1", content="Entry one", tags=["alpha"]),
            PlaybookEntry(id="e2", content="Entry two", tags=["beta"]),
        ])
        rendered = pb.render(tags=None)
        assert "Entry one" in rendered
        assert "Entry two" in rendered

    def test_system_prompt_passes_tags(self):
        from clawloop.layers.harness import Playbook, PlaybookEntry
        harness = Harness(system_prompts={"test": "Base prompt."})
        harness.playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="Privacy rule", tags=["confidential_company_knowledge"]),
            PlaybookEntry(id="e2", content="Handle time rule", tags=["handle_time"]),
        ])
        prompt = harness.system_prompt("test", task_tags={"handle_time"})
        assert "Handle time rule" in prompt
        assert "Privacy rule" not in prompt


class TestLoopWithAdaptiveIntensity:
    """Set up with intensity(reflect_every_n=2), run 4 iterations.
    Verify reflector was called fewer than 4 times.
    """

    def test_loop_with_adaptive_intensity(self) -> None:
        client = _MockLLMClient()
        # Use batch reflection so call count reflects iteration gating, not per-episode splits
        reflector = Reflector(client=client, config=ReflectorConfig(reflection_batch_size=5))
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )
        intensity = AdaptiveIntensity(reflect_every_n=2)
        # Use failure reward so episodes go to harness (support-query separation)
        adapter = _MockAdapter(reward=0.2)
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


class TestCrossLayerRollback:
    """When one layer's optim_step fails, all layers should rollback."""

    def test_optim_failure_rolls_back_all_layers(self) -> None:
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )
        state = AgentState(harness=harness)

        # Capture harness and router state before learning
        harness_before = json.dumps(state.harness.to_dict(), sort_keys=True)
        router_before = json.dumps(state.router.to_dict(), sort_keys=True)

        # Make router.optim_step fail after harness succeeds
        def failing_router_optim():
            raise RuntimeError("simulated optim failure")

        state.router.optim_step = failing_router_optim

        adapter = _MockAdapter(reward=0.8)
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness", "router"],
        )

        # Harness should be rolled back to pre-optim state
        harness_after = json.dumps(state.harness.to_dict(), sort_keys=True)
        assert harness_after == harness_before, (
            "Harness should be rolled back when router optim fails"
        )

        # Router should also be rolled back to pre-optim state
        router_after = json.dumps(state.router.to_dict(), sort_keys=True)
        assert router_after == router_before, (
            "Router should be rolled back when its own optim fails"
        )

    def test_optim_error_status_triggers_rollback(self) -> None:
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )
        state = AgentState(harness=harness)

        # Capture harness state before learning
        harness_before = json.dumps(state.harness.to_dict(), sort_keys=True)

        # Patch router.optim_step to return an error status (not raise)
        def error_status_router_optim():
            return Future.immediate(OptimResult(status="error", updates_applied=0))

        state.router.optim_step = error_status_router_optim

        adapter = _MockAdapter(reward=0.8)
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness", "router"],
        )

        # Harness should be rolled back when router optim returns error status
        harness_after = json.dumps(state.harness.to_dict(), sort_keys=True)
        assert harness_after == harness_before, (
            "Harness should be rolled back when router optim_step returns error status"
        )

    def test_fb_error_clears_pending_state(self) -> None:
        state = AgentState()

        # Track whether clear_pending_state was called
        clear_called: list[bool] = []
        original_clear = state.router.clear_pending_state

        def tracking_clear():
            clear_called.append(True)
            return original_clear()

        state.router.clear_pending_state = tracking_clear

        # Patch router.forward_backward to return an error FBResult
        def failing_fb(data):
            return Future.immediate(FBResult(status="error"))

        state.router.forward_backward = failing_fb

        adapter = _MockAdapter(reward=0.8)
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["router"],
        )

        assert len(clear_called) > 0, (
            "clear_pending_state should be called when forward_backward returns error"
        )
