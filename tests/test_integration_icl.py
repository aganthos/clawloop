"""Full integration test: LfXAgent + MathEnvironment + Reflector."""

import json

from lfx.agent import LfXAgent
from lfx.envs.math import MathEnvironment
from lfx.llm import MockLLMClient
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _insight_response(content: str) -> str:
    """Build a JSON reflector response that adds a single insight."""
    return json.dumps([
        {
            "action": "add",
            "content": content,
            "target_entry_id": None,
            "tags": ["strategy"],
            "source_episode_ids": [],
        }
    ])


def _empty_insight_response() -> str:
    """Build a JSON reflector response with no insights."""
    return json.dumps([])


# ---------------------------------------------------------------------------
# Test 1
# ---------------------------------------------------------------------------

class TestMathLearningLoopEndToEnd:
    """Full loop: run math tasks, reflect, improve playbook."""

    def test_math_learning_loop_end_to_end(self) -> None:
        # Task client returns varied answers. Some are correct for built-in
        # problems, some are wrong. The cycling ensures a mix of rewards.
        #
        # Built-in problems include:
        #   "What is 17 + 28?"  -> 45
        #   "What is 144 / 12?" -> 12
        #   "What is 15 * 13?"  -> 195
        #   "Solve for x: 3x + 7 = 22." -> 5
        #   "Solve for x: x^2 - 5x + 6 = 0. Give the larger root." -> 3
        #   ... and many more
        #
        # We cycle through 6 responses: some correct answers for likely-sampled
        # problems, some intentionally wrong.
        task_responses = [
            "The answer is 45",   # correct for "What is 17 + 28?"
            "The answer is 99",   # wrong for most problems
            "The answer is 12",   # correct for "What is 144 / 12?" or GCD(36,48)
            "The answer is 0",    # wrong for most problems
            "The answer is 5",    # correct for "Solve for x: 3x + 7 = 22."
            "The answer is 77",   # wrong for most problems
        ]
        task_client = MockLLMClient(responses=task_responses)

        # Reflector returns an insight on first call, empty on second call.
        reflector_responses = [
            _insight_response("For summation problems, use n(n+1)/2"),
            _empty_insight_response(),
        ]
        reflector_client = MockLLMClient(responses=reflector_responses)

        agent = LfXAgent(
            task_client=task_client,
            reflector_client=reflector_client,
            bench="math",
            base_system_prompt="You are a math problem solver.",
        )

        env = MathEnvironment()

        results = agent.learn(env, iterations=2, episodes_per_iter=2)

        # Assert: rewards list has 2 entries (one per iteration)
        assert len(results["rewards"]) == 2, (
            f"Expected 2 reward entries, got {len(results['rewards'])}"
        )
        for r in results["rewards"]:
            assert isinstance(r, float)

        # Assert: at least 1 playbook entry from the reflector insight
        assert results["n_entries"] >= 1, (
            f"Expected at least 1 playbook entry, got {results['n_entries']}"
        )

        # Assert: system prompt contains the learned strategy
        prompt = agent.get_system_prompt()
        assert "n(n+1)/2" in prompt, (
            f"Expected learned strategy 'n(n+1)/2' in system prompt, got:\n{prompt}"
        )


# ---------------------------------------------------------------------------
# Test 2
# ---------------------------------------------------------------------------

class TestSaveLoadPreservesLearning:
    """Run 1 iteration to generate a playbook entry. Save and reload."""

    def test_save_load_preserves_learning(self, tmp_path) -> None:
        task_client = MockLLMClient(responses=["The answer is 45"])
        reflector_client = MockLLMClient(responses=[
            _insight_response("For summation problems, use n(n+1)/2"),
        ])

        agent = LfXAgent(
            task_client=task_client,
            reflector_client=reflector_client,
            bench="math",
            base_system_prompt="You are a math problem solver.",
        )

        env = MathEnvironment()

        # Run 1 iteration to generate a playbook entry
        results = agent.learn(env, iterations=1, episodes_per_iter=2)
        assert results["n_entries"] >= 1, "Learning should have produced at least 1 entry"

        # Save to tmp_path
        save_path = str(tmp_path / "playbook.json")
        agent.save_playbook(save_path)

        # Create a brand-new agent and load the playbook
        agent2 = LfXAgent(
            task_client=MockLLMClient(),
            reflector_client=MockLLMClient(),
            bench="math",
            base_system_prompt="You are a math problem solver.",
        )

        # Before loading, the new agent should not have the strategy
        prompt_before = agent2.get_system_prompt()
        assert "n(n+1)/2" not in prompt_before

        agent2.load_playbook(save_path)

        # After loading, the learned strategy should be in the system prompt
        prompt_after = agent2.get_system_prompt()
        assert "n(n+1)/2" in prompt_after, (
            f"Expected 'n(n+1)/2' in loaded agent's system prompt, got:\n{prompt_after}"
        )


# ---------------------------------------------------------------------------
# Test 3
# ---------------------------------------------------------------------------

class TestIngestExternalEpisodes:
    """Create an Episode manually and ingest it via agent.ingest()."""

    def test_ingest_external_episodes(self) -> None:
        # Build an Episode manually: user asks "2+2?", assistant says "5", reward=0.0
        episode = Episode(
            id=Episode.new_id(),
            state_id="agent",
            task_id="2+2?",
            bench="math",
            messages=[
                Message(role="system", content="You are a math solver."),
                Message(role="user", content="2+2?"),
                Message(role="assistant", content="5"),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=10.0)],
            summary=EpisodeSummary(total_reward=0.0),
        )

        # Reflector returns an "add" insight: "Show work"
        reflector_client = MockLLMClient(responses=[
            _insight_response("Show work"),
        ])

        agent = LfXAgent(
            task_client=MockLLMClient(),
            reflector_client=reflector_client,
            bench="math",
            base_system_prompt="You are a math solver.",
        )

        agent.ingest([episode])

        # Assert "Show work" appears in the system prompt via the playbook
        prompt = agent.get_system_prompt()
        assert "Show work" in prompt, (
            f"Expected 'Show work' in system prompt after ingest, got:\n{prompt}"
        )
