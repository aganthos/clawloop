"""Tests for clawloop.core.reflector — LLM-based trace analysis."""

import json

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.learning_layers.harness import Insight, Playbook, PlaybookEntry
from clawloop.llm import MockLLMClient


def _make_episode(
    task_id: str = "task-001",
    reward: float = 0.8,
    bench: str = "test",
    n_steps: int = 2,
) -> Episode:
    """Helper to build a minimal Episode for testing."""
    messages: list[Message] = []
    step_boundaries: list[int] = []
    steps: list[StepMeta] = []

    messages.append(Message(role="system", content="You are a helpful assistant."))

    for t in range(n_steps):
        step_boundaries.append(len(messages))
        messages.append(Message(role="user", content=f"Task step {t}"))
        messages.append(
            Message(role="assistant", content=f"Response {t}", model="test-model")
        )
        is_terminal = t == n_steps - 1
        steps.append(
            StepMeta(
                t=t,
                reward=reward if is_terminal else 0.0,
                done=is_terminal,
                timing_ms=100.0,
            )
        )

    return Episode(
        id=Episode.new_id(),
        state_id="test-state",
        task_id=task_id,
        bench=bench,
        messages=messages,
        step_boundaries=step_boundaries,
        steps=steps,
        summary=EpisodeSummary(total_reward=reward),
    )


def _valid_insights_json() -> str:
    """JSON response with valid insight objects."""
    return json.dumps([
        {
            "action": "add",
            "content": "Always verify input format before processing.",
            "target_entry_id": None,
            "tags": ["validation", "robustness"],
            "source_episode_ids": ["ep-1"],
        }
    ])


def _update_insight_json(target_id: str) -> str:
    """JSON response with an update-action insight."""
    return json.dumps([
        {
            "action": "update",
            "content": "Updated strategy: check both input and output formats.",
            "target_entry_id": target_id,
            "tags": ["validation"],
            "source_episode_ids": ["ep-1"],
        }
    ])


class TestReflector:
    def test_reflect_returns_insights(self) -> None:
        """Mock LLM returns valid JSON, get back Insight objects."""
        client = MockLLMClient(responses=[_valid_insights_json()])
        reflector = Reflector(client=client, config=ReflectorConfig())
        episodes = [_make_episode()]
        playbook = Playbook()

        insights = reflector.reflect(episodes, playbook)

        assert len(insights) == 1
        assert isinstance(insights[0], Insight)
        assert insights[0].action == "add"
        assert insights[0].content == "Always verify input format before processing."
        assert "validation" in insights[0].tags

    def test_reflect_with_existing_playbook(self) -> None:
        """Update action with target_entry_id references existing playbook entry."""
        entry = PlaybookEntry(
            id="str-abc12345",
            content="Check input format.",
            tags=["validation"],
        )
        playbook = Playbook(entries=[entry])
        client = MockLLMClient(
            responses=[_update_insight_json("str-abc12345")]
        )
        reflector = Reflector(client=client, config=ReflectorConfig())
        episodes = [_make_episode()]

        insights = reflector.reflect(episodes, playbook)

        assert len(insights) == 1
        assert insights[0].action == "update"
        assert insights[0].target_entry_id == "str-abc12345"

    def test_reflect_empty_episodes_returns_empty(self) -> None:
        """No LLM call made when episodes list is empty."""
        client = MockLLMClient(responses=["should not be called"])
        reflector = Reflector(client=client, config=ReflectorConfig())
        playbook = Playbook()

        insights = reflector.reflect([], playbook)

        assert insights == []
        assert len(client.call_log) == 0

    def test_reflect_bad_json_returns_empty(self) -> None:
        """Graceful degradation on bad JSON — returns empty list, no exception."""
        client = MockLLMClient(responses=["this is not valid json at all"])
        reflector = Reflector(client=client, config=ReflectorConfig())
        episodes = [_make_episode()]
        playbook = Playbook()

        insights = reflector.reflect(episodes, playbook)

        assert insights == []

    def test_reflect_prompt_includes_episode_traces(self) -> None:
        """Check that episode trace information appears in the prompt."""
        client = MockLLMClient(responses=[json.dumps([])])
        reflector = Reflector(client=client, config=ReflectorConfig())
        ep = _make_episode(task_id="task-xyz")
        playbook = Playbook()

        reflector.reflect([ep], playbook)

        # Inspect the messages sent to the LLM
        assert len(client.call_log) == 1
        messages, _kwargs = client.call_log[0]
        # The user message should contain episode trace info
        user_content = messages[-1]["content"]
        assert "task-xyz" in user_content
        assert "EPISODE TRACES" in user_content

    def test_reflect_prompt_includes_playbook(self) -> None:
        """Playbook text should appear in the prompt when non-empty."""
        entry = PlaybookEntry(
            id="str-pb001",
            content="Always double-check arithmetic.",
            tags=["math"],
        )
        playbook = Playbook(entries=[entry])
        client = MockLLMClient(responses=[json.dumps([])])
        reflector = Reflector(client=client, config=ReflectorConfig())
        episodes = [_make_episode()]

        reflector.reflect(episodes, playbook)

        messages, _kwargs = client.call_log[0]
        user_content = messages[-1]["content"]
        assert "CURRENT PLAYBOOK" in user_content
        assert "Always double-check arithmetic." in user_content

    def test_reflect_includes_sibling_context(self) -> None:
        """Sibling context data should appear in the prompt when provided."""
        client = MockLLMClient(responses=[json.dumps([])])
        reflector = Reflector(client=client, config=ReflectorConfig())
        episodes = [_make_episode()]
        playbook = Playbook()
        sibling_ctx = "Sibling agent found that retry logic improves robustness."

        reflector.reflect(episodes, playbook, sibling_context=sibling_ctx)

        messages, _kwargs = client.call_log[0]
        user_content = messages[-1]["content"]
        assert "SIBLING CONTEXT" in user_content
        assert "retry logic improves robustness" in user_content
