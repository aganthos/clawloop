"""Tests for Harness integration with the Reflector."""

import json

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.reflector import Reflector
from lfx.core.types import Datum
from lfx.layers.harness import Harness, PlaybookEntry
from lfx.llm import MockLLMClient


def _make_episode(reward: float = 0.3) -> Episode:
    return Episode(
        id="ep-test", state_id="s1", task_id="t1", bench="math",
        messages=[
            Message(role="system", content="Solve math."),
            Message(role="user", content="2+2?"),
            Message(role="assistant", content="5"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


def _valid_insight_json() -> str:
    """JSON the mock LLM returns — one 'add' insight."""
    return json.dumps([
        {
            "action": "add",
            "content": "Always verify input format before processing.",
            "target_entry_id": None,
            "tags": ["validation", "robustness"],
            "source_episode_ids": ["ep-test"],
        }
    ])


class TestHarnessReflector:
    def test_forward_backward_with_reflector_accumulates_insights(self) -> None:
        """Harness(reflector=reflector) + forward_backward -> _pending.insights has the insight."""
        client = MockLLMClient(responses=[_valid_insight_json()])
        reflector = Reflector(client=client)
        h = Harness(reflector=reflector)

        datum = Datum(episodes=[_make_episode()])
        result = h.forward_backward(datum).result()

        assert result.status == "ok"
        assert result.metrics["insights_generated"] == 1
        assert len(h._pending.insights) == 1
        assert h._pending.insights[0].content == "Always verify input format before processing."

    def test_optim_step_applies_reflector_insights(self) -> None:
        """forward_backward then optim_step -> playbook has the new entry."""
        client = MockLLMClient(responses=[_valid_insight_json()])
        reflector = Reflector(client=client)
        h = Harness(reflector=reflector)

        datum = Datum(episodes=[_make_episode()])
        h.forward_backward(datum)

        # Before optim_step, playbook is empty
        assert len(h.playbook.entries) == 0

        result = h.optim_step().result()
        assert result.status == "ok"
        assert result.updates_applied >= 1

        # After optim_step, the insight should be in the playbook
        assert len(h.playbook.entries) == 1
        assert "verify input format" in h.playbook.entries[0].content

    def test_forward_backward_without_reflector_still_works(self) -> None:
        """Harness() (no reflector) with existing playbook entry -> still counts signals."""
        h = Harness()
        h.playbook.add(PlaybookEntry(id="s-1", content="Check arithmetic"))

        datum = Datum(episodes=[_make_episode(reward=0.8)])
        result = h.forward_backward(datum).result()

        assert result.status == "ok"
        assert result.metrics["episodes_processed"] == 1
        assert result.metrics["entries_signaled"] == 1
        # No insights_generated key when no reflector
        assert "insights_generated" not in result.metrics

    def test_forward_backward_no_mutation_with_reflector(self) -> None:
        """to_dict() before and after forward_backward must be identical (the core contract)."""
        client = MockLLMClient(responses=[_valid_insight_json()])
        reflector = Reflector(client=client)
        h = Harness(
            system_prompts={"math": "Solve problems"},
            reflector=reflector,
        )
        h.playbook.add(PlaybookEntry(id="s-1", content="Check work", helpful=2))

        state_before = json.dumps(h.to_dict(), sort_keys=True)
        h.forward_backward(Datum(episodes=[_make_episode()]))
        state_after = json.dumps(h.to_dict(), sort_keys=True)

        assert state_before == state_after

    def test_reflector_failure_degrades_gracefully(self) -> None:
        """MockLLMClient returns invalid JSON -> forward_backward still returns ok, no insights."""
        client = MockLLMClient(responses=["INVALID JSON!!!"])
        reflector = Reflector(client=client)
        h = Harness(reflector=reflector)

        datum = Datum(episodes=[_make_episode()])
        result = h.forward_backward(datum).result()

        assert result.status == "ok"
        # Reflector returns [] on parse failure, so 0 insights
        assert result.metrics["insights_generated"] == 0
        assert len(h._pending.insights) == 0

    def test_system_prompt_improves_after_learning(self) -> None:
        """forward_backward + optim_step -> system_prompt includes the new playbook entry."""
        client = MockLLMClient(responses=[_valid_insight_json()])
        reflector = Reflector(client=client)
        h = Harness(
            system_prompts={"math": "Solve problems carefully."},
            reflector=reflector,
        )

        datum = Datum(episodes=[_make_episode()])
        h.forward_backward(datum)
        h.optim_step()

        prompt = h.system_prompt("math")
        assert "Solve problems carefully." in prompt
        assert "verify input format" in prompt
