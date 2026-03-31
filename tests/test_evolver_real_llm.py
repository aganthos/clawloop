"""Real LLM test for Evolver protocol integration.

Validates the Evolver contract end-to-end with a real LLM:
1. LocalEvolver (Reflector + GEPA) produces insights from real episodes
2. Harness applies evolved playbook entries via optim_step
3. Multi-cycle learning loop accumulates playbook entries
4. HarnessSnapshot serialization roundtrip preserves real data

Uses Gemini Flash Lite via GOOGLE_API_KEY. No proxy required.
Skipped automatically if GOOGLE_API_KEY is not set.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.evolution import EvolverConfig, PromptEvolver
from clawloop.core.evolver import EvolverContext, HarnessSnapshot
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.core.types import Datum
from clawloop.evolvers.local import LocalEvolver
from clawloop.layers.harness import Harness, PlaybookEntry, PromptCandidate, ParetoFront
from clawloop.llm import LiteLLMClient

log = logging.getLogger(__name__)

_MODEL = os.environ.get("CLAWLOOP_MODEL", "gemini/gemini-3.1-flash-lite-preview")

skip_no_key = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY not set — skipping real LLM test",
)


def _make_episode(
    task_id: str,
    reward: float,
    question: str,
    answer: str,
    bench: str = "math",
) -> Episode:
    return Episode(
        id=Episode.new_id(),
        state_id="evolver-real-test",
        task_id=task_id,
        bench=bench,
        messages=[
            Message(role="system", content="You are a math tutor."),
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=200.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


@skip_no_key
class TestEvolverProtocolRealLLM:
    """Validates the Evolver protocol contract using real LLM calls."""

    def test_reflector_produces_insights_from_failures(self) -> None:
        """Real Reflector analyzes failure episodes and produces actionable insights."""
        client = LiteLLMClient(model=_MODEL)
        reflector = Reflector(
            client=client,
            config=ReflectorConfig(reflection_batch_size=3),
        )
        evolver = LocalEvolver(reflector=reflector)

        episodes = [
            _make_episode("t1", -0.8, "What is 15% of 200?", "25"),
            _make_episode("t2", -0.5, "Solve: 3x + 7 = 22", "x = 3"),
            _make_episode("t3", -0.3, "What is the area of a circle with r=5?", "25"),
        ]

        snapshot = HarnessSnapshot(
            system_prompts={"math": "You are a math tutor."},
            playbook_entries=[],
            pareto_fronts={},
            playbook_generation=0,
            playbook_version=0,
        )
        ctx = EvolverContext(reward_history=[-0.8, -0.5, -0.3], iteration=0)

        result = evolver.evolve(episodes, snapshot, ctx)

        assert len(result.insights) > 0, "Reflector should produce insights from failure episodes"
        assert result.provenance.backend == "local"
        for insight in result.insights:
            assert insight.content, "Insight must have content"
            assert insight.action in ("add", "update", "remove")

    def test_harness_forward_backward_optim_with_real_reflector(self) -> None:
        """Full cycle: Harness + real Reflector → forward_backward → optim_step → playbook grows."""
        client = LiteLLMClient(model=_MODEL)
        reflector = Reflector(
            client=client,
            config=ReflectorConfig(reflection_batch_size=3),
        )

        h = Harness(
            system_prompts={"math": "You are a math tutor."},
            evolver=LocalEvolver(reflector=reflector),
        )

        episodes = [
            _make_episode("t1", -0.7, "What is 12 * 15?", "160"),
            _make_episode("t2", -0.4, "Simplify: (x^2 - 4)/(x - 2)", "x - 2"),
        ]

        fb = h.forward_backward(Datum(episodes=episodes)).result()
        assert fb.status == "ok"
        n_insights = fb.metrics.get("insights_generated", 0)
        assert n_insights > 0, f"Expected insights from real reflector, got {n_insights}"

        result = h.optim_step().result()
        assert result.status == "ok"
        assert result.updates_applied >= 1
        assert len(h.playbook.entries) > 0, "Playbook should have entries after optim_step"

        log.info(
            "Real reflector produced %d insights → %d playbook entries",
            n_insights,
            len(h.playbook.entries),
        )

    def test_snapshot_roundtrip_with_real_playbook(self) -> None:
        """Harness with real evolved playbook → snapshot → to_dict → reconstruct."""
        client = LiteLLMClient(model=_MODEL)
        reflector = Reflector(
            client=client,
            config=ReflectorConfig(reflection_batch_size=2),
        )

        h = Harness(
            system_prompts={"math": "Solve math problems step by step."},
            evolver=LocalEvolver(reflector=reflector),
        )

        episodes = [
            _make_episode("t1", -0.6, "What is sqrt(144)?", "14"),
            _make_episode("t2", -0.9, "What is 7! (7 factorial)?", "720"),
        ]

        h.forward_backward(Datum(episodes=episodes))
        h.optim_step()

        # Snapshot the harness
        snapshot = h._build_snapshot()
        assert len(snapshot.playbook_entries) > 0

        # Roundtrip through dict
        snap_dict = snapshot.to_dict()
        assert "system_prompts" in snap_dict
        assert "playbook_entries" in snap_dict
        assert len(snap_dict["playbook_entries"]) == len(snapshot.playbook_entries)

        # Each entry should have real content from the reflector
        for entry in snap_dict["playbook_entries"]:
            assert entry["content"], "Entry content should be non-empty"
            assert isinstance(entry["tags"], list)

    def test_two_cycle_learning_accumulates(self) -> None:
        """Two forward_backward + optim_step cycles — playbook accumulates entries."""
        client = LiteLLMClient(model=_MODEL)
        reflector = Reflector(
            client=client,
            config=ReflectorConfig(reflection_batch_size=2),
        )

        h = Harness(
            system_prompts={"math": "You are a math tutor."},
            evolver=LocalEvolver(reflector=reflector),
        )

        # Cycle 1: arithmetic failures
        eps_1 = [
            _make_episode("t1", -0.5, "What is 17 * 23?", "381"),
            _make_episode("t2", -0.7, "What is 256 / 16?", "14"),
        ]
        h.forward_backward(Datum(episodes=eps_1))
        h.optim_step()
        entries_after_1 = len(h.playbook.entries)
        assert entries_after_1 > 0, "Cycle 1 should add playbook entries"

        # Cycle 2: algebra failures
        eps_2 = [
            _make_episode("t3", -0.4, "Solve: 2x - 5 = 11", "x = 3"),
            _make_episode("t4", -0.6, "Factor: x^2 + 5x + 6", "(x+1)(x+6)"),
        ]
        h.forward_backward(Datum(episodes=eps_2))
        h.optim_step()
        entries_after_2 = len(h.playbook.entries)

        # Should have at least as many entries (may not grow if reflector
        # produces update insights instead of add, but shouldn't shrink)
        assert entries_after_2 >= entries_after_1, (
            f"Playbook should not shrink: {entries_after_1} → {entries_after_2}"
        )

        log.info(
            "Two-cycle learning: %d → %d playbook entries",
            entries_after_1,
            entries_after_2,
        )
