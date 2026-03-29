"""Real LLM test for SkyDiscover evaluator via CLIProxyAPI.

Exercises the evaluator with a real LLM call routed through CLIProxyAPI.
Validates that:
1. ClawLoopEvaluator can score a candidate program using real episodes
2. The full Harness integration works with real LLM-generated insights
3. Serialization roundtrip works with realistic data

Requires CLIProxyAPI running at localhost:8317.
Skipped automatically if the proxy is not reachable.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Any

import pytest

from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.core.types import Datum
from clawloop.evolvers.local import LocalEvolver
from clawloop.layers.harness import Harness
from clawloop.llm import LiteLLMClient

from enterprise.evolution.backends.skydiscover_evaluator import ClawLoopEvaluator
from enterprise.evolution.backends.skydiscover_utils import (
    harness_to_program,
    program_to_evolver_result,
)


def _proxy_available() -> bool:
    """Check if CLIProxyAPI is running at localhost:8317."""
    try:
        with socket.create_connection(("127.0.0.1", 8317), timeout=1):
            return True
    except (ConnectionRefusedError, OSError):
        return False


skip_no_proxy = pytest.mark.skipif(
    not _proxy_available(),
    reason="CLIProxyAPI not running at localhost:8317",
)

_API_BASE = "http://127.0.0.1:8317/v1"
_API_KEY = os.environ.get("LFX_API_KEY", "kuhhandel-bench-key")
_MODEL = os.environ.get("CLAWLOOP_MODEL", "openai/claude-haiku-4-5-20251001")


def _make_real_episode(
    task_id: str,
    reward: float,
    question: str,
    answer: str,
) -> Episode:
    return Episode(
        id=Episode.new_id(),
        state_id="real-proxy-test",
        task_id=task_id,
        bench="math",
        messages=[
            Message(role="system", content="You are a math tutor."),
            Message(role="user", content=question),
            Message(role="assistant", content=answer),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=200.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class _SimpleAdapter:
    """Adapter that runs a single LLM call per episode via CLIProxyAPI."""

    def __init__(self) -> None:
        self._client = LiteLLMClient(
            model=_MODEL,
            api_base=_API_BASE,
            api_key=_API_KEY,
        )

    def run_episode(self, task: str, agent_state: Any) -> Episode:
        """Run one episode: ask the LLM a math question and score it."""
        result = self._client.complete(
            messages=[
                {"role": "system", "content": "Answer math questions. Reply with just the number."},
                {"role": "user", "content": task},
            ],
        )
        answer = str(result)
        # Simple scoring: check if "4" is in the answer for "2+2"
        reward = 0.5 if "4" in answer else -0.5

        return _make_real_episode(
            task_id=task,
            reward=reward,
            question=task,
            answer=answer,
        )


def _simple_agent_state_factory(
    system_prompt: str,
    playbook: list[dict[str, Any]],
) -> Any:
    """Create a minimal agent state for the evaluator."""
    from unittest.mock import MagicMock
    state = MagicMock()
    state.system_prompt = system_prompt
    state.playbook = playbook
    return state


@skip_no_proxy
class TestEvaluatorWithRealLLM:
    """Evaluator runs real LLM episodes through CLIProxyAPI."""

    def test_evaluator_scores_candidate_program(self, tmp_path: Path) -> None:
        """ClawLoopEvaluator calls real LLM and returns a valid score."""
        adapter = _SimpleAdapter()
        evaluator = ClawLoopEvaluator(
            adapter=adapter,
            tasks=["What is 2+2?", "What is 3*3?"],
            agent_state_factory=_simple_agent_state_factory,
            n_episodes=2,
        )

        program = {
            "system_prompt": "You are a math tutor. Show your work.",
            "playbook": [
                {"content": "Always verify arithmetic.", "tags": ["math"]},
            ],
        }
        prog_path = tmp_path / "candidate.json"
        prog_path.write_text(json.dumps(program))

        result = evaluator(str(prog_path))

        assert "combined_score" in result
        assert isinstance(result["combined_score"], float)
        assert -1.0 <= result["combined_score"] <= 1.0
        assert result["n_episodes"] == 2
        assert len(result["rewards"]) == 2


@skip_no_proxy
class TestSerializationWithRealReflector:
    """Real Reflector generates insights, serialization handles them."""

    def test_reflector_insights_survive_roundtrip(self, tmp_path: Path) -> None:
        """Real LLM Reflector → insights → Harness → snapshot → serialize → parse."""
        client = LiteLLMClient(
            model=_MODEL,
            api_base=_API_BASE,
            api_key=_API_KEY,
        )
        reflector = Reflector(
            client=client,
            config=ReflectorConfig(reflection_batch_size=2),
        )

        h = Harness(
            system_prompts={"math": "You are a math tutor."},
            evolver=LocalEvolver(reflector=reflector),
        )

        # Feed the harness some failing episodes
        episodes = [
            _make_real_episode("t1", -0.5, "What is 15% of 200?", "25"),
            _make_real_episode("t2", -0.8, "Solve: 3x + 7 = 22", "x = 3"),
        ]

        fb = h.forward_backward(Datum(episodes=episodes)).result()
        assert fb.status == "ok"

        # The reflector should have generated at least one insight
        n_insights = fb.metrics.get("insights_generated", 0)
        assert n_insights > 0, "Real reflector should produce insights from failure episodes"

        # Apply insights
        h.optim_step()

        # Serialize the harness state as a SkyDiscover program
        snapshot = h._build_snapshot()
        prog_path = harness_to_program(snapshot, str(tmp_path / "real_harness.json"))

        # Parse it back — should be a clean roundtrip (no diffs)
        result = program_to_evolver_result(prog_path, snapshot)
        assert result.insights == [], "Roundtrip should produce no diffs"
        assert result.candidates == {}

        # Verify the program file has the real playbook content
        data = json.loads(Path(prog_path).read_text())
        assert len(data["playbook"]) > 0, "Playbook should have entries from reflector"
        assert data["system_prompt"] == "You are a math tutor."
