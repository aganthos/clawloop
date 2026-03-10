# Harness ICL Learning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the Harness layer learn from experience via a Reflector-Curator pipeline, demonstrated on MATH/AIME problems with measurable accuracy improvement.

**Architecture:** ACE-style playbook with incremental delta edits. A Reflector LLM reads episode traces and produces Insight objects. A deterministic Curator applies them to the playbook. SkyDiscover-inspired adaptive intensity controls reflection frequency. Paradigm breakthrough escapes stagnation. Two LLM client instances (task vs reflector) enable split billing.

**Tech Stack:** Python 3.11+, LiteLLM (multi-provider), pytest. No new heavy deps.

**Design doc:** `docs/plans/2026-03-10-harness-icl-learning-design.md`

---

## Task 1: LLM Client Abstraction

**Files:**
- Create: `lfx/llm.py`
- Test: `tests/test_llm.py`

**Step 1: Write the failing test**

```python
# tests/test_llm.py
"""Tests for lfx.llm — LLM client abstraction."""

import pytest
from lfx.llm import LiteLLMClient, MockLLMClient


class TestMockLLMClient:
    def test_returns_canned_response(self) -> None:
        client = MockLLMClient(responses=["hello"])
        result = client.complete([{"role": "user", "content": "hi"}])
        assert result == "hello"

    def test_cycles_responses(self) -> None:
        client = MockLLMClient(responses=["a", "b"])
        assert client.complete([]) == "a"
        assert client.complete([]) == "b"
        assert client.complete([]) == "a"

    def test_records_calls(self) -> None:
        client = MockLLMClient(responses=["ok"])
        client.complete([{"role": "user", "content": "test"}])
        assert len(client.call_log) == 1
        assert client.call_log[0][0] == [{"role": "user", "content": "test"}]

    def test_default_response(self) -> None:
        client = MockLLMClient()
        result = client.complete([])
        assert isinstance(result, str)


class TestLiteLLMClient:
    def test_init_stores_config(self) -> None:
        client = LiteLLMClient(model="claude-haiku-4-5-20251001")
        assert client.model == "claude-haiku-4-5-20251001"

    def test_init_with_api_key(self) -> None:
        client = LiteLLMClient(model="gpt-4o-mini", api_key="sk-test")
        assert client.api_key == "sk-test"

    def test_init_with_kwargs(self) -> None:
        client = LiteLLMClient(model="haiku", temperature=0.7, max_tokens=1000)
        assert client.default_kwargs["temperature"] == 0.7
        assert client.default_kwargs["max_tokens"] == 1000
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_llm.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lfx.llm'`

**Step 3: Write minimal implementation**

```python
# lfx/llm.py
"""LLM client abstraction — thin wrapper over LiteLLM.

Provides a Protocol for LLM completion and two implementations:
- LiteLLMClient: production client using litellm.completion()
- MockLLMClient: deterministic mock for testing (no API calls)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMClient(Protocol):
    """Minimal interface for LLM completion."""

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str: ...


@dataclass
class LiteLLMClient:
    """Production LLM client using LiteLLM.

    Parameters
    ----------
    model:
        LiteLLM model string (e.g. "claude-haiku-4-5-20251001", "gpt-4o-mini").
    api_key:
        Optional API key. If None, uses environment variable.
    **kwargs:
        Default parameters passed to every completion call
        (temperature, max_tokens, etc.).
    """

    model: str
    api_key: str | None = None
    default_kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self, model: str, api_key: str | None = None, **kwargs: Any,
    ) -> None:
        self.model = model
        self.api_key = api_key
        self.default_kwargs = kwargs

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """Call LiteLLM completion and return the response text."""
        import litellm

        merged = {**self.default_kwargs, **kwargs}
        response = litellm.completion(
            model=self.model,
            messages=messages,
            api_key=self.api_key,
            **merged,
        )
        return response.choices[0].message.content or ""


@dataclass
class MockLLMClient:
    """Deterministic mock for testing — no API calls.

    Cycles through ``responses`` on each call.
    Records all calls in ``call_log``.
    """

    responses: list[str] = field(default_factory=lambda: ["mock response"])
    call_log: list[tuple[list[dict], dict]] = field(default_factory=list)
    _call_idx: int = field(default=0, repr=False)

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        self.call_log.append((messages, kwargs))
        response = self.responses[self._call_idx % len(self.responses)]
        self._call_idx += 1
        return response
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_llm.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```
feat: add LLM client abstraction with LiteLLM and Mock implementations
```

---

## Task 2: TaskEnvironment Protocol + Sample/EvalResult

**Files:**
- Create: `lfx/core/env.py`
- Test: `tests/test_env.py`

**Step 1: Write the failing test**

```python
# tests/test_env.py
"""Tests for lfx.core.env — environment protocol and data types."""

from lfx.core.env import EvalResult, Sample, StaticTaskEnvironment


class TestSample:
    def test_creation(self) -> None:
        s = Sample(question="What is 2+2?", ground_truth="4")
        assert s.question == "What is 2+2?"
        assert s.ground_truth == "4"
        assert s.context == ""
        assert s.metadata == {}

    def test_with_metadata(self) -> None:
        s = Sample(
            question="Solve x^2=4",
            ground_truth="2",
            metadata={"difficulty": "easy", "source": "MATH"},
        )
        assert s.metadata["source"] == "MATH"


class TestEvalResult:
    def test_creation(self) -> None:
        r = EvalResult(score=0.8, feedback="Close but not exact")
        assert r.score == 0.8
        assert r.feedback == "Close but not exact"

    def test_defaults(self) -> None:
        r = EvalResult(score=1.0)
        assert r.feedback == ""
        assert r.metrics == {}

    def test_with_metrics(self) -> None:
        r = EvalResult(score=0.5, metrics={"precision": 0.6, "recall": 0.4})
        assert r.metrics["precision"] == 0.6


class TestStaticTaskEnvironment:
    def test_get_tasks(self) -> None:
        samples = [
            Sample(question="2+2?", ground_truth="4"),
            Sample(question="3+3?", ground_truth="6"),
        ]
        env = StaticTaskEnvironment(
            tasks=samples,
            evaluate_fn=lambda s, r: EvalResult(score=1.0),
        )
        assert len(env.get_tasks()) == 2

    def test_evaluate_calls_fn(self) -> None:
        sample = Sample(question="2+2?", ground_truth="4")
        env = StaticTaskEnvironment(
            tasks=[sample],
            evaluate_fn=lambda s, r: EvalResult(
                score=1.0 if r.strip() == s.ground_truth else 0.0,
                feedback=f"Expected {s.ground_truth}, got {r}",
            ),
        )
        r1 = env.evaluate(sample, "4")
        assert r1.score == 1.0
        r2 = env.evaluate(sample, "5")
        assert r2.score == 0.0
        assert "Expected 4, got 5" in r2.feedback
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_env.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lfx.core.env'`

**Step 3: Write minimal implementation**

```python
# lfx/core/env.py
"""Task environment protocol and data types.

Provides the interface for benchmarks and evaluation environments.
Users implement ``TaskEnvironment`` to plug in their own tasks and
scoring functions. ``StaticTaskEnvironment`` is a convenience for
fixed task sets with a simple scoring function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class Sample:
    """A single task for the agent to solve."""

    question: str
    context: str = ""
    ground_truth: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating an agent response against a sample."""

    score: float  # 0.0 - 1.0
    feedback: str = ""  # Actionable Side Information
    metrics: dict[str, float] = field(default_factory=dict)


class TaskEnvironment(Protocol):
    """Interface for benchmark environments."""

    def get_tasks(self) -> list[Sample]: ...

    def evaluate(self, sample: Sample, response: str) -> EvalResult: ...


@dataclass
class StaticTaskEnvironment:
    """Convenience environment for a fixed task set with a scoring function."""

    tasks: list[Sample]
    evaluate_fn: Callable[[Sample, str], EvalResult]

    def get_tasks(self) -> list[Sample]:
        return list(self.tasks)

    def evaluate(self, sample: Sample, response: str) -> EvalResult:
        return self.evaluate_fn(sample, response)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_env.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```
feat: add TaskEnvironment protocol with Sample and EvalResult types
```

---

## Task 3: Reflector — LLM-based trace analysis

**Files:**
- Create: `lfx/core/reflector.py`
- Test: `tests/test_reflector.py`

**Step 1: Write the failing test**

```python
# tests/test_reflector.py
"""Tests for lfx.core.reflector — LLM-based trace analysis."""

import json

from lfx.core.reflector import Reflector, ReflectorConfig
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.layers.harness import Harness, Insight, PlaybookEntry
from lfx.llm import MockLLMClient


def _make_episode(reward: float = 0.3, task_id: str = "t1") -> Episode:
    return Episode(
        id="ep-001", state_id="s1", task_id=task_id, bench="math",
        messages=[
            Message(role="system", content="Solve math problems."),
            Message(role="user", content="What is 2+2?"),
            Message(role="assistant", content="The answer is 5."),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class TestReflectorConfig:
    def test_defaults(self) -> None:
        cfg = ReflectorConfig()
        assert cfg.temperature == 0.7
        assert cfg.max_episodes_per_prompt == 5

    def test_custom(self) -> None:
        cfg = ReflectorConfig(temperature=0.3, max_episodes_per_prompt=10)
        assert cfg.temperature == 0.3


class TestReflector:
    def test_reflect_returns_insights(self) -> None:
        mock_response = json.dumps([
            {"action": "add", "content": "Show work step by step",
             "tags": ["strategy"], "target_entry_id": None,
             "source_episode_ids": ["ep-001"]},
        ])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)

        episodes = [_make_episode(reward=0.3)]
        playbook = Harness().playbook
        insights = reflector.reflect(episodes, playbook)

        assert len(insights) == 1
        assert insights[0].action == "add"
        assert "step by step" in insights[0].content

    def test_reflect_with_existing_playbook(self) -> None:
        mock_response = json.dumps([
            {"action": "update", "content": "Always verify by substitution",
             "tags": ["verification"], "target_entry_id": "s-existing",
             "source_episode_ids": ["ep-001"]},
        ])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)

        playbook = Harness().playbook
        playbook.add(PlaybookEntry(id="s-existing", content="Check your work"))
        insights = reflector.reflect([_make_episode()], playbook)

        assert len(insights) == 1
        assert insights[0].action == "update"
        assert insights[0].target_entry_id == "s-existing"

    def test_reflect_empty_episodes_returns_empty(self) -> None:
        client = MockLLMClient()
        reflector = Reflector(client=client)
        insights = reflector.reflect([], Harness().playbook)
        assert insights == []
        assert len(client.call_log) == 0

    def test_reflect_bad_json_returns_empty(self) -> None:
        client = MockLLMClient(responses=["not valid json at all"])
        reflector = Reflector(client=client)
        insights = reflector.reflect([_make_episode()], Harness().playbook)
        assert insights == []

    def test_reflect_prompt_includes_episode_traces(self) -> None:
        mock_response = json.dumps([])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)
        reflector.reflect([_make_episode()], Harness().playbook)

        # Check the prompt sent to the LLM
        assert len(client.call_log) == 1
        messages = client.call_log[0][0]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "What is 2+2?" in user_msg
        assert "The answer is 5" in user_msg

    def test_reflect_prompt_includes_playbook(self) -> None:
        mock_response = json.dumps([])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)

        playbook = Harness().playbook
        playbook.add(PlaybookEntry(id="s-1", content="Think carefully"))
        reflector.reflect([_make_episode()], playbook)

        messages = client.call_log[0][0]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "Think carefully" in user_msg

    def test_reflect_includes_sibling_context(self) -> None:
        mock_response = json.dumps([])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)

        sibling_context = {
            "s-1": [
                {"content": "v1: show work", "avg_reward": 0.4},
                {"content": "v2: verify answer", "avg_reward": 0.7},
            ]
        }
        reflector.reflect(
            [_make_episode()], Harness().playbook,
            sibling_context=sibling_context,
        )

        messages = client.call_log[0][0]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "v1: show work" in user_msg or "sibling" in user_msg.lower()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_reflector.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# lfx/core/reflector.py
"""Reflector — LLM-based episode trace analysis.

Reads episode traces + current playbook and produces Insight objects
(add/update/remove operations on the playbook). Inspired by ACE's
Reflector with SkyDiscover's sibling context.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from lfx.core.episode import Episode
from lfx.layers.harness import Insight, Playbook

log = logging.getLogger(__name__)


@dataclass
class ReflectorConfig:
    """Configuration for the Reflector LLM call."""

    temperature: float = 0.7
    max_episodes_per_prompt: int = 5
    max_tokens: int = 2000


@dataclass
class Reflector:
    """Analyses episode traces and produces playbook Insights.

    Parameters
    ----------
    client:
        An LLM client (LiteLLMClient or MockLLMClient).
    config:
        Reflector configuration.
    """

    client: Any  # LLMClient protocol
    config: ReflectorConfig = field(default_factory=ReflectorConfig)

    def reflect(
        self,
        episodes: list[Episode],
        playbook: Playbook,
        *,
        sibling_context: dict[str, list[dict[str, Any]]] | None = None,
    ) -> list[Insight]:
        """Analyse episodes and return playbook Insights.

        Parameters
        ----------
        episodes:
            Batch of episodes to analyse.
        playbook:
            Current playbook state (for context in the prompt).
        sibling_context:
            Optional SkyDiscover-style context: entry_id -> list of
            previous mutations with their avg_reward.

        Returns
        -------
        list[Insight]
            Proposed playbook operations.
        """
        if not episodes:
            return []

        # Limit batch size
        episodes = episodes[: self.config.max_episodes_per_prompt]

        prompt = self._build_prompt(episodes, playbook, sibling_context)
        messages = [
            {"role": "system", "content": _REFLECTOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self.client.complete(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return self._parse_response(raw)
        except Exception:
            log.exception("Reflector LLM call failed")
            return []

    def _build_prompt(
        self,
        episodes: list[Episode],
        playbook: Playbook,
        sibling_context: dict[str, list[dict[str, Any]]] | None,
    ) -> str:
        sections: list[str] = []

        # Current playbook
        pb_text = playbook.render()
        if pb_text:
            sections.append(f"## CURRENT PLAYBOOK\n{pb_text}")
        else:
            sections.append("## CURRENT PLAYBOOK\n(empty — no strategies yet)")

        # Episode traces
        sections.append("## EPISODE TRACES")
        for ep in episodes:
            reward = ep.summary.total_reward
            trace_lines = [f"### Episode {ep.id} — task={ep.task_id}, reward={reward:.2f}"]
            for msg in ep.messages:
                role = msg.role.upper()
                content = msg.content[:500]  # truncate long messages
                trace_lines.append(f"  [{role}]: {content}")
            # Include score breakdown if available
            if ep.summary.score_breakdown:
                trace_lines.append(f"  Score breakdown: {ep.summary.score_breakdown}")
            sections.append("\n".join(trace_lines))

        # Sibling context (SkyDiscover)
        if sibling_context:
            sections.append("## SIBLING CONTEXT (previous mutations)")
            for entry_id, siblings in sibling_context.items():
                lines = [f"Entry [{entry_id}] — previous mutations:"]
                for sib in siblings:
                    lines.append(
                        f"  - {sib.get('content', '?')} → avg_reward={sib.get('avg_reward', '?')}"
                    )
                sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _parse_response(self, raw: str) -> list[Insight]:
        """Parse JSON array of insight dicts from the LLM response."""
        # Try to extract JSON from the response
        text = raw.strip()

        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("Reflector returned invalid JSON: %s", text[:200])
            return []

        if not isinstance(data, list):
            data = [data]

        insights: list[Insight] = []
        for item in data:
            try:
                insight = Insight(
                    content=item["content"],
                    action=item.get("action", "add"),
                    target_entry_id=item.get("target_entry_id"),
                    tags=item.get("tags", []),
                    source_episode_ids=item.get("source_episode_ids", []),
                )
                insights.append(insight)
            except (KeyError, ValueError) as e:
                log.warning("Skipping malformed insight: %s — %s", item, e)

        return insights


_REFLECTOR_SYSTEM_PROMPT = """\
You are a learning analyst for an AI agent system. Your job is to analyse
episode execution traces and extract reusable strategies for the agent's
playbook.

For each batch of episodes, you will:
1. Identify WHY episodes succeeded or failed.
2. Extract general strategies (not task-specific answers).
3. Propose playbook operations: add new entries, update existing ones,
   or remove unhelpful ones.

Respond with a JSON array of insight objects:
```json
[
  {
    "action": "add" | "update" | "remove",
    "content": "The strategy or lesson learned",
    "target_entry_id": "entry-id (for update/remove, null for add)",
    "tags": ["category1", "category2"],
    "source_episode_ids": ["ep-id1"]
  }
]
```

Rules:
- Extract GENERAL strategies, not specific answers to specific problems.
- Each insight should be actionable and concise (1-2 sentences).
- For updates, explain what changed and why.
- For removals, explain why the entry is no longer helpful.
- If episodes show the current playbook is working well, return [].
- Maximum 3 insights per batch to prevent noise.
"""
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_reflector.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```
feat: add Reflector for LLM-based episode trace analysis
```

---

## Task 4: Paradigm Breakthrough

**Files:**
- Create: `lfx/core/paradigm.py`
- Test: `tests/test_paradigm.py`

**Step 1: Write the failing test**

```python
# tests/test_paradigm.py
"""Tests for lfx.core.paradigm — stagnation escape via paradigm shifts."""

import json

from lfx.core.paradigm import ParadigmBreakthrough, ParadigmConfig
from lfx.layers.harness import Harness, PlaybookEntry
from lfx.llm import MockLLMClient


class TestParadigmBreakthrough:
    def test_generate_returns_insights(self) -> None:
        mock_response = json.dumps([
            {"content": "Try decomposing problems into sub-problems",
             "tags": ["paradigm", "decomposition"]},
            {"content": "Use worked examples as templates",
             "tags": ["paradigm", "examples"]},
        ])
        client = MockLLMClient(responses=[mock_response])
        pb = ParadigmBreakthrough(client=client)
        playbook = Harness().playbook

        insights = pb.generate(
            playbook=playbook,
            reward_history=[0.3, 0.31, 0.29, 0.30, 0.32],
            tried_paradigms=[],
        )
        assert len(insights) == 2
        assert all("paradigm" in i.tags for i in insights)

    def test_generate_includes_tried_paradigms(self) -> None:
        mock_response = json.dumps([])
        client = MockLLMClient(responses=[mock_response])
        pb = ParadigmBreakthrough(client=client)

        tried = ["decomposition approach", "chain-of-thought"]
        pb.generate(
            playbook=Harness().playbook,
            reward_history=[0.3, 0.3],
            tried_paradigms=tried,
        )

        messages = client.call_log[0][0]
        user_msg = next(m["content"] for m in messages if m["role"] == "user")
        assert "decomposition approach" in user_msg

    def test_bad_json_returns_empty(self) -> None:
        client = MockLLMClient(responses=["garbage"])
        pb = ParadigmBreakthrough(client=client)
        insights = pb.generate(
            playbook=Harness().playbook,
            reward_history=[0.3],
            tried_paradigms=[],
        )
        assert insights == []


class TestParadigmConfig:
    def test_defaults(self) -> None:
        cfg = ParadigmConfig()
        assert cfg.max_paradigms == 3
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_paradigm.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lfx/core/paradigm.py
"""Paradigm Breakthrough — SkyDiscover-inspired stagnation escape.

When the learning loop stagnates (reward stops improving), this module
asks a strong LLM to generate entirely new strategic directions for
the agent's playbook.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from lfx.layers.harness import Insight, Playbook

log = logging.getLogger(__name__)


@dataclass
class ParadigmConfig:
    """Configuration for paradigm breakthrough generation."""

    max_paradigms: int = 3
    temperature: float = 0.9
    max_tokens: int = 1500


@dataclass
class ParadigmBreakthrough:
    """Generates new strategic directions when learning stagnates."""

    client: Any  # LLMClient protocol
    config: ParadigmConfig = field(default_factory=ParadigmConfig)

    def generate(
        self,
        playbook: Playbook,
        reward_history: list[float],
        tried_paradigms: list[str],
    ) -> list[Insight]:
        """Generate paradigm-shift insights.

        Parameters
        ----------
        playbook:
            Current playbook state.
        reward_history:
            Recent average rewards per iteration.
        tried_paradigms:
            Descriptions of previously tried paradigm shifts.
        """
        prompt = self._build_prompt(playbook, reward_history, tried_paradigms)
        messages = [
            {"role": "system", "content": _PARADIGM_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self.client.complete(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            return self._parse_response(raw)
        except Exception:
            log.exception("Paradigm breakthrough LLM call failed")
            return []

    def _build_prompt(
        self,
        playbook: Playbook,
        reward_history: list[float],
        tried_paradigms: list[str],
    ) -> str:
        sections: list[str] = []

        # Current playbook
        pb_text = playbook.render()
        if pb_text:
            sections.append(f"## CURRENT PLAYBOOK\n{pb_text}")
        else:
            sections.append("## CURRENT PLAYBOOK\n(empty)")

        # Reward history
        history_str = ", ".join(f"{r:.3f}" for r in reward_history)
        sections.append(f"## REWARD HISTORY\n[{history_str}]")

        # Previously tried paradigms
        if tried_paradigms:
            lines = ["## PREVIOUSLY TRIED (do NOT repeat these)"]
            for p in tried_paradigms:
                lines.append(f"- {p}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    def _parse_response(self, raw: str) -> list[Insight]:
        text = raw.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            log.warning("Paradigm returned invalid JSON: %s", text[:200])
            return []

        if not isinstance(data, list):
            data = [data]

        insights: list[Insight] = []
        for item in data[: self.config.max_paradigms]:
            tags = item.get("tags", [])
            if "paradigm" not in tags:
                tags.append("paradigm")
            try:
                insights.append(Insight(
                    content=item["content"],
                    action="add",
                    tags=tags,
                ))
            except (KeyError, ValueError) as e:
                log.warning("Skipping malformed paradigm: %s — %s", item, e)

        return insights


_PARADIGM_SYSTEM_PROMPT = """\
You are a strategic advisor for an AI agent learning system. The agent's
performance has stagnated — incremental improvements are no longer working.

Your job is to propose FUNDAMENTALLY NEW strategic directions. Not tweaks
to existing strategies, but entirely new approaches the agent hasn't tried.

Respond with a JSON array of paradigm objects:
```json
[
  {
    "content": "Description of the new strategic direction",
    "tags": ["paradigm", "category"]
  }
]
```

Rules:
- Propose 1-3 genuinely new directions.
- Each should be a high-level strategy, not a specific tactic.
- Do NOT repeat previously tried paradigms.
- Be creative but practical — the strategy must be implementable as
  a system prompt instruction.
"""
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_paradigm.py -v`
Expected: PASS (all 4 tests)

**Step 5: Commit**

```
feat: add ParadigmBreakthrough for stagnation escape
```

---

## Task 5: Adaptive Reflection Intensity

**Files:**
- Create: `lfx/core/intensity.py`
- Test: `tests/test_intensity.py`

**Step 1: Write the failing test**

```python
# tests/test_intensity.py
"""Tests for lfx.core.intensity — adaptive reflection scheduling."""

from lfx.core.intensity import AdaptiveIntensity


class TestAdaptiveIntensity:
    def test_initial_should_reflect_true(self) -> None:
        ai = AdaptiveIntensity()
        assert ai.should_reflect(iteration=0) is True

    def test_improving_reflects_less(self) -> None:
        ai = AdaptiveIntensity(reflect_every_n=3)
        # Record improving rewards
        ai.record_reward(0.3)
        ai.record_reward(0.4)
        ai.record_reward(0.5)
        ai.record_reward(0.6)
        # When improving, only reflect every Nth
        assert ai.should_reflect(iteration=3) is True
        assert ai.should_reflect(iteration=4) is False
        assert ai.should_reflect(iteration=5) is False
        assert ai.should_reflect(iteration=6) is True

    def test_stagnating_always_reflects(self) -> None:
        ai = AdaptiveIntensity(
            reflect_every_n=3,
            stagnation_window=3,
            stagnation_threshold=0.01,
        )
        ai.record_reward(0.3)
        ai.record_reward(0.3)
        ai.record_reward(0.3)
        # Stagnating — should always reflect
        assert ai.should_reflect(iteration=4) is True
        assert ai.should_reflect(iteration=5) is True

    def test_is_stagnating(self) -> None:
        ai = AdaptiveIntensity(stagnation_window=3, stagnation_threshold=0.01)
        ai.record_reward(0.5)
        ai.record_reward(0.5)
        ai.record_reward(0.5)
        assert ai.is_stagnating() is True

    def test_not_stagnating_with_improvement(self) -> None:
        ai = AdaptiveIntensity(stagnation_window=3, stagnation_threshold=0.01)
        ai.record_reward(0.3)
        ai.record_reward(0.4)
        ai.record_reward(0.5)
        assert ai.is_stagnating() is False

    def test_not_stagnating_insufficient_data(self) -> None:
        ai = AdaptiveIntensity(stagnation_window=5)
        ai.record_reward(0.3)
        ai.record_reward(0.3)
        assert ai.is_stagnating() is False

    def test_improvement_signal(self) -> None:
        ai = AdaptiveIntensity()
        ai.record_reward(0.3)
        ai.record_reward(0.5)
        assert ai.improvement_signal() > 0

    def test_improvement_signal_no_data(self) -> None:
        ai = AdaptiveIntensity()
        assert ai.improvement_signal() == 0.0
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_intensity.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# lfx/core/intensity.py
"""Adaptive reflection intensity — SkyDiscover-inspired scheduling.

Controls when the Reflector fires based on the improvement signal.
When improving fast, reflect less frequently (exploit). When stagnating,
reflect every iteration and trigger paradigm breakthrough.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdaptiveIntensity:
    """Tracks reward history and decides when to reflect.

    Parameters
    ----------
    reflect_every_n:
        When improving, reflect every N iterations (skip otherwise).
    stagnation_window:
        Number of recent rewards to check for stagnation.
    stagnation_threshold:
        If max - min of recent rewards < threshold, we're stagnating.
    """

    reflect_every_n: int = 3
    stagnation_window: int = 5
    stagnation_threshold: float = 0.02
    _rewards: list[float] = field(default_factory=list)

    def record_reward(self, avg_reward: float) -> None:
        """Record the average reward of an iteration."""
        self._rewards.append(avg_reward)

    def should_reflect(self, iteration: int) -> bool:
        """Decide whether to run the Reflector this iteration."""
        # Always reflect on the first iteration
        if iteration == 0 or len(self._rewards) < 2:
            return True
        # Always reflect when stagnating
        if self.is_stagnating():
            return True
        # When improving, reflect every Nth iteration
        return iteration % self.reflect_every_n == 0

    def is_stagnating(self) -> bool:
        """Check if recent rewards show stagnation."""
        if len(self._rewards) < self.stagnation_window:
            return False
        recent = self._rewards[-self.stagnation_window:]
        return (max(recent) - min(recent)) < self.stagnation_threshold

    def improvement_signal(self) -> float:
        """Return the recent improvement magnitude (G from SkyDiscover)."""
        if len(self._rewards) < 2:
            return 0.0
        return self._rewards[-1] - self._rewards[-2]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_intensity.py -v`
Expected: PASS (all 8 tests)

**Step 5: Commit**

```
feat: add AdaptiveIntensity for reflection scheduling
```

---

## Task 6: MATH/AIME Environment

**Files:**
- Create: `lfx/envs/__init__.py`
- Create: `lfx/envs/math.py`
- Create: `data/math_aime_50.json` (curated problem set)
- Test: `tests/test_math_env.py`

**Step 1: Write the failing test**

```python
# tests/test_math_env.py
"""Tests for lfx.envs.math — MATH/AIME environment."""

from lfx.envs.math import MathEnvironment, extract_answer


class TestExtractAnswer:
    def test_boxed_answer(self) -> None:
        assert extract_answer(r"The answer is \boxed{42}") == "42"

    def test_boxed_fraction(self) -> None:
        assert extract_answer(r"\boxed{\frac{1}{2}}") == "\\frac{1}{2}"

    def test_plain_number_last_line(self) -> None:
        assert extract_answer("Step 1: ...\nStep 2: ...\n42") == "42"

    def test_answer_is_prefix(self) -> None:
        assert extract_answer("The answer is 7.") == "7"

    def test_no_answer_returns_full(self) -> None:
        resp = "I don't know"
        assert extract_answer(resp) == resp.strip()

    def test_negative_number(self) -> None:
        assert extract_answer(r"\boxed{-3}") == "-3"


class TestMathEnvironment:
    def test_get_tasks_returns_samples(self) -> None:
        env = MathEnvironment()
        tasks = env.get_tasks()
        assert len(tasks) > 0
        assert tasks[0].question
        assert tasks[0].ground_truth is not None

    def test_evaluate_correct(self) -> None:
        env = MathEnvironment()
        tasks = env.get_tasks()
        sample = tasks[0]
        # Simulate correct response
        response = f"The answer is \\boxed{{{sample.ground_truth}}}"
        result = env.evaluate(sample, response)
        assert result.score == 1.0

    def test_evaluate_incorrect(self) -> None:
        env = MathEnvironment()
        tasks = env.get_tasks()
        sample = tasks[0]
        result = env.evaluate(sample, "The answer is \\boxed{WRONG}")
        assert result.score == 0.0
        assert "Expected" in result.feedback

    def test_evaluate_feedback_is_informative(self) -> None:
        env = MathEnvironment()
        tasks = env.get_tasks()
        sample = tasks[0]
        result = env.evaluate(sample, "I have no idea")
        assert sample.ground_truth in result.feedback
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_math_env.py -v`
Expected: FAIL

**Step 3: Create the curated problem set**

```python
# data/math_aime_50.json
# A curated set of 50 MATH/AIME-style problems at varying difficulty.
# Format: [{"question": "...", "answer": "...", "difficulty": "...", "source": "..."}]
```

This file should contain 50 hand-picked problems. For the plan, include
a script that generates the seed data. The actual problems will be curated
from publicly available MATH and AIME datasets.

**Step 3b: Write implementation**

```python
# lfx/envs/__init__.py
"""Benchmark environments."""

# lfx/envs/math.py
"""MATH/AIME environment — deterministic math problem evaluation.

Provides a curated set of competition math problems with exact-match
scoring. Answers are extracted from \\boxed{} notation or parsed from
the final line of the response.
"""

from __future__ import annotations

import json
import logging
import re
from importlib import resources
from pathlib import Path
from typing import Any

from lfx.core.env import EvalResult, Sample, StaticTaskEnvironment

log = logging.getLogger(__name__)

# Path to the curated problem set
_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "math_aime_50.json"


def extract_answer(response: str) -> str:
    """Extract the final answer from an LLM response.

    Tries in order:
    1. \\boxed{...} notation
    2. "answer is X" pattern
    3. Last number on the last non-empty line
    4. Full response stripped
    """
    # Try \boxed{...}
    boxed = re.findall(r"\\boxed\{([^}]+)\}", response)
    if boxed:
        return boxed[-1].strip()

    # Try "answer is X"
    answer_match = re.search(
        r"(?:answer|result)\s+is\s+([^\s.,]+)", response, re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip().rstrip(".")

    # Try last number on last line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        nums = re.findall(r"-?\d+(?:\.\d+)?", lines[-1])
        if nums:
            return nums[-1]

    return response.strip()


def _normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    s = answer.strip().lower()
    # Remove common LaTeX wrappers
    s = s.replace("$", "").replace("\\text{", "").replace("}", "")
    # Normalize fractions
    s = s.replace(" ", "")
    return s


def _load_problems() -> list[dict[str, Any]]:
    """Load the curated problem set from JSON."""
    if _DATA_PATH.exists():
        with open(_DATA_PATH) as f:
            return json.load(f)

    # Fallback: built-in mini set for testing/demo
    return _BUILTIN_PROBLEMS


class MathEnvironment:
    """MATH/AIME environment with deterministic exact-match scoring."""

    def __init__(self, problems: list[dict[str, Any]] | None = None) -> None:
        raw = problems if problems is not None else _load_problems()
        self._samples = [
            Sample(
                question=p["question"],
                ground_truth=str(p["answer"]),
                metadata={
                    k: v for k, v in p.items()
                    if k not in ("question", "answer")
                },
            )
            for p in raw
        ]

    def get_tasks(self) -> list[Sample]:
        return list(self._samples)

    def evaluate(self, sample: Sample, response: str) -> EvalResult:
        extracted = extract_answer(response)
        gt = sample.ground_truth or ""

        correct = _normalize_answer(extracted) == _normalize_answer(gt)
        score = 1.0 if correct else 0.0

        if correct:
            feedback = f"Correct: {gt}"
        else:
            feedback = f"Incorrect. Expected {gt}, got {extracted}."

        return EvalResult(
            score=score,
            feedback=feedback,
            metrics={"exact_match": score},
        )


# Built-in mini problem set (always available, no external file needed)
_BUILTIN_PROBLEMS = [
    {
        "question": "Find the value of $2^{10}$.",
        "answer": "1024",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "What is the sum of the first 100 positive integers?",
        "answer": "5050",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "If $f(x) = x^2 + 3x + 2$, find $f(5)$.",
        "answer": "42",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "How many integers between 1 and 1000 are divisible by both 3 and 5?",
        "answer": "66",
        "difficulty": "medium",
        "source": "MATH",
    },
    {
        "question": "Find the remainder when $7^{2023}$ is divided by 5.",
        "answer": "3",
        "difficulty": "medium",
        "source": "AIME",
    },
    {
        "question": "In triangle ABC, if a=5, b=7, c=8, find the area using Heron's formula. Give answer as a decimal rounded to 2 places.",
        "answer": "17.32",
        "difficulty": "medium",
        "source": "MATH",
    },
    {
        "question": "Find the number of positive divisors of 360.",
        "answer": "24",
        "difficulty": "medium",
        "source": "MATH",
    },
    {
        "question": "What is $\\binom{10}{3}$?",
        "answer": "120",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "Find the sum $1 + 2 + 4 + 8 + \\ldots + 2^9$.",
        "answer": "1023",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "If $\\log_2(x) = 5$, what is $x$?",
        "answer": "32",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "Find the value of $\\sqrt{144} + \\sqrt{169}$.",
        "answer": "25",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "How many ways can 5 people be arranged in a line?",
        "answer": "120",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "Find the greatest common divisor of 48 and 180.",
        "answer": "12",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "What is the least common multiple of 12 and 18?",
        "answer": "36",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "Solve for x: $3x + 7 = 22$.",
        "answer": "5",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "Find the sum of all prime numbers less than 20.",
        "answer": "77",
        "difficulty": "easy",
        "source": "MATH",
    },
    {
        "question": "A fair six-sided die is rolled twice. What is the probability that the sum is 7? Express as a simplified fraction.",
        "answer": "1/6",
        "difficulty": "medium",
        "source": "MATH",
    },
    {
        "question": "Find the coefficient of $x^3$ in the expansion of $(1+x)^7$.",
        "answer": "35",
        "difficulty": "medium",
        "source": "MATH",
    },
    {
        "question": "If $x + \\frac{1}{x} = 5$, find $x^2 + \\frac{1}{x^2}$.",
        "answer": "23",
        "difficulty": "medium",
        "source": "AIME",
    },
    {
        "question": "Find the number of trailing zeros in 25!.",
        "answer": "6",
        "difficulty": "medium",
        "source": "MATH",
    },
]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_math_env.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```
feat: add MATH/AIME environment with exact-match scoring
```

---

## Task 7: Wire Reflector into Harness.forward_backward

**Files:**
- Modify: `lfx/layers/harness.py`
- Test: `tests/test_harness_reflector.py`

This is the critical integration: when Harness.forward_backward is called
with a Reflector configured, it calls the Reflector LLM and accumulates
the resulting Insights into `_pending`.

**Step 1: Write the failing test**

```python
# tests/test_harness_reflector.py
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


class TestHarnessWithReflector:
    def test_forward_backward_with_reflector_accumulates_insights(self) -> None:
        mock_response = json.dumps([
            {"action": "add", "content": "Show your work",
             "tags": ["strategy"], "target_entry_id": None,
             "source_episode_ids": ["ep-test"]},
        ])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)
        h = Harness(reflector=reflector)

        datum = Datum(episodes=[_make_episode()])
        result = h.forward_backward(datum).result()
        assert result.status == "ok"
        # Insight should be in pending
        assert len(h._pending.insights) == 1
        assert h._pending.insights[0].content == "Show your work"

    def test_optim_step_applies_reflector_insights(self) -> None:
        mock_response = json.dumps([
            {"action": "add", "content": "Verify by substitution",
             "tags": ["verification"], "target_entry_id": None,
             "source_episode_ids": []},
        ])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)
        h = Harness(reflector=reflector)

        h.forward_backward(Datum(episodes=[_make_episode()]))
        result = h.optim_step().result()
        assert result.updates_applied >= 1
        # Playbook should now have the new entry
        assert any("Verify" in e.content for e in h.playbook.entries)

    def test_forward_backward_without_reflector_still_works(self) -> None:
        """Existing behavior: no reflector, just playbook signal counting."""
        h = Harness()
        h.playbook.add(PlaybookEntry(id="s-1", content="existing"))
        datum = Datum(episodes=[_make_episode(reward=0.9)])
        result = h.forward_backward(datum).result()
        assert result.status == "ok"

    def test_forward_backward_no_mutation_with_reflector(self) -> None:
        """forward_backward must not mutate observable state even with reflector."""
        mock_response = json.dumps([
            {"action": "add", "content": "New strategy", "tags": [],
             "target_entry_id": None, "source_episode_ids": []},
        ])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)
        h = Harness(
            reflector=reflector,
            system_prompts={"math": "Solve problems."},
        )
        state_before = json.dumps(h.to_dict(), sort_keys=True)
        h.forward_backward(Datum(episodes=[_make_episode()]))
        state_after = json.dumps(h.to_dict(), sort_keys=True)
        assert state_before == state_after

    def test_reflector_failure_degrades_gracefully(self) -> None:
        """If reflector LLM fails, forward_backward still succeeds."""
        client = MockLLMClient(responses=["INVALID JSON!!!"])
        reflector = Reflector(client=client)
        h = Harness(reflector=reflector)

        datum = Datum(episodes=[_make_episode()])
        result = h.forward_backward(datum).result()
        assert result.status == "ok"
        # No insights accumulated due to parse failure
        assert len(h._pending.insights) == 0

    def test_system_prompt_improves_after_learning(self) -> None:
        """End-to-end: system prompt includes new playbook entries."""
        mock_response = json.dumps([
            {"action": "add", "content": "Always double-check arithmetic",
             "tags": ["math"], "target_entry_id": None,
             "source_episode_ids": []},
        ])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)
        h = Harness(
            reflector=reflector,
            system_prompts={"math": "You solve math problems."},
        )

        h.forward_backward(Datum(episodes=[_make_episode()]))
        h.optim_step()

        prompt = h.system_prompt("math")
        assert "double-check arithmetic" in prompt
        assert "You solve math problems" in prompt
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_harness_reflector.py -v`
Expected: FAIL — Harness doesn't accept `reflector` parameter yet

**Step 3: Modify Harness to accept and use Reflector**

In `lfx/layers/harness.py`, add:

1. An optional `reflector` field to the `Harness` dataclass
2. In `forward_backward`, after the existing playbook signal counting,
   call `self.reflector.reflect()` if configured and accumulate results
   into `self._pending.insights`

Key changes:
- Add `reflector: Any | None = None` field to `Harness` (after existing fields)
- In `forward_backward`, add reflector call block after the existing
  signal counting loop
- Exclude `reflector` from `to_dict()` (it's not serializable state)

```python
# Add to Harness dataclass fields (after validators):
reflector: Any | None = field(default=None, repr=False)

# Add at end of forward_backward, before return:
if self.reflector is not None:
    try:
        insights = self.reflector.reflect(data.episodes, self.playbook)
        self._pending.insights.extend(insights)
        metrics["insights_generated"] = len(insights)
    except Exception:
        log.exception("Reflector failed during forward_backward")
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_harness_reflector.py -v`
Expected: PASS (all 6 tests)

**Step 5: Run all existing tests to verify no regression**

Run: `python -m pytest tests/ -v`
Expected: All 131 existing tests + 6 new tests PASS

**Step 6: Commit**

```
feat: wire Reflector into Harness.forward_backward
```

---

## Task 8: Wire Adaptive Intensity + Paradigm into the Learning Loop

**Files:**
- Modify: `lfx/core/loop.py`
- Test: `tests/test_loop_icl.py`

**Step 1: Write the failing test**

```python
# tests/test_loop_icl.py
"""Tests for the learning loop with ICL features (reflector, intensity, paradigm)."""

import json

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.intensity import AdaptiveIntensity
from lfx.core.loop import AgentState, learning_loop
from lfx.core.paradigm import ParadigmBreakthrough
from lfx.core.reflector import Reflector
from lfx.layers.harness import Harness
from lfx.llm import MockLLMClient


def _make_episode(reward: float = 0.5) -> Episode:
    return Episode(
        id="ep-1", state_id="s1", task_id="t1", bench="test",
        messages=[
            Message(role="system", content="test"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=50.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class _MockAdapter:
    def __init__(self, reward: float = 0.5) -> None:
        self.reward = reward
        self.call_count = 0

    def run_episode(self, task, agent_state) -> Episode:
        self.call_count += 1
        return _make_episode(reward=self.reward)


class TestLoopWithReflector:
    def test_loop_calls_reflector_via_harness(self) -> None:
        mock_response = json.dumps([
            {"action": "add", "content": "Be careful",
             "tags": [], "target_entry_id": None,
             "source_episode_ids": []},
        ])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)

        state = AgentState(harness=Harness(reflector=reflector))
        adapter = _MockAdapter()
        state, sid = learning_loop(
            adapter=adapter, agent_state=state,
            tasks=["t1"], n_episodes=1, n_iterations=1,
        )
        # Reflector should have been called
        assert len(client.call_log) > 0
        # Playbook should have the new entry
        assert len(state.harness.playbook.entries) > 0

    def test_loop_with_adaptive_intensity(self) -> None:
        """Adaptive intensity reduces reflector calls when improving."""
        mock_response = json.dumps([])
        client = MockLLMClient(responses=[mock_response])
        reflector = Reflector(client=client)
        intensity = AdaptiveIntensity(reflect_every_n=2)

        state = AgentState(harness=Harness(reflector=reflector))
        adapter = _MockAdapter()
        state, sid = learning_loop(
            adapter=adapter, agent_state=state,
            tasks=["t1"], n_episodes=1, n_iterations=4,
            intensity=intensity,
        )
        # With reflect_every_n=2 and 4 iterations,
        # should reflect fewer than 4 times (iteration 0 always reflects)
        assert len(client.call_log) <= 4

    def test_loop_without_reflector_still_works(self) -> None:
        """Backward compat: loop works without reflector."""
        state = AgentState()
        adapter = _MockAdapter()
        state, sid = learning_loop(
            adapter=adapter, agent_state=state,
            tasks=["t1"], n_episodes=1, n_iterations=2,
        )
        assert sid.combined_hash
        assert adapter.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_loop_icl.py -v`
Expected: FAIL — `learning_loop` doesn't accept `intensity` parameter

**Step 3: Modify learning_loop**

Add optional `intensity` and `paradigm` parameters. Use `intensity.should_reflect()`
to gate the forward_backward call on the harness layer. After each iteration, call
`intensity.record_reward()`. If `intensity.is_stagnating()` and paradigm is configured,
fire paradigm breakthrough.

Key changes to `lfx/core/loop.py`:
- Add `intensity: AdaptiveIntensity | None = None` parameter
- Add `paradigm: ParadigmBreakthrough | None = None` parameter
- After collecting episodes, compute avg_reward and call `intensity.record_reward()`
- Before forward_backward on harness, check `intensity.should_reflect()`
- After optim_step, if stagnating and paradigm configured, fire paradigm
- Record tried paradigms for deduplication

```python
# Add to function signature:
def learning_loop(
    adapter: AdapterLike,
    agent_state: AgentState,
    tasks: list[Any],
    n_episodes: int,
    n_iterations: int,
    *,
    active_layers: list[str] | None = None,
    intensity: AdaptiveIntensity | None = None,
    paradigm: ParadigmBreakthrough | None = None,
) -> tuple[AgentState, StateID]:
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_loop_icl.py tests/test_layer_protocol.py -v`
Expected: PASS (new + all existing loop tests)

**Step 5: Commit**

```
feat: wire adaptive intensity and paradigm into learning loop
```

---

## Task 9: LfXAgent Convenience Wrapper

**Files:**
- Create: `lfx/agent.py`
- Test: `tests/test_agent.py`

**Step 1: Write the failing test**

```python
# tests/test_agent.py
"""Tests for lfx.agent — high-level convenience wrapper."""

import json

from lfx.agent import LfXAgent
from lfx.core.env import EvalResult, Sample, StaticTaskEnvironment
from lfx.llm import MockLLMClient


def _make_env() -> StaticTaskEnvironment:
    return StaticTaskEnvironment(
        tasks=[
            Sample(question="What is 2+2?", ground_truth="4"),
            Sample(question="What is 3+3?", ground_truth="6"),
        ],
        evaluate_fn=lambda s, r: EvalResult(
            score=1.0 if s.ground_truth in r else 0.0,
            feedback=f"Expected {s.ground_truth}",
        ),
    )


class TestLfXAgent:
    def test_learn_runs_loop(self) -> None:
        reflector_response = json.dumps([])
        task_client = MockLLMClient(responses=["The answer is 4"])
        reflector_client = MockLLMClient(responses=[reflector_response])

        agent = LfXAgent(
            task_client=task_client,
            reflector_client=reflector_client,
        )
        results = agent.learn(env=_make_env(), iterations=1, episodes_per_iter=2)
        assert "rewards" in results
        assert len(results["rewards"]) == 1

    def test_get_system_prompt_empty_initially(self) -> None:
        agent = LfXAgent(
            task_client=MockLLMClient(),
            reflector_client=MockLLMClient(),
        )
        prompt = agent.get_system_prompt()
        assert isinstance(prompt, str)

    def test_get_system_prompt_after_learning(self) -> None:
        reflector_response = json.dumps([
            {"action": "add", "content": "Think step by step",
             "tags": ["strategy"], "target_entry_id": None,
             "source_episode_ids": []},
        ])
        task_client = MockLLMClient(responses=["42"])
        reflector_client = MockLLMClient(responses=[reflector_response])

        agent = LfXAgent(
            task_client=task_client,
            reflector_client=reflector_client,
        )
        agent.learn(env=_make_env(), iterations=1, episodes_per_iter=1)
        prompt = agent.get_system_prompt()
        assert "step by step" in prompt

    def test_ingest_episodes(self) -> None:
        """Level 2: user provides pre-built episodes."""
        from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta

        reflector_response = json.dumps([
            {"action": "add", "content": "Check your work",
             "tags": [], "target_entry_id": None,
             "source_episode_ids": []},
        ])
        reflector_client = MockLLMClient(responses=[reflector_response])
        agent = LfXAgent(
            task_client=MockLLMClient(),
            reflector_client=reflector_client,
        )

        ep = Episode(
            id="ext-1", state_id="s1", task_id="t1", bench="external",
            messages=[
                Message(role="user", content="2+2?"),
                Message(role="assistant", content="5"),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=50)],
            summary=EpisodeSummary(total_reward=0.0),
        )
        agent.ingest([ep])
        prompt = agent.get_system_prompt()
        assert "Check your work" in prompt

    def test_save_load_playbook(self, tmp_path) -> None:
        reflector_response = json.dumps([
            {"action": "add", "content": "Be precise",
             "tags": [], "target_entry_id": None,
             "source_episode_ids": []},
        ])
        agent = LfXAgent(
            task_client=MockLLMClient(responses=["42"]),
            reflector_client=MockLLMClient(responses=[reflector_response]),
        )
        agent.learn(env=_make_env(), iterations=1, episodes_per_iter=1)

        path = tmp_path / "playbook.json"
        agent.save_playbook(str(path))
        assert path.exists()

        agent2 = LfXAgent(
            task_client=MockLLMClient(),
            reflector_client=MockLLMClient(),
        )
        agent2.load_playbook(str(path))
        assert "Be precise" in agent2.get_system_prompt()
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# lfx/agent.py
"""LfXAgent — high-level convenience wrapper for the learning loop.

Three levels of usage:
  Level 1: agent.learn(env, iterations) — full plug-and-play
  Level 2: agent.ingest(episodes)       — bring your own traces
  Level 3: Use learning_loop() directly — full control
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from lfx.core.env import EvalResult, Sample, TaskEnvironment
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.intensity import AdaptiveIntensity
from lfx.core.loop import AgentState, learning_loop
from lfx.core.paradigm import ParadigmBreakthrough, ParadigmConfig
from lfx.core.reflector import Reflector
from lfx.core.types import Datum
from lfx.layers.harness import Harness

log = logging.getLogger(__name__)


@dataclass
class LfXAgent:
    """High-level agent that learns from experience.

    Parameters
    ----------
    task_client:
        LLM client for running tasks (customer's model/key).
    reflector_client:
        LLM client for reflection and paradigm (LfX-managed).
    bench:
        Benchmark name for system prompt resolution.
    base_system_prompt:
        Initial system prompt before any learning.
    """

    task_client: Any  # LLMClient
    reflector_client: Any  # LLMClient
    bench: str = "default"
    base_system_prompt: str = ""

    _harness: Harness = field(init=False)
    _intensity: AdaptiveIntensity = field(init=False)
    _tried_paradigms: list[str] = field(init=False)

    def __post_init__(self) -> None:
        reflector = Reflector(client=self.reflector_client)
        self._harness = Harness(
            reflector=reflector,
            system_prompts={self.bench: self.base_system_prompt},
        )
        self._intensity = AdaptiveIntensity()
        self._tried_paradigms = []

    def learn(
        self,
        env: TaskEnvironment,
        iterations: int = 5,
        episodes_per_iter: int = 5,
    ) -> dict[str, Any]:
        """Run the learning loop on the given environment.

        Returns a dict with reward history and final playbook.
        """
        tasks = env.get_tasks()
        rewards: list[float] = []

        for i in range(iterations):
            # Run episodes
            episodes: list[Episode] = []
            import random
            batch = random.sample(tasks, min(episodes_per_iter, len(tasks)))

            for sample in batch:
                ep = self._run_one(sample, env)
                episodes.append(ep)

            avg_reward = (
                sum(ep.summary.total_reward for ep in episodes) / len(episodes)
                if episodes
                else 0.0
            )
            rewards.append(avg_reward)
            self._intensity.record_reward(avg_reward)

            log.info("Iteration %d/%d — avg reward: %.3f", i + 1, iterations, avg_reward)

            # Reflect (gated by adaptive intensity)
            if self._intensity.should_reflect(i):
                datum = Datum(episodes=episodes)
                self._harness.forward_backward(datum)
                self._harness.optim_step()

            # Paradigm breakthrough on stagnation
            if self._intensity.is_stagnating():
                pb = ParadigmBreakthrough(client=self.reflector_client)
                insights = pb.generate(
                    playbook=self._harness.playbook,
                    reward_history=rewards,
                    tried_paradigms=self._tried_paradigms,
                )
                for insight in insights:
                    self._tried_paradigms.append(insight.content)
                self._harness._pending.insights.extend(insights)
                self._harness.optim_step()

        return {
            "rewards": rewards,
            "playbook": self._harness.playbook.to_dict(),
            "n_entries": len(self._harness.playbook.entries),
        }

    def ingest(self, episodes: list[Episode]) -> None:
        """Learn from externally-provided episode traces (Level 2)."""
        datum = Datum(episodes=episodes)
        self._harness.forward_backward(datum)
        self._harness.optim_step()

    def get_system_prompt(self) -> str:
        """Return the current learned system prompt."""
        return self._harness.system_prompt(self.bench)

    def save_playbook(self, path: str) -> None:
        """Save the current playbook to a JSON file."""
        with open(path, "w") as f:
            json.dump(self._harness.playbook.to_dict(), f, indent=2)

    def load_playbook(self, path: str) -> None:
        """Load a playbook from a JSON file."""
        from lfx.layers.harness import Playbook, PlaybookEntry

        with open(path) as f:
            data = json.load(f)

        entries = [
            PlaybookEntry(
                id=e["id"],
                content=e["content"],
                helpful=e.get("helpful", 0),
                harmful=e.get("harmful", 0),
                tags=e.get("tags", []),
            )
            for e in data.get("entries", [])
        ]
        self._harness.playbook = Playbook(entries=entries)

    def _run_one(self, sample: Sample, env: TaskEnvironment) -> Episode:
        """Run a single task and return the episode."""
        prompt = self.get_system_prompt()
        messages_for_llm = []
        if prompt:
            messages_for_llm.append({"role": "system", "content": prompt})
        messages_for_llm.append({"role": "user", "content": sample.question})

        response = self.task_client.complete(messages_for_llm)

        result = env.evaluate(sample, response)

        # Build Episode
        ep_messages = [
            Message(role="system", content=prompt),
            Message(role="user", content=sample.question),
            Message(role="assistant", content=response),
        ]
        return Episode(
            id=Episode.new_id(),
            state_id="",
            task_id=sample.metadata.get("id", sample.question[:30]),
            bench=self.bench,
            messages=ep_messages,
            step_boundaries=[0],
            steps=[StepMeta(
                t=0,
                reward=result.score,
                done=True,
                timing_ms=0.0,
                info={"feedback": result.feedback, **result.metrics},
            )],
            summary=EpisodeSummary(
                total_reward=result.score,
                score_breakdown=result.metrics if result.metrics else None,
            ),
        )
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent.py -v`
Expected: PASS (all 5 tests)

**Step 5: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 6: Commit**

```
feat: add LfXAgent convenience wrapper with learn/ingest/save/load
```

---

## Task 10: Demo Script

**Files:**
- Create: `examples/demo_math.py`

**Step 1: Write the demo**

```python
# examples/demo_math.py
"""End-to-end demo: LfX learns to solve math problems better.

Usage:
    python examples/demo_math.py

Requires ANTHROPIC_API_KEY (or any LiteLLM-supported provider env var).
Uses Haiku for task execution, Sonnet for reflection.
"""

from __future__ import annotations

import logging
import os
import sys

# Add parent to path for local dev
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from lfx.agent import LfXAgent
from lfx.envs.math import MathEnvironment
from lfx.llm import LiteLLMClient, MockLLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("demo")


def main() -> None:
    # Check for dry-run mode (no API calls)
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        log.info("DRY RUN MODE — using mock LLM clients")
        import json
        import random

        def mock_task_response(msgs, **kw):
            # Simulate varying accuracy
            return f"The answer is \\boxed{{{random.choice(['1024', '42', '999'])}}}"

        task_client = MockLLMClient(
            responses=["\\boxed{1024}", "\\boxed{42}", "\\boxed{999}",
                       "\\boxed{5050}", "\\boxed{7}"]
        )
        reflector_client = MockLLMClient(responses=[
            json.dumps([
                {"action": "add",
                 "content": "Always show intermediate calculation steps before giving the final answer.",
                 "tags": ["strategy", "math"],
                 "target_entry_id": None,
                 "source_episode_ids": []},
            ]),
            json.dumps([
                {"action": "add",
                 "content": "For combinatorics problems, explicitly identify what is being counted and use the multiplication principle.",
                 "tags": ["strategy", "combinatorics"],
                 "target_entry_id": None,
                 "source_episode_ids": []},
            ]),
            json.dumps([]),
            json.dumps([
                {"action": "add",
                 "content": "Verify your answer by plugging it back into the original equation.",
                 "tags": ["verification"],
                 "target_entry_id": None,
                 "source_episode_ids": []},
            ]),
            json.dumps([]),
        ])
    else:
        task_model = os.environ.get("LFX_TASK_MODEL", "claude-haiku-4-5-20251001")
        reflector_model = os.environ.get("LFX_REFLECTOR_MODEL", "claude-sonnet-4-6")

        log.info("Task model: %s", task_model)
        log.info("Reflector model: %s", reflector_model)

        task_client = LiteLLMClient(model=task_model, temperature=0.3, max_tokens=500)
        reflector_client = LiteLLMClient(model=reflector_model, temperature=0.7, max_tokens=2000)

    # Create agent
    agent = LfXAgent(
        task_client=task_client,
        reflector_client=reflector_client,
        bench="math",
        base_system_prompt="You are a math problem solver. Solve the problem and give your final answer in \\boxed{} notation.",
    )

    # Create environment
    env = MathEnvironment()
    log.info("Loaded %d math problems", len(env.get_tasks()))

    # Run learning loop
    n_iterations = int(os.environ.get("LFX_ITERATIONS", "5"))
    episodes_per_iter = int(os.environ.get("LFX_EPISODES", "5"))

    log.info("Starting learning: %d iterations, %d episodes each", n_iterations, episodes_per_iter)
    log.info("---")

    results = agent.learn(
        env=env,
        iterations=n_iterations,
        episodes_per_iter=episodes_per_iter,
    )

    # Print results
    log.info("---")
    log.info("RESULTS")
    log.info("Reward curve: %s", [f"{r:.3f}" for r in results["rewards"]])
    log.info("Playbook entries: %d", results["n_entries"])
    log.info("")
    log.info("LEARNED SYSTEM PROMPT:")
    log.info(agent.get_system_prompt())

    # Save playbook
    agent.save_playbook("playbook.json")
    log.info("Playbook saved to playbook.json")


if __name__ == "__main__":
    main()
```

**Step 2: Test dry-run mode**

Run: `python examples/demo_math.py --dry-run`
Expected: Runs without errors, prints reward curve and playbook

**Step 3: Commit**

```
feat: add end-to-end math learning demo script
```

---

## Task 11: Update pyproject.toml and package exports

**Files:**
- Modify: `pyproject.toml`
- Modify: `lfx/__init__.py`
- Modify: `lfx/core/__init__.py`

**Step 1: Update pyproject.toml with litellm dependency**

```toml
[project]
dependencies = [
    "litellm>=1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
]
tau2 = [
    "tau-bench",
]
```

**Step 2: Update lfx/__init__.py**

```python
"""LfX — Learning from Experience unified learning API."""

__version__ = "0.1.0"

from lfx.agent import LfXAgent
from lfx.core.env import EvalResult, Sample, StaticTaskEnvironment
from lfx.llm import LiteLLMClient, MockLLMClient

__all__ = [
    "LfXAgent",
    "LiteLLMClient",
    "MockLLMClient",
    "EvalResult",
    "Sample",
    "StaticTaskEnvironment",
]
```

**Step 3: Run all tests**

Run: `python -m pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```
feat: update package exports and add litellm dependency
```

---

## Task 12: Final Integration Test

**Files:**
- Create: `tests/test_integration_icl.py`

**Step 1: Write full-stack integration test**

```python
# tests/test_integration_icl.py
"""Full integration test: LfXAgent + MathEnvironment + Reflector."""

import json

from lfx.agent import LfXAgent
from lfx.envs.math import MathEnvironment
from lfx.llm import MockLLMClient


class TestFullIntegration:
    def test_math_learning_loop_end_to_end(self) -> None:
        """Full loop: run math tasks, reflect, improve playbook."""
        # Task client that gets some right, some wrong
        task_responses = [
            "\\boxed{1024}",  # correct for 2^10
            "\\boxed{999}",   # wrong for sum of 1..100
            "\\boxed{42}",    # correct for f(5)
            "\\boxed{1024}",  # repeat
            "\\boxed{5050}",  # correct for sum
        ]
        # Reflector produces insights based on failures
        reflector_responses = [
            json.dumps([
                {"action": "add",
                 "content": "For summation problems, use the formula n(n+1)/2.",
                 "tags": ["formula", "math"],
                 "target_entry_id": None,
                 "source_episode_ids": []},
            ]),
            json.dumps([]),  # no new insights second round
        ]

        agent = LfXAgent(
            task_client=MockLLMClient(responses=task_responses),
            reflector_client=MockLLMClient(responses=reflector_responses),
            bench="math",
            base_system_prompt="Solve math problems. Use \\boxed{} for the answer.",
        )

        env = MathEnvironment()
        results = agent.learn(env=env, iterations=2, episodes_per_iter=2)

        # Should have reward history
        assert len(results["rewards"]) == 2
        # Playbook should have at least 1 entry
        assert results["n_entries"] >= 1
        # System prompt should include learned strategy
        prompt = agent.get_system_prompt()
        assert "n(n+1)/2" in prompt

    def test_save_load_preserves_learning(self, tmp_path) -> None:
        """Learning survives save/load cycle."""
        reflector_response = json.dumps([
            {"action": "add", "content": "Always verify",
             "tags": [], "target_entry_id": None,
             "source_episode_ids": []},
        ])
        agent = LfXAgent(
            task_client=MockLLMClient(responses=["\\boxed{42}"]),
            reflector_client=MockLLMClient(responses=[reflector_response]),
            bench="math",
        )
        env = MathEnvironment()
        agent.learn(env=env, iterations=1, episodes_per_iter=1)

        path = str(tmp_path / "pb.json")
        agent.save_playbook(path)

        agent2 = LfXAgent(
            task_client=MockLLMClient(),
            reflector_client=MockLLMClient(),
            bench="math",
        )
        agent2.load_playbook(path)
        assert "Always verify" in agent2.get_system_prompt()

    def test_ingest_external_episodes(self) -> None:
        """Level 2: learn from pre-built episodes."""
        from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta

        reflector_response = json.dumps([
            {"action": "add", "content": "Show work",
             "tags": [], "target_entry_id": None,
             "source_episode_ids": []},
        ])
        agent = LfXAgent(
            task_client=MockLLMClient(),
            reflector_client=MockLLMClient(responses=[reflector_response]),
        )
        ep = Episode(
            id="ext-1", state_id="s", task_id="t", bench="ext",
            messages=[
                Message(role="user", content="2+2?"),
                Message(role="assistant", content="5"),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=0.0, done=True, timing_ms=50)],
            summary=EpisodeSummary(total_reward=0.0),
        )
        agent.ingest([ep])
        assert "Show work" in agent.get_system_prompt()
```

**Step 2: Run all tests**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL PASS

**Step 3: Run demo dry-run**

Run: `python examples/demo_math.py --dry-run`
Expected: Runs cleanly, prints reward curve

**Step 4: Commit**

```
feat: add full integration tests for ICL learning loop
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | LLM Client | `lfx/llm.py` | 7 |
| 2 | TaskEnvironment | `lfx/core/env.py` | 7 |
| 3 | Reflector | `lfx/core/reflector.py` | 7 |
| 4 | Paradigm Breakthrough | `lfx/core/paradigm.py` | 4 |
| 5 | Adaptive Intensity | `lfx/core/intensity.py` | 8 |
| 6 | Math Environment | `lfx/envs/math.py` | 7 |
| 7 | Wire Reflector → Harness | modify `harness.py` | 6 |
| 8 | Wire Intensity → Loop | modify `loop.py` | 3 |
| 9 | LfXAgent wrapper | `lfx/agent.py` | 5 |
| 10 | Demo script | `examples/demo_math.py` | manual |
| 11 | Package config | `pyproject.toml`, `__init__.py` | — |
| 12 | Integration tests | `tests/test_integration_icl.py` | 3 |

**Total: ~57 new tests, 12 tasks, ~8 new files, ~2 modified files**

Dependencies: Tasks 1-6 are independent (can parallelize). Task 7 depends on 1+3. Task 8 depends on 5+7. Task 9 depends on all. Task 10-12 depend on 9.
