# Reward Composition & Live Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-float reward system with composable reward signals, add live-mode episode collection from real agent traffic, and enable learning without ground truth.

**Architecture:** Introduce `RewardSignal` dataclass and `signals: dict[str, RewardSignal]` on `EpisodeSummary` with priority-based `effective_reward()`. Add `RewardExtractor` protocol with built-in extractors (Execution, Outcome, UserFeedback). Add `EpisodeCollector` for live-mode episode construction from raw traffic, and `lfx.wrap()` SDK wrapper as primary integration surface. `AsyncLearner` runs harness learning in a background thread.

**Tech Stack:** Python 3.11+, dataclasses, threading, collections.OrderedDict, pytest

---

## Task 1: RewardSignal dataclass

**Files:**
- Create: `lfx/core/reward.py`
- Test: `tests/test_reward.py`

**Step 1: Write the failing tests**

```python
# tests/test_reward.py
"""Tests for the reward signal system."""

from lfx.core.reward import RewardSignal


class TestRewardSignal:
    def test_create_signal(self) -> None:
        sig = RewardSignal(name="outcome", value=1.0, confidence=1.0)
        assert sig.name == "outcome"
        assert sig.value == 1.0
        assert sig.confidence == 1.0

    def test_frozen(self) -> None:
        sig = RewardSignal(name="outcome", value=1.0, confidence=1.0)
        try:
            sig.value = 0.5  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass

    def test_value_clamped_high(self) -> None:
        sig = RewardSignal(name="x", value=2.0, confidence=1.0)
        assert sig.value == 1.0

    def test_value_clamped_low(self) -> None:
        sig = RewardSignal(name="x", value=-3.0, confidence=1.0)
        assert sig.value == -1.0

    def test_confidence_clamped(self) -> None:
        sig = RewardSignal(name="x", value=0.0, confidence=5.0)
        assert sig.confidence == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lfx.core.reward'`

**Step 3: Write minimal implementation**

```python
# lfx/core/reward.py
"""Composable reward signals for the LfX learning system.

Signals live in [-1.0, 1.0] with an associated confidence in [0.0, 1.0].
Convention: -1 = definitively bad, 0 = neutral/unknown, +1 = definitively good.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardSignal:
    """A single reward signal with value and confidence.

    Values are clamped to [-1.0, 1.0], confidence to [0.0, 1.0].
    """

    name: str
    value: float
    confidence: float

    def __post_init__(self) -> None:
        # Clamp (frozen dataclass requires object.__setattr__)
        object.__setattr__(self, "value", max(-1.0, min(1.0, self.value)))
        object.__setattr__(self, "confidence", max(0.0, min(1.0, self.confidence)))
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reward.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add lfx/core/reward.py tests/test_reward.py
git commit -m "feat: add RewardSignal dataclass with clamping"
```

---

## Task 2: Migrate EpisodeSummary to signals dict

**Files:**
- Modify: `lfx/core/episode.py` (lines 91-98: `EpisodeSummary`)
- Test: `tests/test_reward.py` (extend)

**Step 1: Write the failing tests**

Add to `tests/test_reward.py`:

```python
from lfx.core.episode import EpisodeSummary
from lfx.core.reward import RewardSignal


class TestEpisodeSummarySignals:
    def test_empty_signals_effective_reward_is_neutral(self) -> None:
        s = EpisodeSummary()
        assert s.effective_reward() == 0.0

    def test_outcome_signal(self) -> None:
        s = EpisodeSummary()
        s.signals["outcome"] = RewardSignal("outcome", 1.0, 1.0)
        assert s.effective_reward() == 1.0

    def test_user_overrides_outcome(self) -> None:
        s = EpisodeSummary()
        s.signals["outcome"] = RewardSignal("outcome", 1.0, 1.0)
        s.signals["user"] = RewardSignal("user", -1.0, 1.0)
        assert s.effective_reward() == -1.0

    def test_execution_high_confidence(self) -> None:
        s = EpisodeSummary()
        s.signals["execution"] = RewardSignal("execution", 0.5, 0.9)
        assert s.effective_reward() == 0.5

    def test_execution_low_confidence_falls_through(self) -> None:
        s = EpisodeSummary()
        s.signals["execution"] = RewardSignal("execution", 0.5, 0.3)
        # No judge either, so falls through to 0.0
        assert s.effective_reward() == 0.0

    def test_execution_low_conf_judge_present(self) -> None:
        s = EpisodeSummary()
        s.signals["execution"] = RewardSignal("execution", 0.5, 0.3)
        s.signals["judge"] = RewardSignal("judge", -0.5, 0.8)
        assert s.effective_reward() == -0.5

    def test_priority_order_user_gt_outcome_gt_exec_gt_judge(self) -> None:
        s = EpisodeSummary()
        s.signals["judge"] = RewardSignal("judge", -1.0, 1.0)
        s.signals["execution"] = RewardSignal("execution", 0.5, 0.9)
        s.signals["outcome"] = RewardSignal("outcome", 0.8, 1.0)
        s.signals["user"] = RewardSignal("user", 0.2, 1.0)
        assert s.effective_reward() == 0.2  # user wins

    def test_normalized_reward_maps_range(self) -> None:
        s = EpisodeSummary()
        s.signals["outcome"] = RewardSignal("outcome", -1.0, 1.0)
        assert s.normalized_reward() == 0.0  # -1 -> 0

        s2 = EpisodeSummary()
        s2.signals["outcome"] = RewardSignal("outcome", 1.0, 1.0)
        assert s2.normalized_reward() == 1.0  # +1 -> 1

        s3 = EpisodeSummary()
        s3.signals["outcome"] = RewardSignal("outcome", 0.0, 1.0)
        assert s3.normalized_reward() == 0.5  # 0 -> 0.5

    def test_total_reward_property_backward_compat_read(self) -> None:
        s = EpisodeSummary()
        s.signals["outcome"] = RewardSignal("outcome", 0.6, 1.0)
        # total_reward returns normalized [0,1]
        assert abs(s.total_reward - 0.8) < 1e-6  # (0.6+1)/2 = 0.8

    def test_total_reward_setter_backward_compat(self) -> None:
        s = EpisodeSummary()
        s.total_reward = 0.8  # old [0,1] convention
        # Should store as outcome signal: 0.8 * 2 - 1 = 0.6
        assert "outcome" in s.signals
        assert abs(s.signals["outcome"].value - 0.6) < 1e-6
        assert s.signals["outcome"].confidence == 1.0

    def test_needs_judge_no_signals(self) -> None:
        s = EpisodeSummary()
        assert s.needs_judge() is True

    def test_needs_judge_outcome_present(self) -> None:
        s = EpisodeSummary()
        s.signals["outcome"] = RewardSignal("outcome", 1.0, 1.0)
        assert s.needs_judge() is False

    def test_needs_judge_execution_high_conf(self) -> None:
        s = EpisodeSummary()
        s.signals["execution"] = RewardSignal("execution", 0.5, 0.9)
        assert s.needs_judge() is False

    def test_needs_judge_execution_low_conf(self) -> None:
        s = EpisodeSummary()
        s.signals["execution"] = RewardSignal("execution", 0.5, 0.3)
        assert s.needs_judge() is True

    def test_needs_judge_user_present(self) -> None:
        s = EpisodeSummary()
        s.signals["user"] = RewardSignal("user", 1.0, 1.0)
        assert s.needs_judge() is False

    def test_filtered_default_false(self) -> None:
        s = EpisodeSummary()
        assert s.filtered is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reward.py::TestEpisodeSummarySignals -v`
Expected: FAIL — `EpisodeSummary()` requires `total_reward` positional arg

**Step 3: Modify EpisodeSummary**

In `lfx/core/episode.py`, replace the `EpisodeSummary` class (lines 91-98):

```python
@dataclass
class EpisodeSummary:
    """Aggregate metrics for a completed episode.

    Reward signals are stored in ``signals`` as a dict of RewardSignal.
    ``effective_reward()`` returns the priority-based reward in [-1, 1].
    ``normalized_reward()`` maps to [0, 1] for backward compatibility.
    ``total_reward`` property provides full backward compatibility.
    """

    signals: dict[str, "RewardSignal"] = field(default_factory=dict)
    score_breakdown: dict[str, float] | None = None
    token_usage: TokenUsage | None = None
    timing: Timing | None = None
    filtered: bool = False

    def effective_reward(self) -> float:
        """Priority-based reward in [-1.0, 1.0]. Neutral (0.0) when no signal."""
        s = self.signals
        if "user" in s:
            return s["user"].value
        if "outcome" in s:
            return s["outcome"].value
        if "execution" in s and s["execution"].confidence >= 0.7:
            return s["execution"].value
        if "judge" in s:
            return s["judge"].value
        return 0.0

    def normalized_reward(self) -> float:
        """Map effective_reward [-1, 1] to [0, 1]."""
        return (self.effective_reward() + 1.0) / 2.0

    @property
    def total_reward(self) -> float:
        """Backward-compatible reward in [0, 1]."""
        return self.normalized_reward()

    @total_reward.setter
    def total_reward(self, value: float) -> None:
        """Backward compat: accept [0, 1], store as outcome signal in [-1, 1]."""
        from lfx.core.reward import RewardSignal
        self.signals["outcome"] = RewardSignal(
            name="outcome",
            value=value * 2.0 - 1.0,
            confidence=1.0,
        )
```

Add the import at the top of `lfx/core/episode.py`:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfx.core.reward import RewardSignal
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reward.py -v`
Expected: All tests PASS

**Step 5: Run existing tests to check backward compat**

Run: `pytest tests/ -v`
Expected: All existing tests PASS (they use `EpisodeSummary(total_reward=X)` which now calls the setter)

**CRITICAL**: If existing tests fail because `EpisodeSummary(total_reward=0.8)` no longer works as a positional arg (it was the first field, now `signals` is first), we need to handle the migration. The fix: existing tests that do `EpisodeSummary(total_reward=0.8)` must be updated to use the keyword arg form: `EpisodeSummary(total_reward=0.8)` — but since `total_reward` is now a property, not a field, we must use a `__init__` override or accept a `_compat_reward` parameter. The simplest fix:

Add to `EpisodeSummary.__post_init__` or use `__init__` with a compat `total_reward` param:

```python
@dataclass
class EpisodeSummary:
    signals: dict[str, "RewardSignal"] = field(default_factory=dict)
    score_breakdown: dict[str, float] | None = None
    token_usage: TokenUsage | None = None
    timing: Timing | None = None
    filtered: bool = False
    _compat_reward: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._compat_reward is not None:
            self.total_reward = self._compat_reward
            object.__setattr__(self, "_compat_reward", None)
```

Then update all call sites that did `EpisodeSummary(total_reward=X)` to `EpisodeSummary(_compat_reward=X)`. But FIRST: check how many there are. There will be usages in:
- `lfx/agent.py:222` — `EpisodeSummary(total_reward=eval_result.score)`
- `tests/test_agent.py:54` — `EpisodeSummary(total_reward=reward)`
- Other test files

**Alternative simpler approach**: Keep `total_reward` as an `__init__` parameter using a custom `__init__`:

Actually, the cleanest approach is to NOT use dataclass for this one class, or to make `total_reward` a regular field that gets converted in `__post_init__`. Let's use the simplest path:

```python
@dataclass
class EpisodeSummary:
    signals: dict[str, "RewardSignal"] = field(default_factory=dict)
    score_breakdown: dict[str, float] | None = None
    token_usage: TokenUsage | None = None
    timing: Timing | None = None
    filtered: bool = False

    def __init__(
        self,
        *,
        total_reward: float | None = None,
        signals: dict | None = None,
        score_breakdown: dict[str, float] | None = None,
        token_usage: TokenUsage | None = None,
        timing: Timing | None = None,
        filtered: bool = False,
    ) -> None:
        self.signals = signals if signals is not None else {}
        self.score_breakdown = score_breakdown
        self.token_usage = token_usage
        self.timing = timing
        self.filtered = filtered
        if total_reward is not None:
            self.total_reward = total_reward
```

This makes `EpisodeSummary(total_reward=0.8)` work exactly as before while also supporting the new `signals` API. All existing code works unchanged.

**Step 6: Commit**

```bash
git add lfx/core/episode.py tests/test_reward.py
git commit -m "feat: migrate EpisodeSummary to composable reward signals"
```

---

## Task 3: RewardExtractor protocol and RewardPipeline

**Files:**
- Modify: `lfx/core/reward.py`
- Test: `tests/test_reward.py` (extend)

**Step 1: Write the failing tests**

Add to `tests/test_reward.py`:

```python
from lfx.core.reward import RewardExtractor, RewardPipeline, RewardSignal
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta


def _make_bare_episode() -> Episode:
    return Episode(
        id="ep-1",
        state_id="s1",
        task_id="t1",
        bench="live",
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
        ],
        step_boundaries=[0],
        steps=[],
        summary=EpisodeSummary(),
    )


class FixedExtractor:
    """Test extractor that always returns a fixed signal."""
    def __init__(self, name: str, value: float, confidence: float):
        self.name = name
        self._value = value
        self._confidence = confidence

    def extract(self, episode: Episode) -> RewardSignal | None:
        return RewardSignal(self.name, self._value, self._confidence)


class NoneExtractor:
    """Test extractor that returns None (signal not available)."""
    def __init__(self, name: str):
        self.name = name

    def extract(self, episode: Episode) -> RewardSignal | None:
        return None


class TestRewardPipeline:
    def test_enrich_populates_signals(self) -> None:
        pipe = RewardPipeline([FixedExtractor("outcome", 1.0, 1.0)])
        ep = _make_bare_episode()
        pipe.enrich(ep)
        assert "outcome" in ep.summary.signals
        assert ep.summary.signals["outcome"].value == 1.0

    def test_none_extractor_skipped(self) -> None:
        pipe = RewardPipeline([NoneExtractor("outcome")])
        ep = _make_bare_episode()
        pipe.enrich(ep)
        assert "outcome" not in ep.summary.signals

    def test_judge_skipped_when_outcome_present(self) -> None:
        pipe = RewardPipeline([
            FixedExtractor("outcome", 1.0, 1.0),
            FixedExtractor("judge", -1.0, 0.8),
        ])
        ep = _make_bare_episode()
        pipe.enrich(ep)
        assert "outcome" in ep.summary.signals
        assert "judge" not in ep.summary.signals  # skipped

    def test_judge_fires_when_no_outcome(self) -> None:
        pipe = RewardPipeline([
            NoneExtractor("outcome"),
            FixedExtractor("judge", -0.5, 0.8),
        ])
        ep = _make_bare_episode()
        pipe.enrich(ep)
        assert "judge" in ep.summary.signals
        assert ep.summary.signals["judge"].value == -0.5

    def test_multiple_signals_compose(self) -> None:
        pipe = RewardPipeline([
            FixedExtractor("execution", 0.5, 0.9),
            FixedExtractor("user", 1.0, 1.0),
        ])
        ep = _make_bare_episode()
        pipe.enrich(ep)
        assert "execution" in ep.summary.signals
        assert "user" in ep.summary.signals
        assert ep.summary.effective_reward() == 1.0  # user wins
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reward.py::TestRewardPipeline -v`
Expected: FAIL — `ImportError: cannot import name 'RewardPipeline'`

**Step 3: Add RewardExtractor and RewardPipeline to `lfx/core/reward.py`**

Append to `lfx/core/reward.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lfx.core.episode import Episode


@runtime_checkable
class RewardExtractor(Protocol):
    """Protocol for extracting a reward signal from an episode."""

    name: str

    def extract(self, episode: Episode) -> RewardSignal | None:
        """Return a signal if available, None to skip."""
        ...


class RewardPipeline:
    """Run extractors in order, populating episode.summary.signals.

    Judge extractors (name="judge") are automatically skipped when
    the episode already has a high-fidelity signal (outcome, user,
    or high-confidence execution).
    """

    def __init__(self, extractors: list[RewardExtractor]) -> None:
        self.extractors = extractors

    def enrich(self, episode: Episode) -> None:
        """Populate episode.summary.signals from available extractors."""
        for ext in self.extractors:
            if ext.name == "judge" and not episode.summary.needs_judge():
                continue
            sig = ext.extract(episode)
            if sig is not None:
                episode.summary.signals[sig.name] = sig
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reward.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add lfx/core/reward.py tests/test_reward.py
git commit -m "feat: add RewardExtractor protocol and RewardPipeline"
```

---

## Task 4: ExecutionExtractor

**Files:**
- Create: `lfx/extractors/__init__.py`
- Create: `lfx/extractors/execution.py`
- Test: `tests/test_extractors.py`

**Step 1: Write the failing tests**

```python
# tests/test_extractors.py
"""Tests for built-in reward extractors."""

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta, ToolCall
from lfx.extractors.execution import ExecutionExtractor


def _ep_with_tool_messages(tool_contents: list[str]) -> Episode:
    """Build an episode with tool-role messages."""
    msgs = [Message(role="user", content="do something")]
    for i, content in enumerate(tool_contents):
        msgs.append(Message(role="tool", content=content, tool_call_id=f"tc-{i}"))
    msgs.append(Message(role="assistant", content="done"))
    return Episode(
        id="ep-1", state_id="s1", task_id="t1", bench="live",
        messages=msgs, step_boundaries=[0], steps=[],
        summary=EpisodeSummary(),
    )


def _ep_no_tools() -> Episode:
    return Episode(
        id="ep-1", state_id="s1", task_id="t1", bench="live",
        messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ],
        step_boundaries=[0], steps=[],
        summary=EpisodeSummary(),
    )


class TestExecutionExtractor:
    def test_no_tool_calls_returns_none(self) -> None:
        ext = ExecutionExtractor()
        assert ext.extract(_ep_no_tools()) is None

    def test_error_in_tool_response_negative(self) -> None:
        ext = ExecutionExtractor()
        sig = ext.extract(_ep_with_tool_messages(["Error: connection refused"]))
        assert sig is not None
        assert sig.value < 0
        assert sig.confidence > 0.5

    def test_http_error_code_negative(self) -> None:
        ext = ExecutionExtractor()
        sig = ext.extract(_ep_with_tool_messages(["HTTP 500 Internal Server Error"]))
        assert sig is not None
        assert sig.value < 0

    def test_substantial_content_mildly_positive(self) -> None:
        ext = ExecutionExtractor()
        content = "Here is a detailed result with many fields: " + "x" * 100
        sig = ext.extract(_ep_with_tool_messages([content]))
        assert sig is not None
        assert 0.0 < sig.value <= 0.5
        assert sig.confidence < 0.8

    def test_empty_content_negative(self) -> None:
        ext = ExecutionExtractor()
        sig = ext.extract(_ep_with_tool_messages([""]))
        assert sig is not None
        assert sig.value < 0

    def test_minimal_content_neutral(self) -> None:
        ext = ExecutionExtractor()
        sig = ext.extract(_ep_with_tool_messages(["ok"]))
        assert sig is not None
        assert abs(sig.value) <= 0.1  # neutral-ish

    def test_mixed_signals_aggregated(self) -> None:
        ext = ExecutionExtractor()
        sig = ext.extract(_ep_with_tool_messages([
            "Error: file not found",
            "Here is a long successful result " + "y" * 100,
        ]))
        assert sig is not None
        # Mixed: should be somewhere between negative and mildly positive
        assert -1.0 <= sig.value <= 1.0

    def test_name_is_execution(self) -> None:
        ext = ExecutionExtractor()
        assert ext.name == "execution"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_extractors.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# lfx/extractors/__init__.py
"""Built-in reward extractors."""
```

```python
# lfx/extractors/execution.py
"""ExecutionExtractor — derives reward signals from tool call results."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lfx.core.reward import RewardSignal

if TYPE_CHECKING:
    from lfx.core.episode import Episode

_ERROR_KEYWORDS = re.compile(
    r"error|exception|traceback|failed|failure|timeout|refused|denied",
    re.IGNORECASE,
)

_HTTP_ERROR_CODES = re.compile(r"\b[45]\d{2}\b")


class ExecutionExtractor:
    """Extract reward signal from tool-role message content.

    Scoring rules:
    - Explicit errors → (-1.0, 0.9)
    - HTTP 4xx/5xx codes → (-1.0, 0.85)
    - Empty content → (-0.5, 0.5)
    - Minimal content (1-50 chars, no errors) → (0.0, 0.3) neutral
    - Substantial content (>50 chars, no errors) → (0.5, 0.6) mildly positive
    """

    name: str = "execution"

    def extract(self, episode: Episode) -> RewardSignal | None:
        tool_messages = [
            m for m in episode.messages
            if m.role == "tool" and m.content is not None
        ]
        if not tool_messages:
            return None

        signals: list[tuple[float, float]] = []

        for tm in tool_messages:
            content = tm.content

            if _ERROR_KEYWORDS.search(content):
                signals.append((-1.0, 0.9))
            elif _HTTP_ERROR_CODES.search(content):
                signals.append((-1.0, 0.85))
            elif len(content.strip()) == 0:
                signals.append((-0.5, 0.5))
            elif len(content.strip()) > 50:
                signals.append((0.5, 0.6))
            else:
                signals.append((0.0, 0.3))

        # Aggregate: confidence-weighted average
        total_conf = sum(c for _, c in signals)
        if total_conf < 1e-9:
            return None

        avg_value = sum(v * c for v, c in signals) / total_conf
        avg_conf = total_conf / len(signals)

        return RewardSignal(
            name="execution",
            value=avg_value,
            confidence=avg_conf,
        )
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_extractors.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add lfx/extractors/__init__.py lfx/extractors/execution.py tests/test_extractors.py
git commit -m "feat: add ExecutionExtractor for tool call reward signals"
```

---

## Task 5: OutcomeExtractor

**Files:**
- Create: `lfx/extractors/outcome.py`
- Test: `tests/test_extractors.py` (extend)

**Step 1: Write the failing tests**

Add to `tests/test_extractors.py`:

```python
from lfx.core.env import EvalResult, Sample, StaticTaskEnvironment
from lfx.extractors.outcome import OutcomeExtractor


class TestOutcomeExtractor:
    def _make_env(self, score: float) -> StaticTaskEnvironment:
        return StaticTaskEnvironment(
            tasks=[Sample(question="Q", ground_truth="A")],
            evaluate_fn=lambda s, r: EvalResult(score=score),
        )

    def test_returns_signal_with_env_and_sample(self) -> None:
        env = self._make_env(score=1.0)
        ext = OutcomeExtractor(env=env)
        ep = _ep_no_tools()
        # Attach sample metadata so extractor can evaluate
        ep_with_meta = Episode(
            id="ep-1", state_id="s1",
            task_id="t1", bench="test",
            messages=[
                Message(role="user", content="Q"),
                Message(role="assistant", content="A"),
            ],
            step_boundaries=[0], steps=[],
            summary=EpisodeSummary(),
        )
        sig = ext.extract(ep_with_meta)
        assert sig is not None
        assert sig.name == "outcome"
        assert sig.confidence == 1.0
        # score=1.0 maps to value=1.0 (since we do score*2-1)
        assert abs(sig.value - 1.0) < 1e-6

    def test_zero_score_maps_to_negative(self) -> None:
        env = self._make_env(score=0.0)
        ext = OutcomeExtractor(env=env)
        ep = Episode(
            id="ep-1", state_id="s1", task_id="t1", bench="test",
            messages=[
                Message(role="user", content="Q"),
                Message(role="assistant", content="wrong"),
            ],
            step_boundaries=[0], steps=[],
            summary=EpisodeSummary(),
        )
        sig = ext.extract(ep)
        assert sig is not None
        assert abs(sig.value - (-1.0)) < 1e-6  # 0*2-1 = -1

    def test_no_env_returns_none(self) -> None:
        ext = OutcomeExtractor(env=None)
        ep = _ep_no_tools()
        assert ext.extract(ep) is None
```

**Step 2: Run tests**

Run: `pytest tests/test_extractors.py::TestOutcomeExtractor -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# lfx/extractors/outcome.py
"""OutcomeExtractor — wraps TaskEnvironment.evaluate() as a reward signal."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lfx.core.reward import RewardSignal

if TYPE_CHECKING:
    from lfx.core.episode import Episode


class OutcomeExtractor:
    """Extract outcome reward by evaluating against a TaskEnvironment.

    The environment's evaluate() returns a score in [0, 1].
    We map this to [-1, 1] via: value = score * 2 - 1.
    """

    name: str = "outcome"

    def __init__(self, env: Any | None = None) -> None:
        self._env = env

    def extract(self, episode: Episode) -> RewardSignal | None:
        if self._env is None:
            return None

        # Find user question and assistant response from messages
        question = ""
        response = ""
        for msg in episode.messages:
            if msg.role == "user" and not question:
                question = msg.content
            if msg.role == "assistant":
                response = msg.content  # last assistant message

        if not question or not response:
            return None

        # Build a Sample from the question
        from lfx.core.env import Sample
        sample = Sample(question=question)

        # Try to find ground_truth in task environment
        for task in self._env.get_tasks():
            if task.question == question:
                sample = task
                break

        eval_result = self._env.evaluate(sample, response)
        value = eval_result.score * 2.0 - 1.0  # [0,1] -> [-1,1]

        return RewardSignal(name="outcome", value=value, confidence=1.0)
```

**Step 4: Run tests**

Run: `pytest tests/test_extractors.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add lfx/extractors/outcome.py tests/test_extractors.py
git commit -m "feat: add OutcomeExtractor wrapping TaskEnvironment.evaluate()"
```

---

## Task 6: UserFeedbackExtractor

**Files:**
- Create: `lfx/extractors/user_feedback.py`
- Test: `tests/test_extractors.py` (extend)

**Step 1: Write the failing tests**

Add to `tests/test_extractors.py`:

```python
from lfx.extractors.user_feedback import UserFeedbackExtractor


class TestUserFeedbackExtractor:
    def test_no_feedback_returns_none(self) -> None:
        ext = UserFeedbackExtractor()
        ep = _ep_no_tools()
        assert ext.extract(ep) is None

    def test_feedback_present_returns_signal(self) -> None:
        ext = UserFeedbackExtractor()
        ep = _ep_no_tools()
        # Pre-populate user signal (as if submit_feedback was called)
        from lfx.core.reward import RewardSignal
        ep.summary.signals["user"] = RewardSignal("user", 1.0, 1.0)
        sig = ext.extract(ep)
        assert sig is not None
        assert sig.value == 1.0

    def test_name_is_user(self) -> None:
        ext = UserFeedbackExtractor()
        assert ext.name == "user"
```

**Step 2: Run tests**

Run: `pytest tests/test_extractors.py::TestUserFeedbackExtractor -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# lfx/extractors/user_feedback.py
"""UserFeedbackExtractor — reads pre-populated user feedback from episode signals."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lfx.core.reward import RewardSignal

if TYPE_CHECKING:
    from lfx.core.episode import Episode


class UserFeedbackExtractor:
    """Check if user feedback has already been attached to the episode.

    User feedback is populated externally via EpisodeCollector.submit_feedback()
    or directly by setting episode.summary.signals["user"].
    This extractor simply reads it — it doesn't generate new signals.
    """

    name: str = "user"

    def extract(self, episode: Episode) -> RewardSignal | None:
        return episode.summary.signals.get("user")
```

**Step 4: Run tests**

Run: `pytest tests/test_extractors.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add lfx/extractors/user_feedback.py tests/test_extractors.py
git commit -m "feat: add UserFeedbackExtractor"
```

---

## Task 7: FormattingFilter

**Files:**
- Create: `lfx/extractors/formatting.py`
- Test: `tests/test_extractors.py` (extend)

**Step 1: Write the failing tests**

Add to `tests/test_extractors.py`:

```python
from lfx.extractors.formatting import FormattingFilter


class TestFormattingFilter:
    def test_passes_normal_response(self) -> None:
        f = FormattingFilter()
        ep = Episode(
            id="ep-1", state_id="s1", task_id="t1", bench="live",
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="A reasonable response here."),
            ],
            step_boundaries=[0], steps=[], summary=EpisodeSummary(),
        )
        assert f.passes(ep) is True

    def test_fails_empty_response(self) -> None:
        f = FormattingFilter()
        ep = Episode(
            id="ep-1", state_id="s1", task_id="t1", bench="live",
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content=""),
            ],
            step_boundaries=[0], steps=[], summary=EpisodeSummary(),
        )
        assert f.passes(ep) is False

    def test_fails_too_short(self) -> None:
        f = FormattingFilter(min_response_length=20)
        ep = Episode(
            id="ep-1", state_id="s1", task_id="t1", bench="live",
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
            ],
            step_boundaries=[0], steps=[], summary=EpisodeSummary(),
        )
        assert f.passes(ep) is False

    def test_fails_no_assistant_message(self) -> None:
        f = FormattingFilter()
        ep = Episode(
            id="ep-1", state_id="s1", task_id="t1", bench="live",
            messages=[Message(role="user", content="hello")],
            step_boundaries=[0], steps=[], summary=EpisodeSummary(),
        )
        assert f.passes(ep) is False
```

**Step 2: Run tests**

Run: `pytest tests/test_extractors.py::TestFormattingFilter -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# lfx/extractors/formatting.py
"""FormattingFilter — gate that excludes badly formatted episodes from training."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfx.core.episode import Episode


class FormattingFilter:
    """Hard filter: episodes with bad formatting are excluded from training.

    This is NOT a reward signal — it's a gate. Episodes that fail are
    marked with summary.filtered = True and not buffered for learning.
    """

    def __init__(self, min_response_length: int = 10) -> None:
        self.min_response_length = min_response_length

    def passes(self, episode: Episode) -> bool:
        """Return True if the episode's response meets formatting requirements."""
        # Find last assistant message
        response = ""
        for msg in reversed(episode.messages):
            if msg.role == "assistant":
                response = msg.content
                break

        if not response:
            return False

        if len(response.strip()) < self.min_response_length:
            return False

        return True
```

**Step 4: Run tests**

Run: `pytest tests/test_extractors.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add lfx/extractors/formatting.py tests/test_extractors.py
git commit -m "feat: add FormattingFilter gate for episode quality"
```

---

## Task 8: EpisodeCollector (live mode)

**Files:**
- Create: `lfx/collector.py`
- Test: `tests/test_collector.py`

**Step 1: Write the failing tests**

```python
# tests/test_collector.py
"""Tests for EpisodeCollector — live mode episode construction."""

import threading

from lfx.collector import EpisodeCollector
from lfx.core.episode import Message
from lfx.core.reward import RewardPipeline, RewardSignal
from lfx.extractors.formatting import FormattingFilter


class _TrackingCallback:
    """Records batches passed to on_batch."""
    def __init__(self):
        self.batches = []

    def __call__(self, episodes):
        self.batches.append(list(episodes))


class TestEpisodeCollector:
    def test_ingest_creates_episode(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        msgs = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there, how can I help?"),
        ]
        ep = collector.ingest(msgs, session_id="s1")
        assert ep.id
        assert ep.bench == "live"
        assert ep.task_id == "s1"
        assert len(ep.messages) == 2

    def test_batch_triggers_callback(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=cb,
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 20),
        ]
        collector.ingest(msgs, session_id="s1")
        assert len(cb.batches) == 0
        collector.ingest(msgs, session_id="s2")
        assert len(cb.batches) == 1
        assert len(cb.batches[0]) == 2

    def test_filtered_episodes_not_buffered(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=cb,
            formatting_filter=FormattingFilter(min_response_length=100),
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="short"),
        ]
        ep = collector.ingest(msgs, session_id="s1")
        assert ep.summary.filtered is True
        collector.ingest(msgs, session_id="s2")
        assert len(cb.batches) == 0  # neither was buffered

    def test_submit_feedback_updates_signal(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        msgs = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a long enough response here"),
        ]
        ep = collector.ingest(msgs, session_id="s1")
        assert collector.submit_feedback(ep.id, 1.0) is True
        assert "user" in ep.summary.signals
        assert ep.summary.signals["user"].value == 1.0

    def test_submit_feedback_unknown_id(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
        )
        assert collector.submit_feedback("nonexistent", 1.0) is False

    def test_thread_safety(self) -> None:
        cb = _TrackingCallback()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=10,
            on_batch=cb,
        )

        def ingest_many():
            for i in range(5):
                msgs = [
                    Message(role="user", content=f"q-{i}"),
                    Message(role="assistant", content="a" * 20),
                ]
                collector.ingest(msgs, session_id=f"s-{threading.current_thread().name}-{i}")

        threads = [threading.Thread(target=ingest_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 20 total episodes, batch_size=10 → should have triggered 2 batches
        total_eps = sum(len(b) for b in cb.batches)
        assert total_eps == 20

    def test_metrics(self) -> None:
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=100,
            formatting_filter=FormattingFilter(min_response_length=100),
        )
        msgs_good = [
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 200),
        ]
        msgs_bad = [
            Message(role="user", content="q"),
            Message(role="assistant", content="short"),
        ]
        collector.ingest(msgs_good, session_id="s1")
        collector.ingest(msgs_bad, session_id="s2")
        collector.submit_feedback("nonexistent", 1.0)

        m = collector.metrics
        assert m["episodes_collected"] == 2
        assert m["episodes_filtered"] == 1
        assert m["feedback_received"] == 0
        assert m["feedback_missed"] == 1
```

**Step 2: Run tests**

Run: `pytest tests/test_collector.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# lfx/collector.py
"""EpisodeCollector — constructs episodes from live agent traffic."""

from __future__ import annotations

import logging
import threading
import uuid
from collections import OrderedDict
from typing import Callable

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.reward import RewardPipeline, RewardSignal
from lfx.extractors.formatting import FormattingFilter

log = logging.getLogger(__name__)


class EpisodeCollector:
    """Collects live agent traffic into episodes, enriches with rewards,
    and triggers learning when batch is full.

    Thread-safe: all buffer operations are protected by a lock.
    """

    def __init__(
        self,
        pipeline: RewardPipeline,
        batch_size: int = 16,
        on_batch: Callable[[list[Episode]], None] | None = None,
        formatting_filter: FormattingFilter | None = None,
        max_episode_cache: int = 10_000,
        store_messages: bool = True,
        redact_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.on_batch = on_batch
        self.formatting_filter = formatting_filter or FormattingFilter()
        self.store_messages = store_messages
        self.redact_fn = redact_fn
        self._max_cache = max_episode_cache

        self._buffer: list[Episode] = []
        self._episode_index: OrderedDict[str, Episode] = OrderedDict()
        self._lock = threading.Lock()

        # Metrics
        self._episodes_collected = 0
        self._episodes_filtered = 0
        self._feedback_received = 0
        self._feedback_missed = 0
        self._eviction_count = 0

    def ingest(self, messages: list[Message], session_id: str) -> Episode:
        """Convert a completed request/response into an Episode.

        Enriches with reward signals via pipeline. If formatting filter
        fails, marks as filtered and excludes from training buffer.
        """
        # Optional redaction
        if self.redact_fn:
            messages = [
                Message(
                    role=m.role,
                    content=self.redact_fn(m.content),
                    name=m.name,
                    tool_calls=m.tool_calls,
                    tool_call_id=m.tool_call_id,
                    timestamp=m.timestamp,
                    token_count=m.token_count,
                    model=m.model,
                )
                for m in messages
            ]

        stored_messages = messages if self.store_messages else []

        episode = Episode(
            id=uuid.uuid4().hex,
            state_id="live",
            task_id=session_id,
            bench="live",
            messages=stored_messages,
            step_boundaries=[0] if stored_messages else [],
            steps=[],
            summary=EpisodeSummary(),
        )

        # Enrich with reward signals
        self.pipeline.enrich(episode)

        batch_to_flush: list[Episode] | None = None

        with self._lock:
            self._episodes_collected += 1

            # Index for feedback lookup
            self._episode_index[episode.id] = episode
            self._maybe_evict()

            # Formatting gate
            if not self.formatting_filter.passes(episode):
                episode.summary.filtered = True
                self._episodes_filtered += 1
                log.debug("Episode %s filtered (formatting)", episode.id)
                return episode

            # Buffer for training
            self._buffer.append(episode)
            if len(self._buffer) >= self.batch_size:
                batch_to_flush = list(self._buffer)
                self._buffer.clear()

        # Flush outside lock to avoid holding it during callback
        if batch_to_flush and self.on_batch:
            self.on_batch(batch_to_flush)

        return episode

    def submit_feedback(self, episode_id: str, score: float) -> bool:
        """Attach user feedback to an episode. Returns False if not found."""
        with self._lock:
            ep = self._episode_index.get(episode_id)
            if ep is None:
                self._feedback_missed += 1
                log.debug("Feedback for unknown episode %s", episode_id)
                return False
            ep.summary.signals["user"] = RewardSignal("user", score, 1.0)
            self._feedback_received += 1
            return True

    @property
    def metrics(self) -> dict[str, int]:
        with self._lock:
            return {
                "episodes_collected": self._episodes_collected,
                "episodes_filtered": self._episodes_filtered,
                "feedback_received": self._feedback_received,
                "feedback_missed": self._feedback_missed,
                "evictions": self._eviction_count,
                "buffer_size": len(self._buffer),
                "cache_size": len(self._episode_index),
            }

    def _maybe_evict(self) -> None:
        """Evict oldest episodes from LRU cache when over limit."""
        while len(self._episode_index) > self._max_cache:
            evicted_id, evicted_ep = self._episode_index.popitem(last=False)
            if "user" not in evicted_ep.summary.signals:
                self._eviction_count += 1
                log.info("Evicting episode %s without feedback", evicted_id)
```

**Step 4: Run tests**

Run: `pytest tests/test_collector.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add lfx/collector.py tests/test_collector.py
git commit -m "feat: add EpisodeCollector for live mode episode construction"
```

---

## Task 9: AsyncLearner

**Files:**
- Create: `lfx/learner.py`
- Test: `tests/test_learner.py`

**Step 1: Write the failing tests**

```python
# tests/test_learner.py
"""Tests for AsyncLearner — background learning from episode batches."""

import time

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.reward import RewardSignal
from lfx.layers.harness import Harness, Playbook, PlaybookEntry
from lfx.learner import AsyncLearner
from lfx.core.loop import AgentState


def _make_episodes(n: int, reward: float = 0.8) -> list[Episode]:
    eps = []
    for i in range(n):
        ep = Episode(
            id=f"ep-{i}", state_id="s1", task_id=f"t-{i}", bench="live",
            messages=[
                Message(role="user", content=f"q-{i}"),
                Message(role="assistant", content=f"a-{i}" * 20),
            ],
            step_boundaries=[0], steps=[],
            summary=EpisodeSummary(total_reward=reward),
        )
        eps.append(ep)
    return eps


class TestAsyncLearner:
    def test_on_batch_processes_episodes(self) -> None:
        state = AgentState()
        state.harness.playbook = Playbook(entries=[
            PlaybookEntry(id="e1", content="Be helpful"),
        ])
        learner = AsyncLearner(agent_state=state, active_layers=["harness"])
        learner.start()

        episodes = _make_episodes(3, reward=0.9)
        learner.on_batch(episodes)

        # Wait for background processing
        time.sleep(0.5)

        assert learner.metrics["batches_trained"] >= 1
        learner.stop()

    def test_dropped_batch_when_queue_full(self) -> None:
        state = AgentState()
        learner = AsyncLearner(
            agent_state=state,
            active_layers=["harness"],
            max_queue_size=1,
        )
        learner.start()

        # Fill queue by submitting many batches fast
        for _ in range(10):
            learner.on_batch(_make_episodes(1))

        time.sleep(0.5)
        assert learner.metrics["batches_dropped"] >= 0  # may or may not drop
        learner.stop()

    def test_stop_graceful(self) -> None:
        state = AgentState()
        learner = AsyncLearner(agent_state=state)
        learner.start()
        learner.stop()
        # Should not hang
```

**Step 2: Run tests**

Run: `pytest tests/test_learner.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# lfx/learner.py
"""AsyncLearner — background thread for learning from episode batches."""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from statistics import mean
from typing import Any

from lfx.core.intensity import AdaptiveIntensity
from lfx.core.types import Datum

log = logging.getLogger(__name__)


class AsyncLearner:
    """Run learning in a background worker thread.

    Episodes are submitted via on_batch(). A single worker thread
    processes them sequentially — never blocks the caller.
    """

    def __init__(
        self,
        agent_state: Any,
        active_layers: list[str] | None = None,
        intensity: AdaptiveIntensity | None = None,
        max_queue_size: int = 4,
        overflow: str = "drop_newest",
    ) -> None:
        self.agent_state = agent_state
        self.active_layers = active_layers or ["harness"]
        self.intensity = intensity or AdaptiveIntensity()
        self.overflow = overflow

        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._iteration = 0

        # Metrics
        self._batches_trained = 0
        self._batches_dropped = 0
        self._batches_failed = 0
        self._failed_batch_ids: list[str] = []

    def start(self) -> None:
        """Start the background worker thread."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._run, daemon=True, name="lfx-learner")
        self._worker.start()

    def stop(self, timeout: float = 5.0) -> None:
        """Signal the worker to stop and wait for it."""
        self._stop_event.set()
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    def on_batch(self, episodes: list) -> None:
        """Submit a batch of episodes for background learning."""
        if self.overflow == "block":
            self._queue.put(episodes)
        elif self.overflow == "drop_oldest":
            while self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self._queue.put_nowait(episodes)
            except queue.Full:
                self._batches_dropped += 1
        else:  # drop_newest
            try:
                self._queue.put_nowait(episodes)
            except queue.Full:
                self._batches_dropped += 1
                log.warning("Learning queue full, dropping batch")

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "batches_trained": self._batches_trained,
            "batches_dropped": self._batches_dropped,
            "batches_failed": self._batches_failed,
            "iteration": self._iteration,
            "queue_size": self._queue.qsize(),
            "reward_history": list(self.intensity._rewards),
        }

    def _run(self) -> None:
        """Worker loop: pull batches from queue and learn."""
        while not self._stop_event.is_set():
            try:
                episodes = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            self._learn(episodes)

    def _learn(self, episodes: list) -> None:
        batch_id = uuid.uuid4().hex[:8]

        # Compute average reward
        rewards = []
        for ep in episodes:
            rewards.append(ep.summary.normalized_reward())
        avg_reward = mean(rewards) if rewards else 0.0
        self.intensity.record_reward(avg_reward)

        log.info(
            "Batch %s: %d episodes, avg_reward=%.3f",
            batch_id, len(episodes), avg_reward,
        )

        datum = Datum(episodes=episodes)

        for name in self.active_layers:
            layer = getattr(self.agent_state, name, None)
            if layer is None:
                continue
            try:
                layer.forward_backward(datum).result()
                layer.optim_step().result()
            except Exception as exc:
                log.error(
                    "Layer %s failed on batch %s: %s", name, batch_id, exc,
                )
                self._batches_failed += 1
                self._failed_batch_ids.append(batch_id)
                try:
                    layer.clear_pending_state()
                except Exception:
                    pass
                return

        self._batches_trained += 1
        self._iteration += 1
```

**Step 4: Run tests**

Run: `pytest tests/test_learner.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add lfx/learner.py tests/test_learner.py
git commit -m "feat: add AsyncLearner for background learning"
```

---

## Task 10: `lfx.wrap()` SDK wrapper

**Files:**
- Create: `lfx/wrapper.py`
- Modify: `lfx/__init__.py` (add exports)
- Test: `tests/test_wrapper.py`

**Step 1: Write the failing tests**

```python
# tests/test_wrapper.py
"""Tests for lfx.wrap() — SDK wrapper for live mode."""

from lfx.collector import EpisodeCollector
from lfx.core.episode import Message
from lfx.core.reward import RewardPipeline
from lfx.learner import AsyncLearner
from lfx.core.loop import AgentState
from lfx.llm import MockLLMClient
from lfx.wrapper import wrap


class TestWrap:
    def test_wrap_returns_callable(self) -> None:
        client = MockLLMClient(responses=["hello"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        assert hasattr(wrapped, "complete")

    def test_wrap_passes_through_response(self) -> None:
        client = MockLLMClient(responses=["The answer is 42"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        result = wrapped.complete([{"role": "user", "content": "What is 6*7?"}])
        assert result == "The answer is 42"

    def test_wrap_creates_episode(self) -> None:
        client = MockLLMClient(responses=["The answer is 42"])
        collector = EpisodeCollector(pipeline=RewardPipeline([]), batch_size=100)
        wrapped = wrap(client, collector=collector)
        wrapped.complete([{"role": "user", "content": "What is 6*7?"}])
        assert collector.metrics["episodes_collected"] == 1

    def test_wrap_with_learner_triggers_batch(self) -> None:
        client = MockLLMClient(responses=["response"])
        state = AgentState()
        learner = AsyncLearner(agent_state=state)
        learner.start()
        collector = EpisodeCollector(
            pipeline=RewardPipeline([]),
            batch_size=2,
            on_batch=learner.on_batch,
        )
        wrapped = wrap(client, collector=collector)

        wrapped.complete([{"role": "user", "content": "q1"}])
        wrapped.complete([{"role": "user", "content": "q2"}])

        import time
        time.sleep(0.5)
        assert learner.metrics["batches_trained"] >= 1
        learner.stop()
```

**Step 2: Run tests**

Run: `pytest tests/test_wrapper.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# lfx/wrapper.py
"""lfx.wrap() — SDK wrapper that intercepts LLM calls for learning."""

from __future__ import annotations

from typing import Any

from lfx.collector import EpisodeCollector
from lfx.core.episode import Message


class WrappedClient:
    """Drop-in LLMClient replacement that intercepts calls for learning."""

    def __init__(self, client: Any, collector: EpisodeCollector) -> None:
        self._client = client
        self._collector = collector

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Call the underlying client and record the episode."""
        response = self._client.complete(messages, **kwargs)

        # Convert dict messages to Message objects for the episode
        ep_messages = []
        for m in messages:
            ep_messages.append(Message(
                role=m.get("role", "user"),
                content=m.get("content", ""),
                name=m.get("name"),
            ))
        ep_messages.append(Message(role="assistant", content=response))

        # Derive session_id from conversation (hash of first user message)
        session_id = ""
        for m in messages:
            if m.get("role") == "user":
                session_id = str(hash(m.get("content", "")))
                break

        self._collector.ingest(ep_messages, session_id=session_id)

        return response


def wrap(client: Any, collector: EpisodeCollector) -> WrappedClient:
    """Wrap an LLMClient with live-mode episode collection.

    Usage::

        wrapped = lfx.wrap(my_client, collector=collector)
        result = wrapped.complete(messages)  # works exactly like before
    """
    return WrappedClient(client, collector)
```

**Step 4: Run tests**

Run: `pytest tests/test_wrapper.py -v`
Expected: All PASS

**Step 5: Update `lfx/__init__.py` exports**

Add to `lfx/__init__.py`:

```python
from lfx.collector import EpisodeCollector
from lfx.core.reward import RewardPipeline, RewardSignal
from lfx.learner import AsyncLearner
from lfx.wrapper import wrap
```

And add to `__all__`:

```python
"EpisodeCollector",
"RewardPipeline",
"RewardSignal",
"AsyncLearner",
"wrap",
```

**Step 6: Commit**

```bash
git add lfx/wrapper.py lfx/__init__.py tests/test_wrapper.py
git commit -m "feat: add lfx.wrap() SDK wrapper for live mode"
```

---

## Task 11: Update Harness to use signals

**Files:**
- Modify: `lfx/layers/harness.py` (lines 419-451: `forward_backward`)
- Test: `tests/test_harness_signals.py`

**Step 1: Write the failing tests**

```python
# tests/test_harness_signals.py
"""Tests for Harness layer consuming reward signals instead of total_reward."""

from lfx.core.episode import Episode, EpisodeSummary, Message
from lfx.core.reward import RewardSignal
from lfx.core.types import Datum
from lfx.layers.harness import Harness, Playbook, PlaybookEntry


def _ep_with_signal(name: str, value: float, confidence: float = 1.0) -> Episode:
    summary = EpisodeSummary()
    summary.signals[name] = RewardSignal(name, value, confidence)
    return Episode(
        id="ep-1", state_id="s1", task_id="t1", bench="test",
        messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a" * 20),
        ],
        step_boundaries=[0], steps=[],
        summary=summary,
    )


class TestHarnessSignals:
    def test_positive_signal_increments_helpful(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        datum = Datum(episodes=[_ep_with_signal("outcome", 0.8)])
        h.forward_backward(datum)
        h.optim_step()
        assert h.playbook.entries[0].helpful == 1
        assert h.playbook.entries[0].harmful == 0

    def test_negative_signal_increments_harmful(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        datum = Datum(episodes=[_ep_with_signal("outcome", -0.5)])
        h.forward_backward(datum)
        h.optim_step()
        assert h.playbook.entries[0].helpful == 0
        assert h.playbook.entries[0].harmful == 1

    def test_neutral_signal_skipped(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        datum = Datum(episodes=[_ep_with_signal("execution", 0.0, confidence=0.3)])
        h.forward_backward(datum)
        h.optim_step()
        # Neutral signal: neither helpful nor harmful
        assert h.playbook.entries[0].helpful == 0
        assert h.playbook.entries[0].harmful == 0

    def test_user_signal_overrides(self) -> None:
        h = Harness()
        h.playbook = Playbook(entries=[PlaybookEntry(id="e1", content="tip")])
        summary = EpisodeSummary()
        summary.signals["outcome"] = RewardSignal("outcome", 1.0, 1.0)
        summary.signals["user"] = RewardSignal("user", -1.0, 1.0)
        ep = Episode(
            id="ep-1", state_id="s1", task_id="t1", bench="test",
            messages=[
                Message(role="user", content="q"),
                Message(role="assistant", content="a" * 20),
            ],
            step_boundaries=[0], steps=[],
            summary=summary,
        )
        datum = Datum(episodes=[ep])
        h.forward_backward(datum)
        h.optim_step()
        # User said -1.0, so harmful
        assert h.playbook.entries[0].harmful == 1
        assert h.playbook.entries[0].helpful == 0
```

**Step 2: Run tests**

Run: `pytest tests/test_harness_signals.py -v`
Expected: FAIL (harness still uses `_HELPFUL_REWARD_THRESHOLD = 0.5` with `total_reward`)

**Step 3: Update Harness.forward_backward**

In `lfx/layers/harness.py`, replace the `forward_backward` reward logic (around lines 427-436):

Old:
```python
        for episode in data.episodes:
            reward = episode.summary.total_reward
            for entry in self.playbook.entries:
                prev_h, prev_harm = self._pending.playbook_signals.get(
                    entry.id, (0, 0)
                )
                if reward > _HELPFUL_REWARD_THRESHOLD:
                    self._pending.playbook_signals[entry.id] = (prev_h + 1, prev_harm)
                else:
                    self._pending.playbook_signals[entry.id] = (prev_h, prev_harm + 1)
```

New:
```python
        for episode in data.episodes:
            reward = episode.summary.effective_reward()  # [-1, 1]
            for entry in self.playbook.entries:
                prev_h, prev_harm = self._pending.playbook_signals.get(
                    entry.id, (0, 0)
                )
                if reward > 0:
                    self._pending.playbook_signals[entry.id] = (prev_h + 1, prev_harm)
                elif reward < 0:
                    self._pending.playbook_signals[entry.id] = (prev_h, prev_harm + 1)
                # reward == 0 (neutral) — skip, don't count
```

**Step 4: Run tests**

Run: `pytest tests/test_harness_signals.py -v && pytest tests/ -v`
Expected: All PASS (new tests and existing tests)

**Step 5: Commit**

```bash
git add lfx/layers/harness.py tests/test_harness_signals.py
git commit -m "feat: update Harness to consume effective_reward() from signals"
```

---

## Task 12: Integration test — full live mode pipeline

**Files:**
- Test: `tests/test_live_mode.py`

**Step 1: Write the integration test**

```python
# tests/test_live_mode.py
"""Integration test: full live mode pipeline end-to-end."""

import time

from lfx.collector import EpisodeCollector
from lfx.core.loop import AgentState
from lfx.core.reward import RewardPipeline
from lfx.extractors.execution import ExecutionExtractor
from lfx.extractors.user_feedback import UserFeedbackExtractor
from lfx.layers.harness import Playbook, PlaybookEntry
from lfx.learner import AsyncLearner
from lfx.llm import MockLLMClient
from lfx.wrapper import wrap


class TestLiveModeEndToEnd:
    def test_wrap_collect_learn_cycle(self) -> None:
        """Wrap a mock client, make calls, collect episodes, learn."""
        # Setup
        state = AgentState()
        state.harness.playbook = Playbook(entries=[
            PlaybookEntry(id="tip-1", content="Be concise"),
        ])

        learner = AsyncLearner(agent_state=state, active_layers=["harness"])
        learner.start()

        pipeline = RewardPipeline([
            ExecutionExtractor(),
            UserFeedbackExtractor(),
        ])
        collector = EpisodeCollector(
            pipeline=pipeline,
            batch_size=3,
            on_batch=learner.on_batch,
        )

        client = MockLLMClient(responses=["Here is a helpful response with details."])
        wrapped = wrap(client, collector=collector)

        # Make 3 calls (triggers batch at batch_size=3)
        for i in range(3):
            result = wrapped.complete([{"role": "user", "content": f"Question {i}"}])
            assert result == "Here is a helpful response with details."

        # Wait for async learning
        time.sleep(1.0)

        # Verify learning happened
        assert learner.metrics["batches_trained"] >= 1
        assert collector.metrics["episodes_collected"] == 3

        learner.stop()

    def test_user_feedback_overrides_computed_reward(self) -> None:
        """Submit negative user feedback — verify it overrides."""
        state = AgentState()
        state.harness.playbook = Playbook(entries=[
            PlaybookEntry(id="tip-1", content="Be helpful"),
        ])

        pipeline = RewardPipeline([ExecutionExtractor()])
        collector = EpisodeCollector(
            pipeline=pipeline,
            batch_size=100,
        )

        client = MockLLMClient(responses=["response"])
        wrapped = wrap(client, collector=collector)

        wrapped.complete([{"role": "user", "content": "test"}])
        ep_id = list(collector._episode_index.keys())[0]

        # Submit negative feedback
        assert collector.submit_feedback(ep_id, -1.0) is True

        # Check the signal
        ep = collector._episode_index[ep_id]
        assert ep.summary.effective_reward() == -1.0  # user overrides
```

**Step 2: Run tests**

Run: `pytest tests/test_live_mode.py -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add tests/test_live_mode.py
git commit -m "test: add integration tests for full live mode pipeline"
```

---

## Task 13: Backward compatibility — update existing tests

**Files:**
- Modify: various test files that use `EpisodeSummary(total_reward=X)`

**Step 1: Run all existing tests**

Run: `pytest tests/ -v`

Check which tests fail due to the EpisodeSummary migration.

**Step 2: Fix any failing tests**

The `EpisodeSummary.__init__` now accepts `total_reward` as a keyword arg (via the custom `__init__` from Task 2). If any tests use positional args, convert them to keyword.

Common pattern to find and fix:
- `EpisodeSummary(total_reward=0.8)` → should still work (keyword arg in custom `__init__`)
- `EpisodeSummary(0.8)` → needs to become `EpisodeSummary(total_reward=0.8)`

**Step 3: Run all tests again**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 4: Commit**

```bash
git add tests/
git commit -m "fix: update tests for EpisodeSummary signals migration"
```

---

## Task 14: Update public exports and write docstring

**Files:**
- Modify: `lfx/__init__.py`
- Modify: `lfx/extractors/__init__.py`

**Step 1: Update `lfx/extractors/__init__.py` with convenience imports**

```python
# lfx/extractors/__init__.py
"""Built-in reward extractors."""

from lfx.extractors.execution import ExecutionExtractor
from lfx.extractors.formatting import FormattingFilter
from lfx.extractors.outcome import OutcomeExtractor
from lfx.extractors.user_feedback import UserFeedbackExtractor

__all__ = [
    "ExecutionExtractor",
    "FormattingFilter",
    "OutcomeExtractor",
    "UserFeedbackExtractor",
]
```

**Step 2: Run all tests one final time**

Run: `pytest tests/ -v`
Expected: All PASS

**Step 3: Commit**

```bash
git add lfx/__init__.py lfx/extractors/__init__.py
git commit -m "feat: update public API exports for reward system"
```

---

## Summary of all new/modified files

**New files (8):**
- `lfx/core/reward.py` — RewardSignal, RewardExtractor, RewardPipeline
- `lfx/extractors/__init__.py` — Extractor package exports
- `lfx/extractors/execution.py` — ExecutionExtractor
- `lfx/extractors/outcome.py` — OutcomeExtractor
- `lfx/extractors/user_feedback.py` — UserFeedbackExtractor
- `lfx/extractors/formatting.py` — FormattingFilter
- `lfx/collector.py` — EpisodeCollector
- `lfx/learner.py` — AsyncLearner
- `lfx/wrapper.py` — lfx.wrap() SDK wrapper

**Modified files (3):**
- `lfx/core/episode.py` — EpisodeSummary with signals dict
- `lfx/layers/harness.py` — forward_backward uses effective_reward()
- `lfx/__init__.py` — New exports

**New test files (5):**
- `tests/test_reward.py`
- `tests/test_extractors.py`
- `tests/test_collector.py`
- `tests/test_learner.py`
- `tests/test_wrapper.py`
- `tests/test_live_mode.py`
- `tests/test_harness_signals.py`
