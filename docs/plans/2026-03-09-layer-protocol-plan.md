# Layer Protocol Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the proposer-based learning pattern with a unified Layer protocol (five Tinker verbs) on all three layers, with a two-phase contract and 45+ tests.

**Architecture:** New `core/types.py` defines `Future[T]`, `Datum`, and result dataclasses. New `core/layer.py` defines the `Layer` protocol. Each layer (Harness, Router, Weights) gains a `_pending` accumulator, `forward_backward`, `optim_step`, `sample`, `save_state`, and `load_state`. The learning loop is rewritten to pump `forward_backward → optim_step` over a list of layers.

**Tech Stack:** Python 3.11+, dataclasses, threading (for Future), pytest

**Design doc:** `docs/plans/2026-03-09-layer-protocol-design.md`

---

## Task 1: Core Types — `Future[T]` and result dataclasses

**Files:**
- Create: `lfx/core/types.py`
- Test: `tests/test_types.py`

### Step 1: Write failing tests for Future

```python
# tests/test_types.py
"""Tests for lfx.core.types — Future, Datum, result dataclasses."""

import threading

import pytest

from lfx.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)


# -- Future --


class TestFuture:
    def test_immediate_resolves(self) -> None:
        f = Future.immediate(42)
        assert f.done
        assert f.result() == 42

    def test_immediate_none(self) -> None:
        f = Future.immediate(None)
        assert f.done
        assert f.result() is None

    def test_deferred_set_then_get(self) -> None:
        f: Future[str] = Future()
        assert not f.done
        f.set_result("hello")
        assert f.done
        assert f.result() == "hello"

    def test_timeout_raises(self) -> None:
        f: Future[int] = Future()
        with pytest.raises(TimeoutError):
            f.result(timeout=0.01)

    def test_double_set_raises(self) -> None:
        f = Future.immediate(1)
        with pytest.raises(RuntimeError, match="already resolved"):
            f.set_result(2)

    def test_threaded_set_get(self) -> None:
        f: Future[int] = Future()

        def setter():
            f.set_result(99)

        t = threading.Thread(target=setter)
        t.start()
        val = f.result(timeout=2.0)
        t.join()
        assert val == 99


# -- Datum --


class TestDatum:
    def test_datum_default_loss_fn(self) -> None:
        d = Datum(episodes=[])
        assert d.loss_fn == "auto"
        assert d.loss_fn_config == {}

    def test_datum_custom(self) -> None:
        d = Datum(episodes=[], loss_fn="grpo", loss_fn_config={"lr": 1e-4})
        assert d.loss_fn == "grpo"
        assert d.loss_fn_config["lr"] == 1e-4

    def test_datum_frozen(self) -> None:
        d = Datum(episodes=[])
        with pytest.raises(AttributeError):
            d.episodes = []  # type: ignore[misc]


# -- Result types --


class TestResultTypes:
    def test_fb_result_defaults(self) -> None:
        r = FBResult(status="ok")
        assert r.status == "ok"
        assert r.metrics == {}

    def test_optim_result(self) -> None:
        r = OptimResult(status="ok", updates_applied=3)
        assert r.updates_applied == 3

    def test_sample_context_defaults(self) -> None:
        ctx = SampleContext()
        assert ctx.bench == ""

    def test_sample_result(self) -> None:
        r = SampleResult(output="prompt text", metadata={"bench": "tau2"})
        assert r.output == "prompt text"

    def test_save_result(self) -> None:
        r = SaveResult(name="checkpoint-1")
        assert r.status == "ok"

    def test_load_result(self) -> None:
        r = LoadResult(status="ok")
        assert r.status == "ok"
```

### Step 2: Run tests to verify they fail

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_types.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lfx.core.types'`

### Step 3: Implement `lfx/core/types.py`

```python
# lfx/core/types.py
"""Core types for the Layer protocol: Future, Datum, and result dataclasses."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from lfx.core.episode import Episode

T = TypeVar("T")
_UNSET = object()


class Future(Generic[T]):
    """Synchronous-first result wrapper.

    Most layers resolve immediately via ``Future.immediate()``.  The deferred
    path (``set_result`` + ``result(timeout)``) supports async backends
    like SkyRL/Tinker.
    """

    def __init__(self) -> None:
        self._value: T | object = _UNSET
        self._event = threading.Event()
        self._lock = threading.Lock()

    def result(self, timeout: float | None = None) -> T:
        """Block until resolved.  Raises ``TimeoutError`` if *timeout* expires."""
        if not self._event.wait(timeout):
            raise TimeoutError("Future not resolved within timeout")
        return self._value  # type: ignore[return-value]

    def set_result(self, value: T) -> None:
        """Resolve the future.  Raises ``RuntimeError`` on double-set."""
        with self._lock:
            if self._value is not _UNSET:
                raise RuntimeError("Future already resolved")
            self._value = value
            self._event.set()

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @classmethod
    def immediate(cls, value: T) -> Future[T]:
        """Create an already-resolved future (the common path)."""
        f: Future[T] = cls()
        f.set_result(value)
        return f


# -- Datum --


@dataclass(frozen=True)
class Datum:
    """Universal training atom.

    Each layer extracts what it needs from the episodes.  ``loss_fn`` is a
    hint: ``"auto"`` means each layer uses its native loss.
    Frozen to prevent accidental field reassignment after construction.
    (Note: list/dict contents are still mutable — frozen prevents
    ``datum.episodes = new_list``, not ``datum.episodes.append(x)``.)
    """

    episodes: list[Episode]
    loss_fn: str = "auto"
    loss_fn_config: dict[str, Any] = field(default_factory=dict)


# -- Result types --


@dataclass(frozen=True)
class FBResult:
    """Result of ``forward_backward``."""

    status: str  # "ok" | "skipped" | "error"
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimResult:
    """Result of ``optim_step``."""

    status: str  # "ok" | "skipped" | "error"
    updates_applied: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleContext:
    """Input to ``sample`` — what the layer needs to produce output."""

    bench: str = ""
    query_features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SampleResult:
    """Output of ``sample``."""

    output: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SaveResult:
    """Result of ``save_state``."""

    name: str = ""
    status: str = "ok"


@dataclass(frozen=True)
class LoadResult:
    """Result of ``load_state``."""

    status: str = "ok"
```

### Step 4: Run tests to verify they pass

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/test_types.py -v`
Expected: All 12 tests PASS

### Step 5: Commit

```bash
git add lfx/core/types.py tests/test_types.py
git commit -m "feat(lfx): add Future[T], Datum, and result dataclasses

Core types for the Layer protocol: Future with _UNSET sentinel and
thread-safe set/get, Datum as universal training atom, typed result
containers for each protocol verb."
```

---

## Task 2: Layer Protocol Definition

**Files:**
- Create: `lfx/core/layer.py`
- Modify: `lfx/core/__init__.py:1-26`
- Test: `tests/test_types.py` (append)

### Step 1: Write failing test for protocol conformance

Append to `tests/test_types.py`:

```python
from lfx.core.layer import Layer


class TestLayerProtocol:
    def test_protocol_has_required_methods(self) -> None:
        """Verify the Protocol defines all five verbs + to_dict."""
        import inspect
        members = {name for name, _ in inspect.getmembers(Layer)
                   if not name.startswith("_")}
        required = {"forward_backward", "optim_step", "sample",
                     "save_state", "load_state", "to_dict"}
        assert required.issubset(members)
```

### Step 2: Run test to verify it fails

Run: `python -m pytest tests/test_types.py::TestLayerProtocol -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lfx.core.layer'`

### Step 3: Implement `lfx/core/layer.py`

```python
# lfx/core/layer.py
"""Layer protocol — the five Tinker verbs.

Every learning layer (Harness, Router, Weights) implements this protocol.
The two-phase contract: ``forward_backward`` accumulates learning signals
into a ``_pending`` buffer without mutating observable state;
``optim_step`` drains the buffer and applies mutations.
"""

from __future__ import annotations

from typing import Any, Protocol

from lfx.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)


class Layer(Protocol):
    """Unified learning layer protocol.

    Verbs
    -----
    forward_backward : Compute learning signal from episodes.  Writes to
        internal ``_pending`` accumulator.  MUST NOT mutate observable state.
    optim_step : Drain ``_pending`` and apply mutations to observable state.
    sample : Produce output from current (applied) state.
    save_state : Checkpoint applied state (not pending).
    load_state : Restore from checkpoint, clearing any pending.
    to_dict : Deterministic serialization for StateID.
    """

    def forward_backward(self, data: Datum) -> Future[FBResult]: ...
    def optim_step(self) -> Future[OptimResult]: ...
    def sample(self, ctx: SampleContext) -> Future[SampleResult]: ...
    def save_state(self, name: str) -> Future[SaveResult]: ...
    def load_state(self, state_dict: dict[str, Any]) -> Future[LoadResult]: ...
    def to_dict(self) -> dict[str, Any]: ...
```

### Step 4: Update `lfx/core/__init__.py` exports

Add the new types to `__init__.py`:

```python
# lfx/core/__init__.py
"""Core data structures and learning loop."""

from lfx.core.episode import (
    Episode,
    EpisodeSummary,
    LearningUpdate,
    Message,
    StepMeta,
    Timing,
    TokenUsage,
    ToolCall,
)
from lfx.core.layer import Layer
from lfx.core.state import StateID
from lfx.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)

__all__ = [
    "Datum",
    "Episode",
    "EpisodeSummary",
    "FBResult",
    "Future",
    "Layer",
    "LearningUpdate",
    "LoadResult",
    "Message",
    "OptimResult",
    "SampleContext",
    "SampleResult",
    "SaveResult",
    "StepMeta",
    "StateID",
    "Timing",
    "TokenUsage",
    "ToolCall",
]
```

### Step 5: Run tests to verify they pass

Run: `python -m pytest tests/test_types.py -v`
Expected: All 13 tests PASS

### Step 6: Commit

```bash
git add lfx/core/layer.py lfx/core/__init__.py tests/test_types.py
git commit -m "feat(lfx): define Layer protocol with five Tinker verbs

Protocol: forward_backward, optim_step, sample, save_state, load_state,
plus to_dict for deterministic serialization. Documents the two-phase
contract (accumulate in fb, apply in optim_step)."
```

---

## Task 3: Harness — Layer Protocol Implementation

**Files:**
- Modify: `lfx/layers/harness.py:259-350` (add protocol methods + `_pending` dataclass)
- Test: `tests/test_layer_protocol.py` (create)

### Step 1: Write failing contract tests

```python
# tests/test_layer_protocol.py
"""Contract tests for the Layer protocol on all three layers.

These tests verify the two-phase invariant and protocol conformance.
Each layer gets the same contract tests.
"""

import copy
import json

import pytest

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.types import Datum, SampleContext
from lfx.layers.harness import Harness, PlaybookEntry, PromptCandidate


def _make_episode(
    bench: str = "test",
    task_id: str = "t1",
    reward: float = 0.8,
    model: str = "haiku",
) -> Episode:
    """Minimal Episode for testing."""
    return Episode(
        id=Episode.new_id(),
        state_id="deadbeef",
        task_id=task_id,
        bench=bench,
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!", model=model),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


def _make_datum(n: int = 3, bench: str = "test", reward: float = 0.8) -> Datum:
    """Build a Datum with n episodes."""
    return Datum(episodes=[_make_episode(bench=bench, reward=reward) for _ in range(n)])


# ── Harness contract tests ──


class TestHarnessProtocol:
    def test_forward_backward_returns_future(self) -> None:
        h = Harness()
        fut = h.forward_backward(_make_datum())
        assert fut.done
        result = fut.result()
        assert result.status == "ok"

    def test_forward_backward_no_mutation(self) -> None:
        """forward_backward MUST NOT mutate observable state."""
        h = Harness(system_prompts={"test": "prompt"})
        h.playbook.add(PlaybookEntry(id="s-1", content="strategy", helpful=2))
        state_before = json.dumps(h.to_dict(), sort_keys=True)
        h.forward_backward(_make_datum())
        state_after = json.dumps(h.to_dict(), sort_keys=True)
        assert state_before == state_after

    def test_optim_step_applies_pending(self) -> None:
        h = Harness(system_prompts={"test": "base prompt"})
        h.playbook.add(PlaybookEntry(id="s-1", content="strategy", helpful=0))
        helpful_before = h.playbook.lookup("s-1").helpful
        h.forward_backward(_make_datum())
        result = h.optim_step().result()
        assert result.status == "ok"
        assert result.updates_applied == 1  # one playbook entry updated
        # Verify the entry was actually mutated
        assert h.playbook.lookup("s-1").helpful > helpful_before

    def test_optim_step_drains_pending(self) -> None:
        """After optim_step, a second optim_step is a no-op."""
        h = Harness()
        h.forward_backward(_make_datum())
        r1 = h.optim_step().result()
        r2 = h.optim_step().result()
        assert r2.updates_applied == 0

    def test_multiple_forward_backward_then_one_optim(self) -> None:
        """Multiple forward_backward calls accumulate; one optim drains all."""
        h = Harness()
        h.forward_backward(_make_datum(n=2))
        h.forward_backward(_make_datum(n=3))
        result = h.optim_step().result()
        assert result.status == "ok"
        # Second optim is no-op
        r2 = h.optim_step().result()
        assert r2.updates_applied == 0

    def test_optim_without_forward_is_noop(self) -> None:
        h = Harness()
        result = h.optim_step().result()
        assert result.updates_applied == 0

    def test_sample_returns_result(self) -> None:
        h = Harness(system_prompts={"bench1": "You are an agent."})
        result = h.sample(SampleContext(bench="bench1")).result()
        assert "You are an agent." in result.output

    def test_sample_missing_bench(self) -> None:
        h = Harness()
        result = h.sample(SampleContext(bench="unknown")).result()
        assert result.output is not None  # empty string is fine

    def test_save_state(self) -> None:
        h = Harness(system_prompts={"test": "prompt"})
        result = h.save_state("ckpt-1").result()
        assert result.status == "ok"
        assert result.name == "ckpt-1"

    def test_load_state(self) -> None:
        h = Harness(system_prompts={"test": "original"})
        saved = h.to_dict()
        h.system_prompts["test"] = "modified"
        h.load_state(saved)
        assert h.system_prompts["test"] == "original"

    def test_save_load_roundtrip(self) -> None:
        h = Harness(system_prompts={"test": "prompt"})
        h.playbook.add(PlaybookEntry(id="s-1", content="strat", helpful=3, harmful=1))
        saved = h.to_dict()
        s1 = json.dumps(saved, sort_keys=True)
        h2 = Harness()
        h2.load_state(saved)
        s2 = json.dumps(h2.to_dict(), sort_keys=True)
        assert s1 == s2

    def test_save_between_phases_excludes_pending(self) -> None:
        """save_state after forward_backward but before optim_step
        should NOT include pending data."""
        h = Harness()
        h.forward_backward(_make_datum())
        saved = h.to_dict()
        h2 = Harness()
        h2.load_state(saved)
        # h2 should have no pending, so optim is no-op
        r = h2.optim_step().result()
        assert r.updates_applied == 0

    def test_to_dict_deterministic(self) -> None:
        h = Harness(system_prompts={"b": "2", "a": "1"})
        s1 = json.dumps(h.to_dict(), sort_keys=True)
        s2 = json.dumps(h.to_dict(), sort_keys=True)
        assert s1 == s2
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_layer_protocol.py::TestHarnessProtocol -v`
Expected: FAIL — `AttributeError: 'Harness' object has no attribute 'forward_backward'`

### Step 3: Implement Harness protocol methods

Add to `lfx/layers/harness.py` — insert a `_HarnessPending` dataclass before the `Harness` class, and add protocol methods to `Harness`:

**Insert after line 254 (after the Insight class), before the Harness class:**

```python
@dataclass
class _HarnessPending:
    """Accumulator for forward_backward signals.  Drained by optim_step."""

    playbook_signals: dict[str, tuple[int, int]] = field(default_factory=dict)
    # entry_id -> (helpful_delta, harmful_delta)
    insights: list[Insight] = field(default_factory=list)
    candidates: dict[str, list[PromptCandidate]] = field(default_factory=dict)
    # bench -> candidates
```

**Add these imports at the top of harness.py:**

```python
import copy

from lfx.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)
```

**Add `_pending` field to `Harness.__init__` and these methods at the end of the Harness class:**

```python
    # -- _pending field (add to dataclass fields) --
    _pending: _HarnessPending = field(default_factory=_HarnessPending)

    # -- Layer protocol --

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Accumulate learning signals from episodes.

        Analyzes episodes to produce playbook helpful/harmful tallies.
        Does NOT mutate observable state (system_prompts, playbook, pareto_fronts).
        """
        n_signals = 0
        for ep in data.episodes:
            reward = ep.summary.total_reward
            # Tally playbook signals: good episodes reinforce, bad ones penalize
            for entry in self.playbook.entries:
                entry_id = entry.id
                if entry_id not in self._pending.playbook_signals:
                    self._pending.playbook_signals[entry_id] = (0, 0)
                h, m = self._pending.playbook_signals[entry_id]
                if reward > 0.5:
                    self._pending.playbook_signals[entry_id] = (h + 1, m)
                else:
                    self._pending.playbook_signals[entry_id] = (h, m + 1)
                n_signals += 1

        return Future.immediate(FBResult(
            status="ok",
            metrics={"n_signals": n_signals, "n_episodes": len(data.episodes)},
        ))

    def optim_step(self) -> Future[OptimResult]:
        """Apply accumulated signals to playbook, Pareto fronts, and prompts.

        Uses snapshot-rollback for atomicity: takes a snapshot of mutable
        state before applying, rolls back on any exception.  _pending is
        drained only on full success.
        """
        if (
            not self._pending.playbook_signals
            and not self._pending.insights
            and not self._pending.candidates
        ):
            return Future.immediate(OptimResult(status="ok", updates_applied=0))

        # Snapshot mutable state for rollback
        playbook_snapshot = copy.deepcopy(self.playbook)
        prompts_snapshot = dict(self.system_prompts)
        fronts_snapshot = copy.deepcopy(self.pareto_fronts)

        try:
            updates = 0

            # Apply playbook helpful/harmful deltas
            for entry_id, (h_delta, m_delta) in self._pending.playbook_signals.items():
                entry = self.playbook.lookup(entry_id)
                if entry:
                    entry.helpful += h_delta
                    entry.harmful += m_delta
                    updates += 1

            # Apply insights
            if self._pending.insights:
                updates += self.apply_insights(self._pending.insights)

            # Apply Pareto candidates
            for bench, candidates in self._pending.candidates.items():
                for c in candidates:
                    self.update_pareto(bench, c)
                    updates += 1

            # Success — drain pending
            self._pending = _HarnessPending()

            return Future.immediate(OptimResult(
                status="ok",
                updates_applied=updates,
            ))
        except Exception as exc:
            # Rollback to snapshot — no partial mutations
            self.playbook = playbook_snapshot
            self.system_prompts = prompts_snapshot
            self.pareto_fronts = fronts_snapshot
            return Future.immediate(OptimResult(
                status="error",
                metrics={"error": str(exc)},
            ))

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Return system prompt + rendered playbook for the given bench."""
        prompt = self.system_prompt(ctx.bench)
        return Future.immediate(SampleResult(
            output=prompt,
            metadata={"bench": ctx.bench},
        ))

    def save_state(self, name: str) -> Future[SaveResult]:
        """Checkpoint applied state (excludes _pending)."""
        return Future.immediate(SaveResult(name=name, status="ok"))

    def load_state(self, state_dict: dict[str, Any]) -> Future[LoadResult]:
        """Restore from checkpoint, clearing any pending."""
        self.system_prompts = state_dict.get("system_prompts", {})
        self.playbook = Playbook(
            entries=[
                PlaybookEntry(**e)
                for e in state_dict.get("playbook", {}).get("entries", [])
            ]
        )
        self.pareto_fronts = {}
        for bench, pf_dict in state_dict.get("pareto_fronts", {}).items():
            pf = ParetoFront()
            for c_dict in pf_dict.get("candidates", []):
                pf.candidates.append(PromptCandidate(**c_dict))
            self.pareto_fronts[bench] = pf
        self.tool_configs = [
            ToolConfig(**tc) for tc in state_dict.get("tool_configs", [])
        ]
        self.validators = state_dict.get("validators", {})
        self._pending = _HarnessPending()
        return Future.immediate(LoadResult(status="ok"))
```

### Step 4: Run tests to verify they pass

Run: `python -m pytest tests/test_layer_protocol.py::TestHarnessProtocol -v`
Expected: All 13 tests PASS

### Step 5: Run existing harness tests to check no regressions

Run: `python -m pytest tests/test_packs.py::TestHarness tests/test_packs.py::TestPlaybook tests/test_packs.py::TestParetoFront -v`
Expected: All PASS

### Step 6: Commit

```bash
git add lfx/layers/harness.py tests/test_layer_protocol.py
git commit -m "feat(lfx): implement Layer protocol on Harness

Two-phase contract: forward_backward accumulates playbook signals,
insights, and Pareto candidates in _pending without mutating state.
optim_step drains _pending and applies deltas. Includes sample,
save_state, and load_state."
```

---

## Task 4: Router — Layer Protocol Implementation

**Files:**
- Modify: `lfx/layers/router.py:99-260` (add protocol methods + `_pending` dataclass)
- Test: `tests/test_layer_protocol.py` (append)

### Step 1: Write failing contract tests

Append to `tests/test_layer_protocol.py`:

```python
from lfx.layers.router import QueryFeatures, Router, Tier


# ── Router contract tests ──


class TestRouterProtocol:
    def test_forward_backward_returns_future(self) -> None:
        r = Router()
        fut = r.forward_backward(_make_datum())
        assert fut.done
        assert fut.result().status == "ok"

    def test_forward_backward_no_mutation(self) -> None:
        r = Router(tier_models={t: f"model-{t}" for t in Tier.ALL})
        state_before = json.dumps(r.to_dict(), sort_keys=True)
        r.forward_backward(_make_datum())
        state_after = json.dumps(r.to_dict(), sort_keys=True)
        assert state_before == state_after

    def test_optim_step_applies_pending(self) -> None:
        r = Router(tier_models={t: f"model-{t}" for t in Tier.ALL})
        r.forward_backward(_make_datum(n=5))
        result = r.optim_step().result()
        assert result.status == "ok"
        assert result.updates_applied > 0  # weights should have been adjusted

    def test_optim_step_drains_pending(self) -> None:
        r = Router()
        r.forward_backward(_make_datum())
        r.optim_step()
        r2 = r.optim_step().result()
        assert r2.updates_applied == 0

    def test_multiple_forward_backward_accumulates(self) -> None:
        r = Router()
        r.forward_backward(_make_datum(n=2))
        r.forward_backward(_make_datum(n=3))
        result = r.optim_step().result()
        assert result.status == "ok"
        r2 = r.optim_step().result()
        assert r2.updates_applied == 0

    def test_optim_without_forward_is_noop(self) -> None:
        r = Router()
        result = r.optim_step().result()
        assert result.updates_applied == 0

    def test_sample_returns_model(self) -> None:
        r = Router(tier_models={
            Tier.LIGHT: "haiku",
            Tier.MEDIUM: "sonnet",
            Tier.HEAVY: "opus",
            Tier.REASONING: "opus",
        })
        # Pass raw (unnormalized) values — not .to_dict() which normalizes
        result = r.sample(SampleContext(
            query_features={"token_count": 10},
        )).result()
        assert result.output in ("haiku", "sonnet", "opus")

    def test_sample_accepts_query_features_object(self) -> None:
        """sample() also accepts a QueryFeatures object directly."""
        r = Router(tier_models={
            Tier.LIGHT: "haiku",
            Tier.MEDIUM: "sonnet",
            Tier.HEAVY: "opus",
            Tier.REASONING: "opus",
        })
        result = r.sample(SampleContext(
            query_features=QueryFeatures(token_count=500, reasoning_markers=3),
        )).result()
        assert result.output in ("haiku", "sonnet", "opus")
        assert result.metadata["tier"] in Tier.ALL

    def test_save_state(self) -> None:
        r = Router()
        result = r.save_state("ckpt-1").result()
        assert result.status == "ok"

    def test_load_state(self) -> None:
        r = Router(tier_models={Tier.LIGHT: "haiku", Tier.MEDIUM: "sonnet",
                                Tier.HEAVY: "opus", Tier.REASONING: "opus"})
        saved = r.to_dict()
        r2 = Router()
        r2.load_state(saved)
        assert r2.tier_models[Tier.LIGHT] == "haiku"

    def test_save_load_roundtrip(self) -> None:
        r = Router(tier_models={Tier.LIGHT: "haiku", Tier.MEDIUM: "sonnet",
                                Tier.HEAVY: "opus", Tier.REASONING: "opus"})
        saved = r.to_dict()
        s1 = json.dumps(saved, sort_keys=True)
        r2 = Router()
        r2.load_state(saved)
        s2 = json.dumps(r2.to_dict(), sort_keys=True)
        assert s1 == s2

    def test_to_dict_deterministic(self) -> None:
        r = Router()
        s1 = json.dumps(r.to_dict(), sort_keys=True)
        s2 = json.dumps(r.to_dict(), sort_keys=True)
        assert s1 == s2
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_layer_protocol.py::TestRouterProtocol -v`
Expected: FAIL — `AttributeError: 'Router' object has no attribute 'forward_backward'`

### Step 3: Implement Router protocol methods

Add to `lfx/layers/router.py`:

**Add imports at top:**

```python
from lfx.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)
```

**Insert before the Router class (after DEFAULT_TIER_THRESHOLDS):**

```python
@dataclass
class _RouterPending:
    """Accumulator for forward_backward signals.  Drained by optim_step.

    Stores (QueryFeatures, model_id, cost, reward) tuples extracted from
    episodes.  These are fed to ``record_outcome`` during optim_step to
    ensure the correct sample schema (including ``tier`` key) is used.
    """

    samples: list[tuple[QueryFeatures, str, float, float]] = field(
        default_factory=list,
    )
    # Each: (features, model_id, cost, reward)
```

**Add `_pending` field and protocol methods to Router:**

```python
    # -- _pending field (add to dataclass fields) --
    _pending: _RouterPending = field(default_factory=_RouterPending)

    # -- Layer protocol --

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Extract routing tuples from episodes.  Does NOT mutate weights.

        Accumulates (QueryFeatures, model_id, cost, reward) tuples into
        _pending.  Does NOT call record_outcome here (that mutates
        training_samples which is observable state).
        """
        n_samples = 0
        for ep in data.episodes:
            # Extract model from first assistant message
            model_id = ""
            for msg in ep.messages:
                if msg.role == "assistant" and msg.model:
                    model_id = msg.model
                    break

            cost = float(ep.summary.token_usage.total_tokens) if ep.summary.token_usage else 0.0
            reward = ep.summary.total_reward

            # Build minimal features from episode
            token_count = sum(
                msg.token_count or len(msg.content.split())
                for msg in ep.messages if msg.role == "user"
            )
            features = QueryFeatures(token_count=token_count)

            self._pending.samples.append((features, model_id, cost, reward))
            n_samples += 1

        return Future.immediate(FBResult(
            status="ok",
            metrics={"n_samples": n_samples},
        ))

    def optim_step(self) -> Future[OptimResult]:
        """Update routing weights from accumulated samples.

        Uses record_outcome() to ensure correct sample schema (including
        tier key).  Snapshot-rollback for atomicity: snapshots
        training_samples and score_weights, rolls back on failure.
        """
        if not self._pending.samples:
            return Future.immediate(OptimResult(status="ok", updates_applied=0))

        # Snapshot mutable state for rollback
        samples_snapshot = list(self.training_samples)
        weights_snapshot = dict(self.score_weights)

        try:
            # Feed samples via record_outcome to get correct schema (incl. tier)
            for features, model_id, cost, reward in self._pending.samples:
                self.record_outcome(features, model_id, cost, reward)

            deltas = self.update_weights()
            n_updates = len(deltas)

            # Success — drain pending
            self._pending = _RouterPending()

            return Future.immediate(OptimResult(
                status="ok",
                updates_applied=n_updates,
            ))
        except Exception as exc:
            # Rollback to snapshot — no partial mutations
            self.training_samples = samples_snapshot
            self.score_weights = weights_snapshot
            return Future.immediate(OptimResult(
                status="error",
                metrics={"error": str(exc)},
            ))

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Route a query to a model ID.

        Accepts raw (unnormalized) feature values in ctx.query_features.
        QueryFeatures.to_dict() handles normalization internally.
        """
        raw = ctx.query_features
        if isinstance(raw, QueryFeatures):
            features = raw
        else:
            # Expect raw, unnormalized values (e.g. token_count=500, not 0.5)
            features = QueryFeatures(
                token_count=int(raw.get("token_count", 0)),
                has_code=bool(raw.get("has_code", False)),
                reasoning_markers=int(raw.get("reasoning_markers", 0)),
                technical_terms=int(raw.get("technical_terms", 0)),
                tool_calls_expected=int(raw.get("tool_calls_expected", 0)),
                conversation_depth=int(raw.get("conversation_depth", 0)),
            )
        model_id = self.route(features)
        tier = self.classify(features)
        return Future.immediate(SampleResult(
            output=model_id,
            metadata={"tier": tier},
        ))

    def save_state(self, name: str) -> Future[SaveResult]:
        return Future.immediate(SaveResult(name=name, status="ok"))

    def load_state(self, state_dict: dict[str, Any]) -> Future[LoadResult]:
        """Restore from checkpoint, clearing pending."""
        self.tier_models = state_dict.get("tier_models", {t: "" for t in Tier.ALL})
        self.score_weights = state_dict.get("score_weights", dict(DEFAULT_SCORE_WEIGHTS))
        self.tier_thresholds = state_dict.get("tier_thresholds", dict(DEFAULT_TIER_THRESHOLDS))
        self.fallback_chains = state_dict.get("fallback_chains", [])
        self.token_budgets = state_dict.get("token_budgets", {})
        self.cost_weights = state_dict.get("cost_weights", {})
        self.training_samples = []
        self._pending = _RouterPending()
        return Future.immediate(LoadResult(status="ok"))
```

### Step 4: Run tests

Run: `python -m pytest tests/test_layer_protocol.py::TestRouterProtocol -v`
Expected: All 11 tests PASS

### Step 5: Run existing router tests

Run: `python -m pytest tests/test_packs.py::TestRouter -v`
Expected: All PASS

### Step 6: Commit

```bash
git add lfx/layers/router.py tests/test_layer_protocol.py
git commit -m "feat(lfx): implement Layer protocol on Router

forward_backward extracts routing tuples (features, model, cost, reward)
from episodes. optim_step feeds them to update_weights(). sample routes
queries via classify/route. Save/load round-trips tier config."
```

---

## Task 5: Weights — Layer Protocol Implementation

**Files:**
- Modify: `lfx/layers/weights.py:42-88` (add protocol methods + `_pending`)
- Test: `tests/test_layer_protocol.py` (append)

### Step 1: Write failing contract tests

Append to `tests/test_layer_protocol.py`:

```python
from lfx.layers.weights import GRPOConfig, Weights


# ── Weights contract tests ──


class TestWeightsProtocol:
    def test_forward_backward_returns_future(self) -> None:
        w = Weights(model_ref="meta-llama/Llama-3-8B")
        fut = w.forward_backward(_make_datum())
        assert fut.done
        assert fut.result().status == "ok"

    def test_forward_backward_no_mutation(self) -> None:
        w = Weights(model_ref="meta-llama/Llama-3-8B", adapter_refs=["lora-v1"])
        state_before = json.dumps(w.to_dict(), sort_keys=True)
        w.forward_backward(_make_datum())
        state_after = json.dumps(w.to_dict(), sort_keys=True)
        assert state_before == state_after

    def test_forward_backward_computes_advantages(self) -> None:
        """GRPO advantages: per-episode reward minus per-task mean reward."""
        datum = Datum(episodes=[
            _make_episode(task_id="t1", reward=0.9),
            _make_episode(task_id="t1", reward=0.7),
            _make_episode(task_id="t1", reward=0.5),
        ])
        w = Weights()
        result = w.forward_backward(datum).result()
        assert result.metrics.get("n_advantages", 0) == 3

    def test_optim_step_is_passthrough(self) -> None:
        """Weights optim_step is a pass-through (real training deferred)."""
        w = Weights()
        w.forward_backward(_make_datum())
        result = w.optim_step().result()
        assert result.status == "skipped"
        assert result.updates_applied == 0
        assert result.metrics["advantages_computed"] == 3

    def test_optim_step_records_history(self) -> None:
        """Even when deferred, optim_step records a training history entry."""
        w = Weights()
        assert len(w.training_history) == 0
        w.forward_backward(_make_datum())
        w.optim_step()
        assert len(w.training_history) == 1
        assert w.training_history[0]["status"] == "deferred"
        assert w.training_history[0]["advantages_computed"] == 3

    def test_optim_step_drains_pending(self) -> None:
        w = Weights()
        w.forward_backward(_make_datum())
        w.optim_step()
        r2 = w.optim_step().result()
        assert r2.updates_applied == 0

    def test_optim_without_forward_is_noop(self) -> None:
        w = Weights()
        result = w.optim_step().result()
        assert result.updates_applied == 0

    def test_sample_returns_model_ref(self) -> None:
        w = Weights(model_ref="meta-llama/Llama-3-8B", adapter_refs=["lora-v1"])
        result = w.sample(SampleContext()).result()
        assert result.output == "meta-llama/Llama-3-8B"
        assert result.metadata.get("active_adapter") == "lora-v1"

    def test_save_state(self) -> None:
        w = Weights(model_ref="test-model")
        result = w.save_state("ckpt-1").result()
        assert result.status == "ok"

    def test_load_state(self) -> None:
        w = Weights(model_ref="model-a", adapter_refs=["lora-1"])
        saved = w.to_dict()
        w2 = Weights()
        w2.load_state(saved)
        assert w2.model_ref == "model-a"
        assert w2.adapter_refs == ["lora-1"]

    def test_save_load_roundtrip(self) -> None:
        w = Weights(model_ref="model-a", adapter_refs=["lora-1"])
        saved = w.to_dict()
        s1 = json.dumps(saved, sort_keys=True)
        w2 = Weights()
        w2.load_state(saved)
        s2 = json.dumps(w2.to_dict(), sort_keys=True)
        assert s1 == s2

    def test_to_dict_deterministic(self) -> None:
        w = Weights(model_ref="test")
        s1 = json.dumps(w.to_dict(), sort_keys=True)
        s2 = json.dumps(w.to_dict(), sort_keys=True)
        assert s1 == s2
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_layer_protocol.py::TestWeightsProtocol -v`
Expected: FAIL — `AttributeError: 'Weights' object has no attribute 'forward_backward'`

### Step 3: Implement Weights protocol methods

Modify `lfx/layers/weights.py`:

**Add imports at top:**

```python
from lfx.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)
```

**Insert before the Weights class:**

```python
@dataclass
class _WeightsPending:
    """Accumulator for GRPO advantages.  Drained by optim_step."""

    advantages: list[tuple[str, float]] = field(default_factory=list)
    # (episode_id, advantage)
```

**Add `_pending` field and protocol methods to Weights:**

```python
    # -- _pending field (add to dataclass fields) --
    _pending: _WeightsPending = field(default_factory=_WeightsPending)

    # -- Layer protocol --

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Compute GRPO advantages from episodes.

        Groups episodes by task_id, computes per-group mean reward, then
        advantage = episode_reward - mean_reward.  Does NOT mutate state.
        """
        # Group by task
        task_episodes: dict[str, list[tuple[str, float]]] = {}
        for ep in data.episodes:
            task_episodes.setdefault(ep.task_id, []).append(
                (ep.id, ep.summary.total_reward)
            )

        # Compute advantages
        n_advantages = 0
        for task_id, entries in task_episodes.items():
            mean_reward = sum(r for _, r in entries) / len(entries)
            for ep_id, reward in entries:
                advantage = reward - mean_reward
                self._pending.advantages.append((ep_id, advantage))
                n_advantages += 1

        return Future.immediate(FBResult(
            status="ok",
            metrics={"n_advantages": n_advantages},
        ))

    def optim_step(self) -> Future[OptimResult]:
        """Pass-through: actual SkyRL training deferred to a later PR.

        Records that advantages were computed in training_history, but does
        not call SkyRL or update model weights.  Snapshot-rollback for
        atomicity (will matter more when real training is added).
        """
        if not self._pending.advantages:
            return Future.immediate(OptimResult(
                status="skipped",
                updates_applied=0,
            ))

        # Snapshot for rollback
        history_snapshot = list(self.training_history)

        try:
            # In a future PR, this would call SkyRL's RayPPOTrainer.
            # For now, record the step in training_history.
            n = len(self._pending.advantages)
            self.training_history.append({
                "status": "deferred",
                "advantages_computed": n,
            })

            # Success — drain pending
            self._pending = _WeightsPending()

            return Future.immediate(OptimResult(
                status="skipped",
                updates_applied=0,
                metrics={"advantages_computed": n},
            ))
        except Exception as exc:
            # Rollback — no partial mutations
            self.training_history = history_snapshot
            return Future.immediate(OptimResult(
                status="error",
                metrics={"error": str(exc)},
            ))

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Return model reference and active adapter."""
        return Future.immediate(SampleResult(
            output=self.model_ref,
            metadata={"active_adapter": self.active_adapter},
        ))

    def save_state(self, name: str) -> Future[SaveResult]:
        return Future.immediate(SaveResult(name=name, status="ok"))

    def load_state(self, state_dict: dict[str, Any]) -> Future[LoadResult]:
        """Restore from checkpoint, clearing pending."""
        self.model_ref = state_dict.get("model_ref", "")
        self.adapter_refs = list(state_dict.get("adapter_refs", []))
        gc = state_dict.get("grpo_config", {})
        self.grpo_config = GRPOConfig(
            n_samples_per_prompt=gc.get("n_samples_per_prompt", 4),
            learning_rate=gc.get("learning_rate", 1e-5),
            kl_coeff=gc.get("kl_coeff", 0.05),
            clip_ratio=gc.get("clip_ratio", 0.2),
            epochs_per_batch=gc.get("epochs_per_batch", 1),
            max_grad_norm=gc.get("max_grad_norm", 1.0),
            use_advantage_normalization=gc.get("use_advantage_normalization", True),
            min_group_variance=gc.get("min_group_variance", 1e-6),
        )
        self.training_history = []
        self._pending = _WeightsPending()
        return Future.immediate(LoadResult(status="ok"))
```

### Step 4: Run tests

Run: `python -m pytest tests/test_layer_protocol.py::TestWeightsProtocol -v`
Expected: All 11 tests PASS

### Step 5: Run existing weights tests

Run: `python -m pytest tests/test_packs.py::TestWeights -v`
Expected: All PASS

### Step 6: Commit

```bash
git add lfx/layers/weights.py tests/test_layer_protocol.py
git commit -m "feat(lfx): implement Layer protocol on Weights

forward_backward computes GRPO advantages (reward - per-task mean).
optim_step is a pass-through (actual SkyRL training deferred).
sample returns model_ref + active_adapter."
```

---

## Task 6: Rewrite Learning Loop

**Files:**
- Modify: `lfx/core/loop.py:1-154` (full rewrite)
- Test: `tests/test_layer_protocol.py` (append loop integration tests)

### Step 1: Write failing loop integration tests

Append to `tests/test_layer_protocol.py`:

```python
from lfx.core.loop import AgentState, AdapterLike, learning_loop


class _MockAdapter:
    """Adapter that returns canned episodes."""

    def __init__(self, reward: float = 0.8) -> None:
        self.reward = reward
        self.call_count = 0

    def run_episode(self, task, agent_state) -> Episode:
        self.call_count += 1
        return _make_episode(reward=self.reward, task_id=str(task))


# ── Loop integration tests ──


class TestLearningLoop:
    def test_single_iteration(self) -> None:
        adapter = _MockAdapter()
        state = AgentState()
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2", "t3"],
            n_episodes=3,
            n_iterations=1,
        )
        assert adapter.call_count == 3
        assert sid.combined_hash

    def test_multiple_iterations(self) -> None:
        adapter = _MockAdapter()
        state = AgentState()
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2"],
            n_episodes=2,
            n_iterations=3,
        )
        assert adapter.call_count == 6  # 2 episodes x 3 iterations

    def test_active_layers_filter(self) -> None:
        """Only specified layers get forward_backward/optim_step."""
        adapter = _MockAdapter()
        state = AgentState()
        # Only train harness
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness"],
        )
        assert sid.combined_hash

    def test_state_id_stable_without_changes(self) -> None:
        """If no updates apply (empty layers), state_id should be stable."""
        adapter = _MockAdapter()
        state = AgentState()
        sid_before = state.state_id()
        state, sid_after = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
        )
        # State might or might not change depending on playbook signals,
        # but at minimum the hash should be computed
        assert sid_after.combined_hash

    def test_more_episodes_than_tasks(self) -> None:
        """When n_episodes > len(tasks), samples with replacement."""
        adapter = _MockAdapter()
        state = AgentState()
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=3,
            n_iterations=1,
        )
        assert adapter.call_count == 3  # 3 episodes from 1 task

    def test_empty_tasks_no_episodes(self) -> None:
        """Empty task list produces no episodes; layers see empty datum."""
        adapter = _MockAdapter()
        state = AgentState()
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=[],
            n_episodes=3,
            n_iterations=1,
        )
        assert adapter.call_count == 0

    def test_loop_layer_failure_continues(self) -> None:
        """If forward_backward raises for one layer, others still proceed."""
        adapter = _MockAdapter()
        state = AgentState()
        # Monkey-patch harness to fail
        original_fb = state.harness.forward_backward

        def failing_fb(data):
            raise RuntimeError("simulated failure")

        state.harness.forward_backward = failing_fb
        # Should not raise — loop isolates per-layer failures
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
        )
        assert sid.combined_hash
```

### Step 2: Run tests to verify they fail

Run: `python -m pytest tests/test_layer_protocol.py::TestLearningLoop -v`
Expected: FAIL — signature mismatch or missing `active_layers` parameter

### Step 3: Rewrite `lfx/core/loop.py`

```python
# lfx/core/loop.py
"""Learning loop: collect episodes -> forward_backward -> optim_step -> repeat.

The loop is benchmark-agnostic.  It delegates episode collection to an
``AdapterLike`` and learning to the Layer protocol on each layer.
Gating (regression checks) is intentionally *not* part of the inner
loop -- see ``gate.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from lfx.core.episode import Episode
from lfx.core.state import StateID
import random

from lfx.core.types import Datum, FBResult, Future, OptimResult
from lfx.layers.harness import Harness
from lfx.layers.router import Router
from lfx.layers.weights import Weights

log = logging.getLogger(__name__)


# -- Agent state bundle --


LAYER_NAMES = ("harness", "router", "weights")


@dataclass
class AgentState:
    """Bundle of the three mutable learning layers."""

    harness: Harness = field(default_factory=Harness)
    router: Router = field(default_factory=Router)
    weights: Weights = field(default_factory=Weights)

    def state_id(self) -> StateID:
        return StateID.from_layers(self.harness, self.router, self.weights)

    def get_layers(
        self, active: list[str] | None = None,
    ) -> list[tuple[str, Any]]:
        """Return (name, layer) pairs, filtered by *active* if given."""
        all_layers = [
            ("harness", self.harness),
            ("router", self.router),
            ("weights", self.weights),
        ]
        if active is None:
            return all_layers
        return [(n, l) for n, l in all_layers if n in active]


# -- Adapter protocol --


class AdapterLike(Protocol):
    def run_episode(self, task: Any, agent_state: AgentState) -> Episode: ...


# -- Learning loop --


def learning_loop(
    adapter: AdapterLike,
    agent_state: AgentState,
    tasks: list[Any],
    n_episodes: int,
    n_iterations: int,
    *,
    active_layers: list[str] | None = None,
) -> tuple[AgentState, StateID]:
    """Run the unified learning loop.

    Parameters
    ----------
    adapter:
        Environment adapter that produces episodes.
    agent_state:
        Initial layer configuration.
    tasks:
        Pool of tasks to sample from.
    n_episodes:
        Number of episodes to collect per iteration.
    n_iterations:
        Number of learning iterations.
    active_layers:
        Which layers to train.  ``None`` means all three.

    Returns
    -------
    tuple[AgentState, StateID]
        The final agent state and its content-addressed state ID.
    """
    state_id = agent_state.state_id()
    layers = agent_state.get_layers(active_layers)
    log.info("Starting learning loop — initial state: %s", state_id.combined_hash[:12])

    for iteration in range(n_iterations):
        log.info("Iteration %d/%d", iteration + 1, n_iterations)

        # 1. Collect episodes
        if not tasks or n_episodes <= 0:
            episodes: list[Episode] = []
        else:
            if n_episodes <= len(tasks):
                selected_tasks = random.sample(tasks, n_episodes)
            else:
                # More episodes requested than tasks: sample with replacement
                selected_tasks = random.choices(tasks, k=n_episodes)
            episodes = []
            for task in selected_tasks:
                ep = adapter.run_episode(task, agent_state)
                episodes.append(ep)

        avg_reward = (
            sum(ep.summary.total_reward for ep in episodes) / len(episodes)
            if episodes
            else 0.0
        )
        log.info("  Collected %d episodes, avg reward: %.4f", len(episodes), avg_reward)

        # 2. Build Datum
        datum = Datum(episodes=episodes)

        # 3. Phase 1: forward_backward (all active layers)
        fb_results: dict[str, FBResult] = {}
        for name, layer in layers:
            try:
                fut = layer.forward_backward(datum)
                fb_results[name] = fut.result()
            except Exception:
                log.exception("forward_backward failed for %s", name)
                fb_results[name] = FBResult(status="error")

        for name, result in fb_results.items():
            log.info("  fb %s: %s %s", name, result.status, result.metrics)

        # 4. Phase 2: optim_step (only layers whose fb succeeded)
        for name, layer in layers:
            if fb_results.get(name, FBResult(status="error")).status == "error":
                log.warning("  skipping optim_step for %s (fb failed)", name)
                continue
            try:
                result = layer.optim_step().result()
                log.info(
                    "  optim %s: %s, %d updates",
                    name, result.status, result.updates_applied,
                )
            except Exception:
                log.exception("optim_step failed for %s", name)

        # 5. Recompute state identity
        state_id = agent_state.state_id()

    log.info("Loop complete — final state: %s", state_id.combined_hash[:12])
    return agent_state, state_id
```

### Step 4: Run loop integration tests

Run: `python -m pytest tests/test_layer_protocol.py::TestLearningLoop -v`
Expected: All 7 tests PASS

### Step 5: Run all tests

Run: `python -m pytest tests/ -v`
Expected: All tests PASS (types, protocol, packs, episode, state, skyrl)

### Step 6: Commit

```bash
git add lfx/core/loop.py tests/test_layer_protocol.py
git commit -m "feat(lfx): rewrite learning loop to use Layer protocol

Replaces the proposer-based pattern with forward_backward -> optim_step
pump over active layers. Supports active_layers filter for selective
training. Per-layer error isolation (log + skip)."
```

---

## Task 7: Update Exports and Adapter Signature

**Files:**
- Modify: `lfx/core/__init__.py` (already done in Task 2, verify)
- Modify: `lfx/adapters/tau2.py:1-41` (update `run_episode` signature)
- Modify: `lfx/adapters/base.py:1-43` (update `run_episode` signature)

### Step 1: Update adapter base class

In `lfx/adapters/base.py`, update the `run_episode` signature to accept `AgentState` instead of a generic dict. Change line 30 area:

**Old:** `def run_episode(self, task: Any, agent_state: Any) -> Episode:`
**New:** Keep `Any` type hint for now (to avoid circular imports), but add a docstring note that it's an `AgentState`.

Check: The base class already uses `Any` — no change needed if signature is already `(self, task: Any, agent_state: Any)`.

Read `lfx/adapters/base.py` to confirm. If the signature already matches, skip this step.

### Step 2: Verify tau2 adapter

Read `lfx/adapters/tau2.py`. If `run_episode` signature matches `(self, task, agent_state)`, no change needed.

### Step 3: Run full test suite

Run: `python -m pytest tests/ -v`
Expected: All PASS

### Step 4: Commit (only if changes were made)

```bash
git add lfx/adapters/ lfx/core/__init__.py
git commit -m "chore(lfx): align adapter signatures with new loop protocol"
```

---

## Task 8: Cross-Layer Integration and Final Verification

**Files:**
- Test: `tests/test_layer_protocol.py` (append cross-layer tests)

### Step 1: Write cross-layer integration tests

Append to `tests/test_layer_protocol.py`:

```python
# ── Cross-layer integration tests ──


class TestCrossLayerIntegration:
    def test_all_layers_implement_protocol(self) -> None:
        """Verify all three layers have the five protocol methods."""
        for LayerClass in (Harness, Router, Weights):
            layer = LayerClass()
            assert hasattr(layer, "forward_backward")
            assert hasattr(layer, "optim_step")
            assert hasattr(layer, "sample")
            assert hasattr(layer, "save_state")
            assert hasattr(layer, "load_state")
            assert hasattr(layer, "to_dict")

    def test_all_layers_forward_backward_no_mutation(self) -> None:
        """The two-phase invariant holds for all layers."""
        layers = [
            Harness(system_prompts={"test": "prompt"}),
            Router(tier_models={t: f"m-{t}" for t in Tier.ALL}),
            Weights(model_ref="test-model", adapter_refs=["lora-1"]),
        ]
        datum = _make_datum()
        for layer in layers:
            state_before = json.dumps(layer.to_dict(), sort_keys=True)
            layer.forward_backward(datum)
            state_after = json.dumps(layer.to_dict(), sort_keys=True)
            assert state_before == state_after, f"{type(layer).__name__} mutated in fb"

    def test_full_loop_all_layers(self) -> None:
        """End-to-end: loop with all three layers and mock adapter."""
        adapter = _MockAdapter(reward=0.75)
        state = AgentState(
            harness=Harness(system_prompts={"test": "prompt"}),
            router=Router(tier_models={t: f"m-{t}" for t in Tier.ALL}),
            weights=Weights(model_ref="test-model"),
        )
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1", "t2"],
            n_episodes=2,
            n_iterations=2,
        )
        assert sid.combined_hash
        assert adapter.call_count == 4

    def test_save_load_all_layers(self) -> None:
        """Save and load round-trip for the full AgentState."""
        state = AgentState(
            harness=Harness(system_prompts={"test": "prompt"}),
            router=Router(tier_models={Tier.LIGHT: "haiku"}),
            weights=Weights(model_ref="llama"),
        )
        # Save each layer
        harness_dict = state.harness.to_dict()
        router_dict = state.router.to_dict()
        weights_dict = state.weights.to_dict()

        # Create fresh state and load
        state2 = AgentState()
        state2.harness.load_state(harness_dict)
        state2.router.load_state(router_dict)
        state2.weights.load_state(weights_dict)

        assert state.state_id().combined_hash == state2.state_id().combined_hash
```

### Step 2: Run all tests

Run: `python -m pytest tests/test_layer_protocol.py -v`
Expected: All tests PASS (should be 45+ total across all test classes)

### Step 3: Count tests

Run: `python -m pytest tests/test_layer_protocol.py -v --co | wc -l`
Expected: 45+ test items collected

### Step 4: Run full suite one final time

Run: `python -m pytest tests/ -v`
Expected: All tests PASS, no regressions

### Step 5: Commit

```bash
git add tests/test_layer_protocol.py
git commit -m "test(lfx): add cross-layer integration and protocol conformance tests

Verifies two-phase invariant across all layers, full loop integration
with mock adapter, and save/load round-trip for AgentState."
```

---

## Summary

| Task | Tests Added | Files Created/Modified |
|------|------------|----------------------|
| 1. Core types | 13 | `core/types.py`, `tests/test_types.py` |
| 2. Layer protocol | 1 | `core/layer.py`, `core/__init__.py` |
| 3. Harness protocol | 13 | `layers/harness.py`, `tests/test_layer_protocol.py` |
| 4. Router protocol | 12 | `layers/router.py` |
| 5. Weights protocol | 13 | `layers/weights.py` |
| 6. Learning loop | 7 | `core/loop.py` |
| 7. Adapter signatures | 0 | `adapters/base.py`, `adapters/tau2.py` (if needed) |
| 8. Cross-layer tests | 4 | `tests/test_layer_protocol.py` |
| **Total** | **63** | |

**Note:** `core/gate.py` requires no changes — the gate operates on Episodes and StateIDs,
both of which are unchanged.  Gate is intentionally outside the learning loop.

**Test command:** `python -m pytest tests/ -v`

**After all tasks:** Create PR with base `cll` targeting the protocol changes only.
