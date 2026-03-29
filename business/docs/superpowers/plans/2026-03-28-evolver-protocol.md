# Evolver Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify ClawLoop's three harness learning mechanisms (Reflector, GEPA, Paradigm) behind an internal Evolver interface, keeping the Layer Protocol (forward_backward/optim_step) as the universal external API. Add lifecycle semantics (status, run_id, cancellation) for future long-running cloud backends.

**Architecture:** The Harness remains a Layer — same forward_backward/optim_step contract as Weights (Tinker) and Router. Internally, a pluggable Evolver replaces the Reflector slot. The existing Reflector + GEPA + Paradigm are wrapped as `LocalEvolver` — the community default. Richness lives in FBResult.info (standardized schema) and Harness management methods (get_archive, evolution_summary, cancel). No new external protocol.

**Key design principle (from Codex review):** Layer Protocol = transport boundary (lifecycle). Management methods = introspection boundary (richness). These are different concerns. The hyperagent's evolutionary complexity lives INSIDE forward_backward, not in a new protocol.

**Tech Stack:** Python 3.11+, pytest, dataclasses. No new dependencies.

**Constraint:** SkyRL/Tinker compatibility preserved — Weights layer untouched.

**Not backwards-compatible:** Repo is not yet public. Breaking changes to Harness constructor acceptable.

**Note:** This plan file is a local gitignored working file (docs/ is in .gitignore). NOT part of the repository.

---

## File Map

```
NEW FILES:
  clawloop/core/evolver.py           — Internal Evolver interface + result types + HarnessSnapshot
  clawloop/core/evolution_log.py     — EvolutionLog for (state, action, reward_delta) tracking
  clawloop/evolvers/__init__.py      — evolvers package
  clawloop/evolvers/local.py         — LocalEvolver wrapping Reflector + GEPA + Paradigm
  tests/test_evolver_protocol.py     — protocol conformance tests
  tests/test_local_evolver.py        — LocalEvolver produces same results as current pipeline
  tests/test_evolution_log.py        — EvolutionLog tracking tests

MODIFIED FILES:
  clawloop/layers/harness.py         — reflector slot → evolver slot, add management methods
  clawloop/core/loop.py              — GEPA block → delegates to evolver, add EvolutionLog
  clawloop/core/types.py             — standardize FBResult.info schema
  clawloop/__init__.py               — export new public types
```

---

## Chunk 1: Evolver Interface + Lifecycle Types

### Task 1: Define internal Evolver interface and lifecycle types

**Files:**
- Create: `clawloop/core/evolver.py`
- Modify: `clawloop/core/types.py` (standardize FBResult.info)
- Test: `tests/test_evolver_protocol.py`

**Context:** The Evolver is an INTERNAL interface (not exported as public API). The Layer Protocol stays the external contract. We add lifecycle semantics to FBResult so forward_backward can return "running" for long-lived cloud backends. Management methods on Harness provide rich introspection.

- [ ] **Step 1: Write protocol conformance test**

```python
# tests/test_evolver_protocol.py
"""Tests for internal Evolver interface and lifecycle types."""

from clawloop.core.evolver import (
    Evolver,
    EvolverContext,
    EvolverResult,
    HarnessSnapshot,
    Provenance,
)


class StubEvolver:
    """Minimal evolver that does nothing — proves interface is satisfiable."""

    def evolve(self, episodes, harness_state, context):
        return EvolverResult()

    def name(self):
        return "stub"


def test_stub_satisfies_interface():
    e = StubEvolver()
    result = e.evolve(
        episodes=[],
        harness_state=HarnessSnapshot(
            system_prompts={},
            playbook_entries=[],
            pareto_fronts={},
            playbook_generation=0,
            playbook_version=0,
        ),
        context=EvolverContext(
            reward_history=[],
            is_stagnating=False,
            iteration=0,
        ),
    )
    assert isinstance(result, EvolverResult)
    assert result.insights == []
    assert result.candidates == {}
    assert result.paradigm_shift is False
    assert result.run_id == ""
    assert e.name() == "stub"


def test_harness_snapshot_serializable():
    snap = HarnessSnapshot(
        system_prompts={"default": "You are helpful."},
        playbook_entries=[{"id": "e1", "content": "Be concise", "helpful": 3, "harmful": 0}],
        pareto_fronts={"default": [{"text": "You are helpful.", "scores": {"t1": 0.8}}]},
        playbook_generation=5,
        playbook_version=12,
    )
    d = snap.to_dict()
    assert d["playbook_generation"] == 5
    assert len(d["playbook_entries"]) == 1


def test_evolver_result_with_all_fields():
    from clawloop.core.reflector import Insight

    result = EvolverResult(
        insights=[Insight(action="add", content="test insight", tags=["test"])],
        candidates={"default": []},
        paradigm_shift=True,
        deprecation_targets=["entry_1", "entry_2"],
        run_id="ev-abc123",
        provenance=Provenance(backend="test", version="0.1", tokens_used=100),
    )
    assert result.paradigm_shift is True
    assert len(result.deprecation_targets) == 2
    assert result.run_id == "ev-abc123"
    assert result.provenance.backend == "test"


def test_evolver_context_defaults():
    ctx = EvolverContext()
    assert ctx.is_stagnating is False
    assert ctx.iteration == 0
    assert ctx.max_tokens is None


def test_fb_info_schema():
    """FBResult.info should follow standardized schema for lifecycle."""
    from clawloop.core.evolver import make_fb_info

    info = make_fb_info(
        status="ok",
        run_id="ev-001",
        candidates_tested=28,
        best_score=0.85,
        backend="local",
    )
    assert info["info_version"] == 1
    assert info["status"] == "ok"
    assert info["run_id"] == "ev-001"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evolver_protocol.py -v`
Expected: FAIL — module doesn't exist yet.

- [ ] **Step 3: Implement `clawloop/core/evolver.py`**

```python
"""Internal Evolver interface — pluggable harness optimization backends.

NOT a public protocol. The external API is the Layer Protocol
(forward_backward/optim_step). This is the internal contract that
different optimization strategies implement within the Harness.

Layer Protocol = transport boundary (lifecycle).
Evolver = implementation boundary (algorithm).
Management methods on Harness = introspection boundary (richness).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from clawloop.core.episode import Episode
from clawloop.core.reflector import Insight
from clawloop.layers.harness import PromptCandidate


# ---------------------------------------------------------------------------
# Harness state snapshot (serializable for cloud evolvers)
# ---------------------------------------------------------------------------

@dataclass
class HarnessSnapshot:
    """Complete harness state for an Evolver to analyze."""

    system_prompts: dict[str, str]
    playbook_entries: list[dict[str, Any]]
    pareto_fronts: dict[str, list[dict[str, Any]]]
    playbook_generation: int
    playbook_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_prompts": self.system_prompts,
            "playbook_entries": self.playbook_entries,
            "pareto_fronts": self.pareto_fronts,
            "playbook_generation": self.playbook_generation,
            "playbook_version": self.playbook_version,
        }


# ---------------------------------------------------------------------------
# Evolver context and result
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    """Metadata about who produced this result and at what cost."""

    backend: str = ""
    version: str = ""
    tokens_used: int = 0
    latency_ms: float = 0.0
    seed: int | None = None


@dataclass
class EvolverContext:
    """Context beyond the current episode batch."""

    reward_history: list[float] = field(default_factory=list)
    is_stagnating: bool = False
    iteration: int = 0
    tried_paradigms: list[str] = field(default_factory=list)
    max_tokens: int | None = None
    max_candidates: int | None = None


@dataclass
class EvolverResult:
    """What an Evolver returns — can touch all three harness mechanisms.

    For synchronous (local) evolvers: fully populated, run_id empty.
    For async (cloud) evolvers: may be partial, run_id set for polling.
    """

    insights: list[Insight] = field(default_factory=list)
    candidates: dict[str, list[PromptCandidate]] = field(default_factory=dict)
    paradigm_shift: bool = False
    deprecation_targets: list[str] = field(default_factory=list)
    run_id: str = ""
    provenance: Provenance = field(default_factory=Provenance)


# ---------------------------------------------------------------------------
# Evolver interface (internal, not exported as public API)
# ---------------------------------------------------------------------------

class Evolver(Protocol):
    """Internal interface for harness optimization backends.

    Receives episode traces + full harness state, returns holistic
    improvements across playbook, prompts, and paradigm.

    Implementations: LocalEvolver (community), CloudEvolver (enterprise).
    """

    def evolve(
        self,
        episodes: list[Episode],
        harness_state: HarnessSnapshot,
        context: EvolverContext,
    ) -> EvolverResult: ...

    def name(self) -> str: ...


# ---------------------------------------------------------------------------
# Standardized FBResult.info schema
# ---------------------------------------------------------------------------

_INFO_VERSION = 1

# Valid lifecycle statuses for FBResult.info["status"]:
#   ok        — evolution complete, results in pending state
#   running   — long-running evolution in progress (cloud backends)
#   paused    — waiting for user input (interactive candidate selection)
#   failed    — evolution failed, see info["error"]
#   cancelled — evolution was cancelled via harness.cancel()
VALID_STATUSES = ("ok", "running", "paused", "failed", "cancelled")


def make_fb_info(
    *,
    status: str = "ok",
    run_id: str = "",
    summary: str = "",
    candidates_tested: int = 0,
    best_score: float | None = None,
    archive_size: int = 0,
    paradigm_shifted: bool = False,
    backend: str = "",
    tokens_used: int = 0,
    progress: float | None = None,
    error: str = "",
) -> dict[str, Any]:
    """Build a standardized FBResult.info dict.

    Schema is versioned via info_version so clients can evolve.
    """
    info: dict[str, Any] = {
        "info_version": _INFO_VERSION,
        "status": status,
        "run_id": run_id,
    }
    if summary:
        info["summary"] = summary
    if candidates_tested:
        info["candidates_tested"] = candidates_tested
    if best_score is not None:
        info["best_score"] = best_score
    if archive_size:
        info["archive_size"] = archive_size
    if paradigm_shifted:
        info["paradigm_shifted"] = True
    if backend:
        info["backend"] = backend
    if tokens_used:
        info["tokens_used"] = tokens_used
    if progress is not None:
        info["progress"] = progress
    if error:
        info["error"] = error
    return info
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_evolver_protocol.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add clawloop/core/evolver.py tests/test_evolver_protocol.py
git commit -m "feat: internal Evolver interface with lifecycle types and FBResult.info schema"
```

---

## Chunk 2: LocalEvolver — Wrap Existing Mechanisms

### Task 2: Implement LocalEvolver

**Files:**
- Create: `clawloop/evolvers/__init__.py`
- Create: `clawloop/evolvers/local.py`
- Test: `tests/test_local_evolver.py`

**Context:** LocalEvolver wraps the existing Reflector + PromptEvolver + ParadigmBreakthrough into a single Evolver. It's the community default. Synchronous — always returns status="ok" with empty run_id.

- [ ] **Step 1: Write test — LocalEvolver returns EvolverResult**

Create `tests/test_local_evolver.py` with tests that verify:
- LocalEvolver returns EvolverResult from reflector input
- LocalEvolver.name() returns "local"
- Without reflector, returns empty result
- With paradigm + stagnation, returns paradigm_shift=True

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_local_evolver.py -v`
Expected: FAIL

- [ ] **Step 3: Create `clawloop/evolvers/__init__.py`**

```python
"""Evolver backends — pluggable harness optimization strategies."""
from clawloop.evolvers.local import LocalEvolver
__all__ = ["LocalEvolver"]
```

- [ ] **Step 4: Implement `clawloop/evolvers/local.py`**

LocalEvolver wraps Reflector (playbook insights), PromptEvolver (GEPA mutation/crossover), and ParadigmBreakthrough (stagnation escape). Delegates to each based on context and action mask. Returns unified EvolverResult. See full implementation in earlier plan version — adapt to use the current Reflector.reflect() and PromptEvolver.mutate() signatures exactly as they exist in the codebase.

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_local_evolver.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add clawloop/evolvers/ tests/test_local_evolver.py
git commit -m "feat: LocalEvolver wrapping Reflector + GEPA + Paradigm"
```

---

## Chunk 3: Wire Evolver into Harness + Management Methods

### Task 3: Replace reflector slot with evolver on Harness, add management methods

**Files:**
- Modify: `clawloop/layers/harness.py`
- Modify: `clawloop/core/loop.py`
- Modify: all test files constructing `Harness(reflector=...)`

**Context:** The Harness constructor changes from `reflector=` to `evolver=`. forward_backward calls evolver.evolve() internally. The GEPA block and paradigm block move from the loop into the Evolver. Add management methods for future introspection. FBResult.info follows the standardized schema.

- [ ] **Step 1: Update Harness constructor — replace reflector with evolver**

In `clawloop/layers/harness.py`:
- Remove `reflector: Reflector | None = None` parameter
- Add `evolver: Evolver | None = None` parameter (import from `clawloop.core.evolver`)
- Store as `self._evolver = evolver`
- Keep `self.reflector` as a read-only property for backwards compat in tests if needed

- [ ] **Step 2: Update forward_backward to use evolver**

Replace the reflector call block with:
- Build `HarnessSnapshot` via `self._build_snapshot()`
- Build `EvolverContext` from intensity state
- Call `self._evolver.evolve(episodes, snapshot, context)`
- Store result in pending state
- Build FBResult with standardized info via `make_fb_info()`

- [ ] **Step 3: Add `_build_snapshot()` helper to Harness**

Serializes current system_prompts, playbook entries, pareto_fronts, generation/version into a `HarnessSnapshot`.

- [ ] **Step 4: Update optim_step to handle pending candidates and paradigm deprecation**

Wire `_pending.candidates` (currently unused): iterate and call `front.add()` for each.
If `_pending.paradigm_deprecation_targets` is set, increase decay_rate on those entries.

- [ ] **Step 5: Add management methods to Harness**

```python
def evolution_summary(self, run_id: str = "") -> dict[str, Any]:
    """Return summary of last/current evolution. Stubs for local, rich for cloud."""
    return {"backend": self._evolver.name() if self._evolver else "none"}

def get_candidates(self, bench: str = "") -> list[dict[str, Any]]:
    """Return current Pareto front candidates for inspection."""
    if bench and bench in self.pareto_fronts:
        return [{"text": c.text, "scores": c.per_task_scores} for c in self.pareto_fronts[bench].candidates]
    return []

def cancel(self, run_id: str = "") -> bool:
    """Cancel a running evolution. No-op for local evolvers."""
    return False
```

- [ ] **Step 6: Remove GEPA block and paradigm block from learning loop**

In `clawloop/core/loop.py`:
- Remove the GEPA evolution block (lines ~335-369) — now inside LocalEvolver
- Remove the paradigm breakthrough block (lines ~372-395) — now inside LocalEvolver
- Pass reward_history, is_stagnating, iteration, tried_paradigms to Harness via a setter or through the EvolverContext

- [ ] **Step 7: Update all test files that construct Harness**

Run: `grep -rn "reflector=" tests/` and update each to use `evolver=LocalEvolver(reflector=...)` or `evolver=None`.

- [ ] **Step 8: Run full test suite**

Run: `pytest tests/ -x -v --tb=short`
Expected: ALL PASS

- [ ] **Step 9: Commit**

```bash
git add clawloop/layers/harness.py clawloop/core/loop.py tests/
git commit -m "feat: replace reflector with Evolver on Harness, add management methods"
```

---

## Chunk 4: EvolutionLog

### Task 4: Implement EvolutionLog for (state, action, reward_delta) tracking

**Files:**
- Create: `clawloop/core/evolution_log.py`
- Modify: `clawloop/core/loop.py`
- Test: `tests/test_evolution_log.py`

**Context:** Append-only JSONL log capturing per-iteration evolution data. Each entry: state hash before, actions taken, state hash after, reward delta. Seeds future learned evolvers.

- [ ] **Step 1: Write test**

Test that EvolutionEntry is serializable and EvolutionLog writes JSONL.

- [ ] **Step 2: Run test to verify it fails**

- [ ] **Step 3: Implement `clawloop/core/evolution_log.py`**

EvolutionEntry dataclass + EvolutionLog append-only JSONL writer.

- [ ] **Step 4: Wire into learning loop**

After each iteration's optim_step, log the entry.

- [ ] **Step 5: Run tests**

- [ ] **Step 6: Commit**

```bash
git add clawloop/core/evolution_log.py tests/test_evolution_log.py clawloop/core/loop.py
git commit -m "feat: EvolutionLog tracks state/action/reward for learned evolvers"
```

---

## Chunk 5: Exports + Regression

### Task 5: Update exports and verify everything

- [ ] **Step 1: Update `clawloop/__init__.py`**

Add `LocalEvolver` to exports. Do NOT export internal types (Evolver, EvolverResult, HarnessSnapshot) — they're internal.

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -x -v --tb=short`

- [ ] **Step 3: Verify SkyRL/Tinker tests still pass**

Run: `pytest tests/test_skyrl_backend.py tests/test_skyrl_compat.py -v`

- [ ] **Step 4: Run audit**

Run: `bash scripts/audit_public.sh`

- [ ] **Step 5: Commit**

```bash
git add clawloop/__init__.py
git commit -m "feat: export LocalEvolver from clawloop"
```

---

## Post-Implementation Checklist

- [ ] All tests pass: `pytest tests/ -x -q`
- [ ] SkyRL/Tinker compat: `pytest tests/test_skyrl_backend.py tests/test_skyrl_compat.py -v`
- [ ] Audit clean: `bash scripts/audit_public.sh`
- [ ] Adapter tests: `pytest tests/test_adapter_*.py -v`
- [ ] Layer Protocol preserved: Harness still satisfies Layer protocol (forward_backward/optim_step/sample/save_state/load_state/clear_pending_state/to_dict)
