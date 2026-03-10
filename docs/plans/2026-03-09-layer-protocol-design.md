# PR 2 — Layer Protocol: Tinker Verbs on All Three Layers

**Base branch:** `cll` (PR 1)
**Date:** 2026-03-09

## Goal

Replace the proposer-based learning pattern from PR 1 with a unified Layer
protocol inspired by Tinker. Every learning layer (Harness, Router, Weights)
speaks the same five verbs. The learning loop becomes a simple
`forward_backward → optim_step` pump over a list of layers.

## Scope

### IN this PR

| Area | Files | What changes |
|------|-------|-------------|
| Core types | `core/types.py` | `Future[T]`, `Datum`, result dataclasses |
| Layer protocol | `core/layer.py` | `Layer` protocol definition (5 verbs + `to_dict`) |
| Harness | `layers/harness.py` | Implements Layer — wires existing Pareto/playbook logic into two-phase contract |
| Router | `layers/router.py` | Implements Layer — wires existing scoring/weight-update into two phases |
| Weights | `layers/weights.py` | Implements Layer — GRPO advantage computation in `forward_backward`, pass-through `optim_step` |
| Learning loop | `core/loop.py` | Rewritten to use Layer protocol with Futures |
| Adapter fixes | `adapters/tau2.py` | Signature alignment with new `AdapterLike` |
| Glue | `core/__init__.py`, `core/gate.py`, `core/episode.py` | Minor adjustments |
| Tests | `tests/test_layer_protocol.py` | 45+ contract tests |

### OUT (separate PRs)

- `integrations/` (LiteLLM wrapper, observability callbacks)
- `exporters/langfuse.py`, `exporters/phoenix.py`
- Real SkyRL weight training in `optim_step` (needs Ray, GPU)
- Real LLM-based reflection in harness (needs API calls)

---

## Design

### 1. `Future[T]` (`core/types.py`)

Synchronous-first result wrapper. Most layers today are immediate; the Future
abstraction allows deferred completion for SkyRL/Tinker backends later.

```python
_UNSET = object()  # sentinel — None is a valid result

@dataclass
class Future(Generic[T]):
    _value: T | object = _UNSET
    _event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def result(self, timeout: float | None = None) -> T:
        """Block until resolved. Raises TimeoutError if timeout expires."""
        if not self._event.wait(timeout):
            raise TimeoutError(...)
        return self._value

    def set_result(self, value: T) -> None:
        """Resolve. Raises RuntimeError on double-set."""
        with self._lock:
            if self._value is not _UNSET:
                raise RuntimeError("Future already resolved")
            self._value = value
            self._event.set()

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @classmethod
    def immediate(cls, value: T) -> "Future[T]":
        f = cls()
        f.set_result(value)
        return f
```

**Design decisions:**
- `_UNSET` sentinel instead of `Optional` — `None` is a valid result value.
- `set_result` raises on double-set (idempotency = hidden bugs).
- Lock around set + event for thread-safety.
- `immediate()` class method is the common path today.

### 2. `Datum` (`core/types.py`)

Universal training atom. Each layer extracts what it needs.

```python
@dataclass(frozen=True)
class Datum:
    episodes: list[Episode]
    loss_fn: str = "auto"
    loss_fn_config: dict[str, Any] = field(default_factory=dict)
```

**`loss_fn` resolution when `"auto"`:** Each layer ignores the hint and uses its
native loss — Harness uses reflection, Router uses routing reward, Weights uses
GRPO. Explicit values override for experimentation.

### 3. Result Types (`core/types.py`)

Minimal typed containers for each verb's return value.

```python
@dataclass(frozen=True)
class FBResult:
    status: str                        # "ok" | "skipped" | "error"
    metrics: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class OptimResult:
    status: str                        # "ok" | "skipped" | "error"
    updates_applied: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SampleContext:
    bench: str = ""
    query_features: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SampleResult:
    output: Any = None                 # layer-specific (prompt, model_id, adapter)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class SaveResult:
    name: str = ""
    status: str = "ok"

@dataclass(frozen=True)
class LoadResult:
    status: str = "ok"
```

### 4. Layer Protocol (`core/layer.py`)

```python
class Layer(Protocol):
    def forward_backward(self, data: Datum) -> Future[FBResult]: ...
    def optim_step(self) -> Future[OptimResult]: ...
    def sample(self, ctx: SampleContext) -> Future[SampleResult]: ...
    def save_state(self, name: str) -> Future[SaveResult]: ...
    def load_state(self, state_dict: dict[str, Any]) -> Future[LoadResult]: ...
    def to_dict(self) -> dict[str, Any]: ...
```

### 5. Two-Phase Contract

This is the central design invariant.

**Phase 1 — `forward_backward(datum)`:**
- Computes learning signals from episodes.
- Writes results to a `_pending` accumulator (layer-internal).
- **MUST NOT mutate observable state** (system_prompts, score_weights, adapter_refs, playbook entries, etc.).
- Can be called N times before a single `optim_step`.
- Returns `Future[FBResult]` with metrics about what was computed.

**Phase 2 — `optim_step()`:**
- Drains the `_pending` accumulator.
- Applies mutations to observable state.
- Resets `_pending` to empty.
- Returns `Future[OptimResult]` with count of updates applied.

**Why:** GRPO needs N rollouts per prompt before one gradient step. The
Harness needs to see all episodes before deciding which playbook entries to
prune. Batching before applying is fundamental.

**Testing invariant:** For each layer:
```python
state_before = layer.to_dict()
layer.forward_backward(datum)
state_after = layer.to_dict()
assert state_before == state_after  # no observable mutation
```

### 6. Failure Semantics

- **Per-layer isolation:** If `forward_backward` fails for one layer, others
  still proceed. The failing layer's `_pending` stays empty.
- **No partial commits:** If `optim_step` fails, state doesn't change
  (`_pending` is not drained). The layer logs the error and returns
  `OptimResult(status="error")`.
- **Loop behavior:** Log + skip on layer errors. The loop continues with
  remaining layers. No rollback across layers.

### 7. Save/Load Semantics

- `save_state` captures **applied state only**, not pending accumulators.
- Rule: always call `save_state` after `optim_step`, never between phases.
- `load_state` restores applied state and clears any `_pending` accumulator.

### 8. Deterministic Serialization

`to_dict()` must produce deterministic output for `StateID` to work:
- Dict keys are sorted.
- Lists maintain stable insertion order (no sets in serialization).
- Floats are rounded to consistent precision.
- **Test:** `to_dict() → load_state() → to_dict()` must produce identical output.

### 9. Layer Implementations

#### Harness

`_pending` accumulates:
- `insights: list[Insight]` — playbook delta proposals
- `candidates: list[PromptCandidate]` — Pareto front additions
- `playbook_signals: dict[str, tuple[int, int]]` — (helpful, harmful) deltas per entry ID

`forward_backward`: Analyzes episodes, produces insights and candidates,
tallies playbook helpful/harmful signals. No LLM calls — uses existing
heuristic/rule-based logic from PR 1.

`optim_step`: Applies insights to playbook, adds candidates to Pareto front,
promotes best candidate to `system_prompts`, prunes low-score entries.

`sample(ctx)`: Returns `system_prompt[ctx.bench]` + `playbook.render()`.

#### Router

`_pending` accumulates:
- `samples: list[tuple[QueryFeatures, str, float, float]]` — (features, model_id, cost, reward)

`forward_backward`: Extracts routing tuples from episodes using existing
`QueryFeatures` extraction logic.

`optim_step`: Runs existing `update_weights()` on accumulated samples to
adjust `score_weights` and `tier_thresholds`.

`sample(ctx)`: Classifies query features → tier → model_id via existing
`route()` logic.

#### Weights

`_pending` accumulates:
- `advantages: list[tuple[str, float]]` — (episode_id, advantage) from GRPO computation

`forward_backward`: Groups episodes by task, computes per-group mean reward,
derives advantages. This is the "forward" part of GRPO — pure computation,
no gradient updates.

`optim_step`: **Pass-through** — records the training step in `training_history`
but does not call SkyRL. Returns `OptimResult(status="skipped", updates_applied=0)`.
Real training deferred to a later PR.

`sample(ctx)`: Returns `model_ref` + `active_adapter`.

### 10. Learning Loop (`core/loop.py`)

```python
def learning_loop(
    agent_state: AgentState,
    adapter: AdapterLike,
    tasks: list[Any],
    *,
    n_iterations: int = 1,
    n_episodes: int = 10,
    active_layers: list[str] | None = None,  # None = all
) -> tuple[AgentState, StateID]:

    layers = agent_state.active_layers(active_layers)

    for iteration in range(n_iterations):
        # 1. Collect episodes
        episodes = [adapter.run_episode(t, agent_state) for t in tasks[:n_episodes]]

        # 2. Build Datum
        datum = Datum(episodes=episodes)

        # 3. Phase 1: forward_backward (all layers)
        fb_futures = []
        for layer in layers:
            try:
                fb_futures.append(layer.forward_backward(datum))
            except Exception as e:
                logger.error("forward_backward failed for %s: %s", layer, e)
                fb_futures.append(Future.immediate(FBResult(status="error")))

        # Wait for all
        fb_results = [f.result() for f in fb_futures]

        # 4. Phase 2: optim_step (all layers)
        step_futures = []
        for layer in layers:
            try:
                step_futures.append(layer.optim_step())
            except Exception as e:
                logger.error("optim_step failed for %s: %s", layer, e)
                step_futures.append(Future.immediate(OptimResult(status="error")))

        step_results = [f.result() for f in step_futures]

        # 5. Recompute state identity
        state_id = agent_state.state_id()

    return agent_state, state_id
```

### 11. Test Plan (45+ tests)

**Future tests:**
- `test_future_immediate` — resolves instantly
- `test_future_deferred` — set_result then result
- `test_future_timeout` — raises TimeoutError
- `test_future_double_set` — raises RuntimeError
- `test_future_done_property`
- `test_future_thread_safety` — concurrent set/get

**Datum tests:**
- `test_datum_frozen`
- `test_datum_auto_loss_fn`
- `test_datum_with_config`

**Protocol contract tests (per layer x 3 = 15+):**
- `test_{layer}_forward_backward_returns_future`
- `test_{layer}_forward_backward_no_mutation`
- `test_{layer}_optim_step_applies_pending`
- `test_{layer}_optim_step_drains_pending`
- `test_{layer}_sample_returns_result`

**Save/load tests (per layer x 3 = 9+):**
- `test_{layer}_save_state`
- `test_{layer}_load_state`
- `test_{layer}_save_load_roundtrip`

**Two-phase invariant tests:**
- `test_multiple_forward_backward_then_one_optim`
- `test_optim_without_forward_is_noop`
- `test_forward_backward_does_not_mutate_state` (deep equality check)
- `test_save_between_phases_excludes_pending`

**Loop integration tests:**
- `test_loop_single_iteration`
- `test_loop_multiple_iterations`
- `test_loop_active_layers_filter`
- `test_loop_layer_failure_continues`
- `test_loop_state_id_changes_after_optim`

**Deterministic serialization tests:**
- `test_{layer}_to_dict_deterministic` (call twice, assert equal)
- `test_{layer}_to_dict_load_roundtrip`

**Result type tests:**
- `test_fb_result_frozen`
- `test_optim_result_frozen`
- `test_sample_result_frozen`
