# Layer Protocol Hardening Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden learner.py and loop.py with status-aware FB checking and cross-layer two-phase commit with rollback, so that a failure in any layer's `optim_step` rolls back all layers atomically.

**Architecture:** Two changes: (1) Restructure `AsyncLearner._learn()` from per-layer sequential (FB+optim per layer) to proper two-phase (all FB first, check statuses, then all optim with cross-layer rollback). (2) Add cross-layer atomicity to `loop.py` — snapshot all layer state via `to_dict()` before the optim phase, rollback all via `load_state()` if any layer fails.

**Tech Stack:** Python, pytest. Uses existing Layer protocol methods (`to_dict`, `load_state`, `clear_pending_state`) — no new dependencies.

**Design decisions:**
- **FB failure policy:** Per-layer skip, not full-batch abort. If harness FB fails but router FB succeeds, router still gets its optim_step. Layers are independent in the accumulate phase. Cross-layer atomicity only applies to the optim (apply) phase.
- **Pending state cleanup:** `clear_pending_state()` is called on FB failure — both on exception AND when FB returns `status="error"` or `"skipped"`. This prevents stale pending data from leaking into subsequent batches.
- **Rollback mechanism:** `to_dict()` snapshot + `load_state()` restore. We call `.result()` on the `load_state()` Future and check `lr.status` for protocol correctness. Snapshot failures abort the optim phase (no valid rollback = no safe optim). Rollback failures are logged at ERROR level.
- **No external side effects:** All three layers are purely in-memory. `optim_step()` mutates Python objects only — no disk, network, or cache writes. Rollback via `load_state()` is therefore fully atomic.
- **All-FB-failed semantics:** If all layers return FB error/skipped, the batch is silently skipped (no `batches_failed` increment). This is not a failure — it means there was nothing to learn from (e.g., all reflector calls timed out). `batches_failed` only increments on optim-phase failures, which represent actual state corruption risk.
- **Test synchronization:** Tests use `time.sleep()` for waiting on the background learner thread, consistent with the existing test suite pattern. The learner processes batches in a single sequential thread, so 0.5-1.0s is reliably sufficient.

**Branch:** `feat/layer-protocol-hardening` (from `main`)

---

## Chunk 1: Learner Hardening

### Task 1: Two-phase status-aware learner with cross-layer rollback

**Files:**
- Modify: `lfx/learner.py:101-137` (`_learn` method)
- Test: `tests/test_learner.py`

**Context:** Currently `learner.py:119-121` calls `optim_step()` unconditionally after `forward_backward()`, and does FB+optim per layer sequentially. Three bugs: (1) doesn't check FBResult.status before optim, (2) wrong phase ordering, (3) no cross-layer rollback.

- [ ] **Step 1: Write failing test — FB error skips optim**

In `tests/test_learner.py`, add to `TestAsyncLearner`:

```python
def test_fb_error_skips_optim(self) -> None:
    """When forward_backward returns status='error', optim_step must NOT run."""
    from lfx.core.types import FBResult, Future

    state = AgentState()
    optim_called = False
    original_optim = state.harness.optim_step

    def tracking_optim():
        nonlocal optim_called
        optim_called = True
        return original_optim()

    def failing_fb(data):
        return Future.immediate(FBResult(status="error"))

    state.harness.forward_backward = failing_fb
    state.harness.optim_step = tracking_optim

    learner = AsyncLearner(agent_state=state, active_layers=["harness"])
    learner.start()
    learner.on_batch(_make_episodes(2))

    time.sleep(0.5)
    learner.stop()

    assert not optim_called, "optim_step should not run when FB returns error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_learner.py::TestAsyncLearner::test_fb_error_skips_optim -v`
Expected: FAIL — current code calls optim_step regardless of FB status.

- [ ] **Step 3: Write failing test — FB skipped also skips optim**

```python
def test_fb_skipped_skips_optim(self) -> None:
    """When forward_backward returns status='skipped', optim_step must NOT run."""
    from lfx.core.types import FBResult, Future

    state = AgentState()
    optim_called = False
    original_optim = state.harness.optim_step

    def tracking_optim():
        nonlocal optim_called
        optim_called = True
        return original_optim()

    def skipping_fb(data):
        return Future.immediate(FBResult(status="skipped"))

    state.harness.forward_backward = skipping_fb
    state.harness.optim_step = tracking_optim

    learner = AsyncLearner(agent_state=state, active_layers=["harness"])
    learner.start()
    learner.on_batch(_make_episodes(2))

    time.sleep(0.5)
    learner.stop()

    assert not optim_called, "optim_step should not run when FB returns skipped"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `pytest tests/test_learner.py::TestAsyncLearner::test_fb_skipped_skips_optim -v`
Expected: FAIL

- [ ] **Step 5: Write failing test — two-phase ordering (all FB before any optim)**

```python
def test_two_phase_ordering(self) -> None:
    """All forward_backward calls must complete before any optim_step runs."""
    state = AgentState()
    call_order: list[str] = []

    original_harness_fb = state.harness.forward_backward
    original_harness_optim = state.harness.optim_step
    original_router_fb = state.router.forward_backward
    original_router_optim = state.router.optim_step

    def harness_fb(data):
        call_order.append("harness_fb")
        return original_harness_fb(data)

    def harness_optim():
        call_order.append("harness_optim")
        return original_harness_optim()

    def router_fb(data):
        call_order.append("router_fb")
        return original_router_fb(data)

    def router_optim():
        call_order.append("router_optim")
        return original_router_optim()

    state.harness.forward_backward = harness_fb
    state.harness.optim_step = harness_optim
    state.router.forward_backward = router_fb
    state.router.optim_step = router_optim

    learner = AsyncLearner(
        agent_state=state, active_layers=["harness", "router"],
    )
    learner.start()
    learner.on_batch(_make_episodes(2))

    time.sleep(0.5)
    learner.stop()

    fb_indices = [i for i, c in enumerate(call_order) if c.endswith("_fb")]
    optim_indices = [i for i, c in enumerate(call_order) if c.endswith("_optim")]
    if fb_indices and optim_indices:
        assert max(fb_indices) < min(optim_indices), (
            f"Two-phase violated: {call_order}"
        )
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_learner.py::TestAsyncLearner::test_two_phase_ordering -v`
Expected: FAIL — current code does FB+optim per layer, not all-FB-then-all-optim.

- [ ] **Step 7: Write test — optim returning status="error" triggers rollback**

```python
def test_optim_error_status_triggers_rollback(self) -> None:
    """When optim_step returns status='error' (no exception), layers are rolled back."""
    import json
    from lfx.core.types import OptimResult, Future

    state = AgentState()
    harness_before = json.dumps(state.harness.to_dict(), sort_keys=True)

    def error_router_optim():
        return Future.immediate(OptimResult(status="error", updates_applied=0))

    state.router.optim_step = error_router_optim

    learner = AsyncLearner(
        agent_state=state, active_layers=["harness", "router"],
    )
    learner.start()
    learner.on_batch(_make_episodes(2))

    time.sleep(0.5)
    learner.stop()

    harness_after = json.dumps(state.harness.to_dict(), sort_keys=True)
    assert harness_after == harness_before, (
        "Harness should be rolled back when router optim returns error"
    )
    assert learner.metrics["batches_failed"] >= 1
```

- [ ] **Step 8: Write test — pending state cleared on FB error (non-exception)**

```python
def test_fb_error_clears_pending_state(self) -> None:
    """When FB returns status='error', clear_pending_state is called."""
    from lfx.core.types import FBResult, Future

    state = AgentState()
    clear_called = False
    original_clear = state.harness.clear_pending_state

    def tracking_clear():
        nonlocal clear_called
        clear_called = True
        original_clear()

    def error_fb(data):
        return Future.immediate(FBResult(status="error"))

    state.harness.forward_backward = error_fb
    state.harness.clear_pending_state = tracking_clear

    learner = AsyncLearner(agent_state=state, active_layers=["harness"])
    learner.start()
    learner.on_batch(_make_episodes(2))

    time.sleep(0.5)
    learner.stop()

    assert clear_called, "clear_pending_state should be called when FB returns error"
```

- [ ] **Step 9: Write test — all-FB-failed is not a batch failure**

```python
def test_all_fb_failed_not_counted_as_batch_failure(self) -> None:
    """When all layers return FB error/skipped, batches_failed stays 0."""
    from lfx.core.types import FBResult, Future

    state = AgentState()

    def error_fb(data):
        return Future.immediate(FBResult(status="error"))

    def skipped_fb(data):
        return Future.immediate(FBResult(status="skipped"))

    state.harness.forward_backward = error_fb
    state.router.forward_backward = skipped_fb

    learner = AsyncLearner(
        agent_state=state, active_layers=["harness", "router"],
    )
    learner.start()
    learner.on_batch(_make_episodes(2))

    time.sleep(0.5)
    learner.stop()

    assert learner.metrics["batches_failed"] == 0, (
        "All-FB-failed should not increment batches_failed"
    )
    assert learner.metrics["batches_trained"] == 0
```

- [ ] **Step 10: Write test — pending state cleared on FB skipped (non-exception)**

```python
def test_fb_skipped_clears_pending_state(self) -> None:
    """When FB returns status='skipped', clear_pending_state is called."""
    from lfx.core.types import FBResult, Future

    state = AgentState()
    clear_called = False
    original_clear = state.harness.clear_pending_state

    def tracking_clear():
        nonlocal clear_called
        clear_called = True
        original_clear()

    def skipped_fb(data):
        return Future.immediate(FBResult(status="skipped"))

    state.harness.forward_backward = skipped_fb
    state.harness.clear_pending_state = tracking_clear

    learner = AsyncLearner(agent_state=state, active_layers=["harness"])
    learner.start()
    learner.on_batch(_make_episodes(2))

    time.sleep(0.5)
    learner.stop()

    assert clear_called, "clear_pending_state should be called when FB returns skipped"
```

- [ ] **Step 11: Write test — cross-layer rollback in learner**

This test needs `_MockLLMClient` added to `test_learner.py` (copy from `test_loop_icl.py`) and imports for `Harness`, `Reflector`, `ReflectorConfig`.

```python
def test_optim_failure_rolls_back_all_layers(self) -> None:
    """When router optim fails, harness changes are rolled back."""
    import json
    from lfx.core.reflector import Reflector, ReflectorConfig

    client = _MockLLMClient()
    reflector = Reflector(client=client, config=ReflectorConfig())
    harness = Harness(
        system_prompts={"test": "You are helpful."},
        reflector=reflector,
    )
    state = AgentState(harness=harness)

    harness_before = json.dumps(state.harness.to_dict(), sort_keys=True)

    def failing_router_optim():
        raise RuntimeError("simulated optim failure")

    state.router.optim_step = failing_router_optim

    learner = AsyncLearner(
        agent_state=state, active_layers=["harness", "router"],
    )
    learner.start()
    learner.on_batch(_make_episodes(2))

    time.sleep(1.0)
    learner.stop()

    harness_after = json.dumps(state.harness.to_dict(), sort_keys=True)
    assert harness_after == harness_before, (
        "Harness should be rolled back when router optim fails"
    )
    assert learner.metrics["batches_failed"] >= 1
```

- [ ] **Step 10: Run tests to verify they fail**

Run: `pytest tests/test_learner.py -v -k "error_skips or skipped_skips or two_phase or rolls_back or error_status or clears_pending"`
Expected: FAIL — current code has none of these behaviors.

- [ ] **Step 11: Implement full two-phase `_learn()` with status checks and rollback**

Replace `lfx/learner.py` `_learn` method (lines 101-137) with:

```python
def _learn(self, episodes: list) -> None:
    batch_id = uuid.uuid4().hex[:8]

    rewards = [ep.summary.normalized_reward() for ep in episodes]
    avg_reward = mean(rewards) if rewards else 0.0
    self.intensity.record_reward(avg_reward)

    log.info(
        "Batch %s: %d episodes, avg_reward=%.3f",
        batch_id, len(episodes), avg_reward,
    )

    datum = Datum(episodes=episodes)

    # Phase 1: forward_backward all layers, collect results
    fb_results: dict[str, FBResult] = {}
    layers: list[tuple[str, Any]] = []
    for name in self.active_layers:
        layer = getattr(self.agent_state, name, None)
        if layer is None:
            continue
        layers.append((name, layer))
        try:
            fb_result = layer.forward_backward(datum).result()
            fb_results[name] = fb_result
            if fb_result.status in ("error", "skipped"):
                try:
                    layer.clear_pending_state()
                except Exception:
                    log.exception(
                        "Failed to clear pending state for %s", name,
                    )
        except Exception as exc:
            log.error(
                "forward_backward failed for %s on batch %s: %s",
                name, batch_id, exc,
            )
            fb_results[name] = FBResult(status="error")
            try:
                layer.clear_pending_state()
            except Exception:
                log.exception(
                    "Failed to clear pending state for %s", name,
                )

    # Phase 2: optim_step with cross-layer rollback
    layers_to_optim = [
        (name, layer) for name, layer in layers
        if fb_results.get(name, FBResult(status="error")).status
        not in ("error", "skipped")
    ]

    if not layers_to_optim:
        log.warning("Batch %s: no layers to optim (all FB error/skipped)", batch_id)
        return

    # Snapshot for rollback
    snapshots: dict[str, dict[str, Any]] = {}
    try:
        for name, layer in layers_to_optim:
            snapshots[name] = layer.to_dict()
    except Exception:
        log.exception("Snapshot failed — aborting optim for batch %s", batch_id)
        for name, layer in layers_to_optim:
            try:
                layer.clear_pending_state()
            except Exception:
                log.exception("Failed to clear pending state for %s", name)
        self._batches_failed += 1
        return

    optim_failed = False
    for name, layer in layers_to_optim:
        try:
            result = layer.optim_step().result()
            if result.status == "error":
                log.error(
                    "optim_step returned error for %s on batch %s",
                    name, batch_id,
                )
                optim_failed = True
                break
        except Exception as exc:
            log.error(
                "optim_step failed for %s on batch %s: %s",
                name, batch_id, exc,
            )
            optim_failed = True
            break

    if optim_failed:
        log.warning(
            "Rolling back all layers to pre-optim state for batch %s",
            batch_id,
        )
        for name, layer in layers_to_optim:
            if name in snapshots:
                try:
                    lr = layer.load_state(snapshots[name]).result()
                    if lr.status != "ok":
                        log.error(
                            "Rollback returned %s for %s", lr.status, name,
                        )
                except Exception:
                    log.exception("Rollback failed for %s", name)
        self._batches_failed += 1
        return

    self._batches_trained += 1
    self._iteration += 1
```

- [ ] **Step 14: Run all learner tests**

Run: `pytest tests/test_learner.py -v`
Expected: ALL PASS

- [ ] **Step 15: Commit**

```bash
git add lfx/learner.py tests/test_learner.py
git commit -m "feat: two-phase status-aware learner with cross-layer rollback"
```

---

## Chunk 2: Loop Hardening

### Task 2: Cross-layer rollback in loop.py

**Files:**
- Modify: `lfx/core/loop.py:214-248` (Phase 1 pending cleanup + Phase 2 optim_step section)
- Test: `tests/test_loop_icl.py`

**Context:** `loop.py` already does all-FB-then-all-optim and checks status. Two gaps: (1) doesn't call `clear_pending_state()` when FB returns error/skipped without raising, (2) no cross-layer rollback if one layer's optim_step fails after another succeeds.

- [ ] **Step 1: Write failing test — optim failure triggers cross-layer rollback**

In `tests/test_loop_icl.py`, add:

```python
class TestCrossLayerRollback:
    """When one layer's optim_step fails, all layers should rollback."""

    def test_optim_failure_rolls_back_all_layers(self) -> None:
        client = _MockLLMClient()
        reflector = Reflector(client=client, config=ReflectorConfig())
        harness = Harness(
            system_prompts={"test": "You are helpful."},
            reflector=reflector,
        )
        state = AgentState(harness=harness)

        # Capture harness state before learning
        harness_before = json.dumps(state.harness.to_dict(), sort_keys=True)

        # Make router.optim_step fail after harness succeeds
        def failing_router_optim():
            raise RuntimeError("simulated optim failure")

        state.router.optim_step = failing_router_optim

        adapter = _MockAdapter(reward=0.8)
        state, sid = learning_loop(
            adapter=adapter,
            agent_state=state,
            tasks=["t1"],
            n_episodes=1,
            n_iterations=1,
            active_layers=["harness", "router"],
        )

        # Harness should be rolled back to pre-optim state
        harness_after = json.dumps(state.harness.to_dict(), sort_keys=True)
        assert harness_after == harness_before, (
            "Harness should be rolled back when router optim fails"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_loop_icl.py::TestCrossLayerRollback -v`
Expected: FAIL — harness changes persist when router fails.

- [ ] **Step 3: Implement pending cleanup + cross-layer rollback in loop.py**

In `learning_loop()`, update Phase 1 to clear pending on non-exception error/skipped, and replace Phase 2 with snapshot+rollback:

**Phase 1 update** — after `fb_results[name] = fut.result()`, add pending cleanup for error/skipped:
```python
            try:
                fut = layer.forward_backward(datum)
                fb_result = fut.result()
                fb_results[name] = fb_result
                if fb_result.status in ("error", "skipped"):
                    try:
                        layer.clear_pending_state()
                    except Exception:
                        log.exception("Failed to clear pending for %s", name)
            except Exception:
                log.exception("forward_backward failed for %s", name)
                fb_results[name] = FBResult(status="error")
                try:
                    layer.clear_pending_state()
                except Exception:
                    log.exception("Failed to clear pending for %s", name)
```

**Phase 2 replacement** (lines ~235-247):
```python
        # 4. Phase 2: optim_step with cross-layer rollback
        layers_to_optim = [
            (name, layer) for name, layer in layers
            if fb_results.get(name, FBResult(status="error")).status
            not in ("error", "skipped")
        ]

        # Snapshot all layers before optim (for cross-layer rollback)
        snapshots: dict[str, dict[str, Any]] = {}
        try:
            for name, layer in layers_to_optim:
                snapshots[name] = layer.to_dict()
        except Exception:
            log.exception("Snapshot failed — skipping optim this iteration")
            for name, layer in layers_to_optim:
                try:
                    layer.clear_pending_state()
                except Exception:
                    log.exception("Failed to clear pending for %s", name)
            layers_to_optim = []

        optim_failed = False
        for name, layer in layers_to_optim:
            try:
                result = layer.optim_step().result()
                log.info(
                    "  optim %s: %s, %d updates",
                    name, result.status, result.updates_applied,
                )
                if result.status == "error":
                    optim_failed = True
                    log.error(
                        "  optim %s returned error — triggering rollback", name,
                    )
                    break
            except Exception:
                log.exception(
                    "optim_step failed for %s — triggering rollback", name,
                )
                optim_failed = True
                break

        if optim_failed:
            log.warning("  rolling back all layers to pre-optim state")
            for name, layer in layers_to_optim:
                if name in snapshots:
                    try:
                        lr = layer.load_state(snapshots[name]).result()
                        if lr.status != "ok":
                            log.error(
                                "  rollback returned %s for %s", lr.status, name,
                            )
                    except Exception:
                        log.exception("  rollback failed for %s", name)
```

- [ ] **Step 4: Run all loop tests**

Run: `pytest tests/test_loop_icl.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add lfx/core/loop.py tests/test_loop_icl.py
git commit -m "feat: cross-layer rollback in learning loop"
```

---

## Chunk 3: Integration Verification

### Task 3: Full regression check

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS, no regressions.

- [ ] **Step 2: Final commit if any adjustments needed**

```bash
git commit -m "fix: adjust tests for protocol hardening"
```
