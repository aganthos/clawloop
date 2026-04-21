"""Unit tests for ``clawloop.core.transaction.LayerTransaction``.

The transaction owns the two-phase ``forward_backward`` → ``optim_step``
protocol with cross-layer rollback. These tests pin the invariants that
the ``learning_loop`` depended on: which layers run, when
``clear_pending_state`` fires, and under what conditions every layer is
rolled back to its pre-optim snapshot.
"""

from __future__ import annotations

from typing import Any

from clawloop.core.loop import AgentState
from clawloop.core.transaction import LayerTransaction
from clawloop.core.types import Datum, FBResult, Future, LoadResult, OptimResult


class _StubLayer:
    """Hand-rolled Layer stub — records call order and returns canned results.

    Using a stub (not MagicMock) so assertions read like a state machine:
    who was called, with what, in what order. That is the exact contract
    the rollback invariant depends on.
    """

    def __init__(
        self,
        fb_result: FBResult | None = None,
        fb_raises: bool = False,
        optim_result: OptimResult | None = None,
        optim_raises: bool = False,
        load_result: LoadResult | None = None,
        load_raises: bool = False,
        state: dict[str, Any] | None = None,
    ) -> None:
        self._fb_result = fb_result or FBResult(status="ok", metrics={})
        self._fb_raises = fb_raises
        self._optim_result = optim_result or OptimResult(status="ok", updates_applied=1)
        self._optim_raises = optim_raises
        self._load_result = load_result or LoadResult(status="ok")
        self._load_raises = load_raises
        self._state = state or {"version": 1}

        self.fb_calls: list[Datum] = []
        self.optim_calls = 0
        self.load_calls: list[dict[str, Any]] = []
        self.clear_calls = 0

    def forward_backward(self, datum: Datum) -> Future[FBResult]:
        self.fb_calls.append(datum)
        if self._fb_raises:
            raise RuntimeError("fb boom")
        return Future.immediate(self._fb_result)

    def optim_step(self) -> Future[OptimResult]:
        self.optim_calls += 1
        if self._optim_raises:
            raise RuntimeError("optim boom")
        return Future.immediate(self._optim_result)

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        self.load_calls.append(state)
        if self._load_raises:
            raise RuntimeError("load boom")
        return Future.immediate(self._load_result)

    def clear_pending_state(self) -> None:
        self.clear_calls += 1

    def to_dict(self) -> dict[str, Any]:
        return dict(self._state)


def test_happy_path_runs_all_layers_and_no_rollback() -> None:
    h, r, w = _StubLayer(), _StubLayer(), _StubLayer()
    layers = [("harness", h), ("router", r), ("weights", w)]
    tx = LayerTransaction(layers, intensity=None, episodes=[], agent_state=AgentState())

    result = tx.run(iteration=0)

    assert result.optim_failed is False
    assert set(result.fb_results.keys()) == {"harness", "router", "weights"}
    assert all(r.status == "ok" for r in result.fb_results.values())
    # Every layer saw both fb and optim; nothing was rolled back or cleared
    assert all(layer.optim_calls == 1 for layer in (h, r, w))
    assert all(layer.load_calls == [] for layer in (h, r, w))
    assert all(layer.clear_calls == 0 for layer in (h, r, w))


def test_fb_exception_clears_pending_and_skips_that_layer_from_optim() -> None:
    h = _StubLayer(fb_raises=True)
    w = _StubLayer()
    layers = [("harness", h), ("weights", w)]
    tx = LayerTransaction(layers, intensity=None, episodes=[], agent_state=AgentState())

    result = tx.run(iteration=0)

    # Harness fb raised → status=error, clear_pending_state called, no optim
    assert result.fb_results["harness"].status == "error"
    assert h.clear_calls == 1
    assert h.optim_calls == 0
    # Weights is independent and proceeds through optim normally
    assert result.fb_results["weights"].status == "ok"
    assert w.optim_calls == 1
    # The failing layer wasn't rolled back (it never reached optim)
    assert h.load_calls == []
    assert result.optim_failed is False


def test_fb_skipped_status_also_clears_pending() -> None:
    # fb_result.status == "skipped" must trigger clear_pending_state
    # (matches the original in-loop behavior where skipped fb also clears)
    h = _StubLayer(fb_result=FBResult(status="skipped"))
    layers = [("harness", h)]
    tx = LayerTransaction(layers, intensity=None, episodes=[], agent_state=AgentState())

    tx.run(iteration=0)

    assert h.clear_calls == 1
    assert h.optim_calls == 0


def test_optim_error_rolls_back_all_snapshotted_layers() -> None:
    # When any optim returns status="error", EVERY layer that had a
    # successful fb gets load_state called with its pre-optim snapshot.
    h = _StubLayer(state={"v": "h-before"})
    r = _StubLayer(
        state={"v": "r-before"},
        optim_result=OptimResult(status="error"),  # trigger rollback
    )
    w = _StubLayer(state={"v": "w-before"})
    layers = [("harness", h), ("router", r), ("weights", w)]
    tx = LayerTransaction(layers, intensity=None, episodes=[], agent_state=AgentState())

    result = tx.run(iteration=0)

    assert result.optim_failed is True
    # Rollback fires for every layer that was snapshotted — even weights,
    # whose optim never ran because the loop broke on router's error
    assert h.load_calls == [{"v": "h-before"}]
    assert r.load_calls == [{"v": "r-before"}]
    assert w.load_calls == [{"v": "w-before"}]


def test_optim_exception_rolls_back_all_snapshotted_layers() -> None:
    # Same rollback contract when optim_step raises (not just returns error)
    h = _StubLayer(state={"v": "h"})
    r = _StubLayer(state={"v": "r"}, optim_raises=True)
    layers = [("harness", h), ("router", r)]
    tx = LayerTransaction(layers, intensity=None, episodes=[], agent_state=AgentState())

    result = tx.run(iteration=0)

    assert result.optim_failed is True
    assert h.load_calls == [{"v": "h"}]
    assert r.load_calls == [{"v": "r"}]


def test_snapshot_failure_skips_optim_and_clears_all() -> None:
    # If deepcopy(layer.to_dict()) raises, optim must not run for anyone,
    # and every layer that had ok fb gets clear_pending_state called.
    class _BadToDict(_StubLayer):
        def to_dict(self) -> dict[str, Any]:
            raise RuntimeError("cannot serialize")

    h = _BadToDict()
    w = _StubLayer()
    layers = [("harness", h), ("weights", w)]
    tx = LayerTransaction(layers, intensity=None, episodes=[], agent_state=AgentState())

    result = tx.run(iteration=0)

    assert h.optim_calls == 0
    assert w.optim_calls == 0
    assert h.clear_calls == 1
    assert w.clear_calls == 1
    # No rollback happened because no optim ran
    assert h.load_calls == []
    assert w.load_calls == []
    # optim_failed stays False — this is a snapshot failure, not an optim failure
    assert result.optim_failed is False


def test_intensity_skips_harness_reflection_but_other_layers_proceed() -> None:
    class _NoReflect:
        def should_reflect(self, iteration: int) -> bool:
            return False

    h = _StubLayer()
    w = _StubLayer()
    layers = [("harness", h), ("weights", w)]
    tx = LayerTransaction(layers, intensity=_NoReflect(), episodes=[], agent_state=AgentState())

    result = tx.run(iteration=3)

    # Harness fb was skipped entirely — no call, no clear (clear fires on
    # error/skipped result; "intensity gate" short-circuits before fb runs
    # and records skipped status)
    assert h.fb_calls == []
    assert result.fb_results["harness"].status == "skipped"
    assert h.optim_calls == 0
    # Weights proceeds normally
    assert w.fb_calls != []
    assert w.optim_calls == 1


def test_paradigm_shift_on_harness_fb_mutates_tried_paradigms() -> None:
    # When harness fb returns metrics["paradigm_shifted"]=True and the
    # harness has _pending insights tagged "paradigm", those contents
    # must be appended to agent_state.tried_paradigms BEFORE optim runs
    # (optim drains _pending, so ordering matters).
    agent_state = AgentState()
    # Seed _pending with a paradigm-tagged insight via the public helper
    from clawloop.learning_layers.harness import Insight, _HarnessPending

    agent_state.harness._pending = _HarnessPending(
        insights=[Insight(content="shift-to-x", tags=["paradigm"])],
    )
    assert [i.content for i in agent_state.harness.pending_paradigm_insights()] == ["shift-to-x"]

    class _HarnessShiftLayer(_StubLayer):
        def __init__(self) -> None:
            super().__init__(fb_result=FBResult(status="ok", metrics={"paradigm_shifted": True}))

    # The transaction must resolve paradigm tracking against
    # agent_state.harness, not the stub layer passed as ("harness", ...).
    layers = [("harness", _HarnessShiftLayer())]
    tx = LayerTransaction(layers, intensity=None, episodes=[], agent_state=agent_state)

    tx.run(iteration=0)

    assert "shift-to-x" in agent_state.tried_paradigms
