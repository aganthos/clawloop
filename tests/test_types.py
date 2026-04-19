"""Tests for clawloop.core.types — Future, Datum, result dataclasses."""

import threading

import pytest

from clawloop.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)


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


from clawloop.core.layer import Layer


class TestLayerProtocol:
    def test_protocol_has_required_methods(self) -> None:
        """Verify the Protocol defines all five verbs + to_dict."""
        import inspect

        members = {name for name, _ in inspect.getmembers(Layer) if not name.startswith("_")}
        required = {
            "forward_backward",
            "optim_step",
            "sample",
            "save_state",
            "load_state",
            "to_dict",
            "clear_pending_state",
        }
        assert required.issubset(members)
