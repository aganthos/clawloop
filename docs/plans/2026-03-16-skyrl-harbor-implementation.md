# SkyRL + Harbor Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Weights layer stub with real SkyRL training, add Harbor env support, and create a unified training script.

**Architecture:** Thin Tinker adapter pattern. `LfXBackend` protocol (identical to `Layer`) wraps SkyRL's `AbstractBackend` for weights and the existing Harness layer for context learning. `HarborTaskEnvironment` runs Harbor trials and produces LfX Episodes. Training script selects mode via YAML config.

**Tech Stack:** Python 3.11+, SkyRL (submodule), Harbor (optional dep), Pydantic (config), pytest, asyncio

**Spec:** `docs/plans/2026-03-16-skyrl-harbor-integration-design.md`

**Codex Review:** SHIP IT after 3 rounds. Key fixes applied:
- BackendError is a frozen dataclass (not an Exception); init failures use SkyRLBackendInitError(Exception)
- _to_forward_backward_input passes raw rewards + trajectory_ids (SkyRL computes advantages)
- Schema-aware translation test validates real ForwardBackwardInput type (skipped without SkyRL)
- Harbor exceptions handled by name check (optional dep)
- HarnessLearningBackend created but not wired in train.py (building block for future unified mode)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `lfx/backends/__init__.py` | Package exports |
| Create | `lfx/backends/base.py` | `LfXBackend` protocol + `BackendError` |
| Create | `lfx/backends/harness_learning.py` | `HarnessLearningBackend` + `HarnessLearningConfig` |
| Create | `lfx/backends/skyrl.py` | `SkyRLWeightsBackend` + `SkyRLWeightsConfig` |
| Create | `lfx/envs/__init__.py` | Package exports |
| Create | `lfx/envs/harbor.py` | `HarborTaskEnvironment` + `HarborAdapter` |
| Create | `lfx/utils/__init__.py` | Package init |
| Create | `lfx/utils/async_bridge.py` | `run_async()` helper |
| Create | `lfx/train.py` | Unified training entry point + `TrainConfig` |
| Modify | `lfx/layers/weights.py` | Add `_backend` delegation |
| Modify | `lfx/core/loop.py:109-118` | Add `inference_url` to `AgentState` |
| Create | `tests/test_backends.py` | Backend protocol + harness learning tests |
| Create | `tests/test_skyrl_backend.py` | SkyRL backend integration tests |
| Create | `tests/test_harbor_env.py` | Harbor env + adapter tests |
| Create | `tests/test_async_bridge.py` | async bridge tests |
| Create | `tests/test_train_config.py` | Training config + script tests |

---

## Chunk 1: Foundation (Backend Protocol + Utilities)

### Task 1: BackendError and LfXBackend protocol

**Files:**
- Create: `lfx/backends/__init__.py`
- Create: `lfx/backends/base.py`
- Test: `tests/test_backends.py`

- [ ] **Step 1: Write failing test for BackendError**

```python
# tests/test_backends.py
from lfx.backends.base import BackendError


class TestBackendError:
    def test_create_error(self):
        err = BackendError(code="gpu_oom", message="Out of memory", recoverable=True)
        assert err.code == "gpu_oom"
        assert err.recoverable is True

    def test_frozen(self):
        import pytest
        err = BackendError(code="unknown", message="x", recoverable=False)
        with pytest.raises(AttributeError):
            err.code = "changed"

    def test_from_exception_known(self):
        err = BackendError.from_exception(MemoryError("CUDA OOM"))
        assert err.code == "gpu_oom"
        assert err.recoverable is True

    def test_from_exception_unknown(self):
        err = BackendError.from_exception(ValueError("bad"))
        assert err.code == "unknown"
        assert err.recoverable is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends.py::TestBackendError -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement BackendError**

```python
# lfx/backends/base.py
"""LfXBackend protocol and BackendError."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from lfx.core.types import (
    Datum, FBResult, Future, LoadResult, OptimResult,
    SampleContext, SampleResult, SaveResult,
)


@dataclass(frozen=True)
class BackendError:
    """Structured error for backend failures."""

    code: str
    message: str
    recoverable: bool

    @classmethod
    def from_exception(cls, e: Exception) -> BackendError:
        _MAP = {
            MemoryError: ("gpu_oom", True),
            ImportError: ("import_error", False),
            ModuleNotFoundError: ("import_error", False),
        }
        for exc_type, (code, recoverable) in _MAP.items():
            if isinstance(e, exc_type):
                return cls(code=code, message=str(e), recoverable=recoverable)
        return cls(code="unknown", message=str(e), recoverable=False)


class LfXBackend(Protocol):
    """Unified backend protocol — identical to the Layer protocol."""

    def forward_backward(self, data: Datum) -> Future[FBResult]: ...
    def optim_step(self) -> Future[OptimResult]: ...
    def sample(self, ctx: SampleContext) -> Future[SampleResult]: ...
    def save_state(self, name: str) -> Future[SaveResult]: ...
    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]: ...
    def clear_pending_state(self) -> None: ...
    def to_dict(self) -> dict[str, Any]: ...
```

```python
# lfx/backends/__init__.py
"""LfX training backends."""

from lfx.backends.base import BackendError, LfXBackend

__all__ = ["BackendError", "LfXBackend"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends.py::TestBackendError -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lfx/backends/__init__.py lfx/backends/base.py tests/test_backends.py
git commit -m "feat: add LfXBackend protocol and BackendError"
```

---

### Task 2: Async bridge utility

**Files:**
- Create: `lfx/utils/__init__.py`
- Create: `lfx/utils/async_bridge.py`
- Test: `tests/test_async_bridge.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_async_bridge.py
import asyncio
from lfx.utils.async_bridge import run_async


class TestRunAsync:
    def test_runs_coroutine_from_sync(self):
        async def add(a, b):
            return a + b
        assert run_async(add(2, 3)) == 5

    def test_returns_value(self):
        async def identity(x):
            return x
        assert run_async(identity("hello")) == "hello"

    def test_propagates_exception(self):
        import pytest
        async def fail():
            raise ValueError("boom")
        with pytest.raises(ValueError, match="boom"):
            run_async(fail())

    def test_works_with_asyncio_gather(self):
        async def batch():
            async def item(i):
                return i * 2
            return await asyncio.gather(item(1), item(2), item(3))
        assert run_async(batch()) == [2, 4, 6]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_async_bridge.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement async bridge**

```python
# lfx/utils/__init__.py
"""LfX utilities."""

# lfx/utils/async_bridge.py
"""Safe async-to-sync bridge."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def run_async(coro):
    """Run an async coroutine from sync code safely.

    No event loop running: uses asyncio.run() (fast path).
    Event loop running (Jupyter, async orchestration): runs in a
    thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        future = _EXECUTOR.submit(asyncio.run, coro)
        return future.result()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_async_bridge.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lfx/utils/__init__.py lfx/utils/async_bridge.py tests/test_async_bridge.py
git commit -m "feat: add run_async sync/async bridge utility"
```

---

### Task 3: HarnessLearningBackend

**Files:**
- Create: `lfx/backends/harness_learning.py`
- Modify: `lfx/backends/__init__.py`
- Test: `tests/test_backends.py` (append)

- [ ] **Step 1: Write failing test**

Append to `tests/test_backends.py`:

```python
import json
from lfx.backends.harness_learning import HarnessLearningBackend, HarnessLearningConfig
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.types import Datum, SampleContext
from lfx.layers.harness import Harness


def _make_episode(reward=0.8):
    return Episode(
        id=Episode.new_id(), state_id="deadbeef", task_id="t1", bench="test",
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class TestHarnessLearningBackend:
    def test_forward_backward_delegates(self):
        h = Harness(system_prompts={"test": "prompt"})
        b = HarnessLearningBackend(h)
        datum = Datum(episodes=[_make_episode()])
        result = b.forward_backward(datum).result()
        assert result.status == "ok"

    def test_optim_step_delegates(self):
        h = Harness()
        b = HarnessLearningBackend(h)
        b.forward_backward(Datum(episodes=[_make_episode()]))
        result = b.optim_step().result()
        assert result.status == "ok"

    def test_sample_delegates(self):
        h = Harness(system_prompts={"bench": "You are an agent."})
        b = HarnessLearningBackend(h)
        result = b.sample(SampleContext(bench="bench")).result()
        assert "You are an agent." in result.output

    def test_to_dict_delegates(self):
        h = Harness(system_prompts={"test": "p"})
        b = HarnessLearningBackend(h)
        assert b.to_dict() == h.to_dict()

    def test_clear_pending_delegates(self):
        h = Harness()
        b = HarnessLearningBackend(h)
        b.forward_backward(Datum(episodes=[_make_episode()]))
        b.clear_pending_state()
        result = b.optim_step().result()
        assert result.updates_applied == 0

    def test_config_defaults(self):
        cfg = HarnessLearningConfig()
        assert cfg.reflector_enabled is True
        assert cfg.paradigm_enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backends.py::TestHarnessLearningBackend -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement HarnessLearningBackend**

```python
# lfx/backends/harness_learning.py
"""HarnessLearningBackend — wraps Harness layer as an LfXBackend."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lfx.core.types import (
    Datum, FBResult, Future, LoadResult, OptimResult,
    SampleContext, SampleResult, SaveResult,
)
from lfx.layers.harness import Harness


@dataclass
class HarnessLearningConfig:
    """Placeholder config for unified mode. Currently unused by the backend."""

    reflector_enabled: bool = True
    intensity_config: dict[str, Any] = field(default_factory=dict)
    paradigm_enabled: bool = True


class HarnessLearningBackend:
    """Wraps the Harness layer as an LfXBackend. Pure delegation."""

    def __init__(self, harness: Harness, config: HarnessLearningConfig | None = None):
        self._harness = harness
        self._config = config or HarnessLearningConfig()

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        return self._harness.forward_backward(data)

    def optim_step(self) -> Future[OptimResult]:
        return self._harness.optim_step()

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        return self._harness.sample(ctx)

    def save_state(self, name: str) -> Future[SaveResult]:
        return self._harness.save_state(name)

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        return self._harness.load_state(state)

    def clear_pending_state(self) -> None:
        self._harness.clear_pending_state()

    def to_dict(self) -> dict[str, Any]:
        return self._harness.to_dict()
```

Update `lfx/backends/__init__.py`:

```python
from lfx.backends.base import BackendError, LfXBackend
from lfx.backends.harness_learning import HarnessLearningBackend, HarnessLearningConfig

__all__ = ["BackendError", "LfXBackend", "HarnessLearningBackend", "HarnessLearningConfig"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_backends.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add lfx/backends/harness_learning.py lfx/backends/__init__.py tests/test_backends.py
git commit -m "feat: add HarnessLearningBackend"
```

---

## Chunk 2: Weights Layer Update + AgentState

### Task 4: Update Weights layer with backend delegation

**Files:**
- Modify: `lfx/layers/weights.py`
- Test: `tests/test_layer_protocol.py` (add tests)

- [ ] **Step 1: Write failing test for backend delegation**

Append to `tests/test_layer_protocol.py` inside `TestWeightsProtocol`:

```python
    def test_backend_forward_backward_delegates(self) -> None:
        """When _backend is set, forward_backward delegates."""
        from unittest.mock import MagicMock
        from lfx.core.types import Future, FBResult

        mock_backend = MagicMock()
        mock_backend.forward_backward.return_value = Future.immediate(
            FBResult(status="ok", metrics={"loss": 0.5})
        )
        w = Weights(model_ref="test", _backend=mock_backend)
        result = w.forward_backward(_make_datum()).result()
        assert result.status == "ok"
        assert result.metrics["loss"] == 0.5
        mock_backend.forward_backward.assert_called_once()

    def test_backend_optim_step_delegates(self) -> None:
        from unittest.mock import MagicMock
        from lfx.core.types import Future, OptimResult

        mock_backend = MagicMock()
        mock_backend.optim_step.return_value = Future.immediate(
            OptimResult(status="ok", updates_applied=1, metrics={"grad_norm": 0.1})
        )
        w = Weights(model_ref="test", _backend=mock_backend)
        result = w.optim_step().result()
        assert result.status == "ok"
        assert result.updates_applied == 1
        mock_backend.optim_step.assert_called_once()

    def test_no_backend_uses_stub(self) -> None:
        """Without backend, stub behavior unchanged."""
        w = Weights(model_ref="test")
        result = w.forward_backward(_make_datum()).result()
        assert result.status == "ok"
        assert "n_advantages" in result.metrics

    def test_backend_clear_pending_delegates(self) -> None:
        from unittest.mock import MagicMock
        mock_backend = MagicMock()
        w = Weights(_backend=mock_backend)
        w.clear_pending_state()
        mock_backend.clear_pending_state.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer_protocol.py::TestWeightsProtocol::test_backend_forward_backward_delegates -v`
Expected: FAIL (TypeError — _backend not a field yet)

- [ ] **Step 3: Update Weights layer**

Modify `lfx/layers/weights.py`. Key changes:
- Add `_backend` field (default `None`)
- Rename existing `forward_backward` to `_stub_forward_backward`
- Rename existing `optim_step` to `_stub_optim_step`
- New `forward_backward` / `optim_step` delegate to `_backend` or fall through to stub
- `clear_pending_state` delegates if backend set
- `sample`, `save_state`, `load_state`, `to_dict` delegate if backend set

See spec section 8 for the full delegation pattern.

- [ ] **Step 4: Run ALL existing tests to verify backward compat**

Run: `pytest tests/test_layer_protocol.py -v`
Expected: ALL PASS (existing tests use `Weights()` without backend = stub mode)

- [ ] **Step 5: Commit**

```bash
git add lfx/layers/weights.py tests/test_layer_protocol.py
git commit -m "feat: add backend delegation to Weights layer"
```

---

### Task 5: Add inference_url to AgentState

**Files:**
- Modify: `lfx/core/loop.py:109-118`
- Test: `tests/test_layer_protocol.py` (add test)

- [ ] **Step 1: Write failing test**

Append to `tests/test_layer_protocol.py` in `TestLearningLoop`:

```python
    def test_agent_state_inference_url(self) -> None:
        state = AgentState(inference_url="http://localhost:8000/v1")
        assert state.inference_url == "http://localhost:8000/v1"

    def test_agent_state_inference_url_default_none(self) -> None:
        state = AgentState()
        assert state.inference_url is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_layer_protocol.py::TestLearningLoop::test_agent_state_inference_url -v`
Expected: FAIL (TypeError — unexpected keyword)

- [ ] **Step 3: Add inference_url to AgentState**

In `lfx/core/loop.py`, modify the `AgentState` dataclass (line 109-118):

```python
@dataclass
class AgentState:
    """Bundle of the three mutable learning layers."""

    harness: Harness = field(default_factory=Harness)
    router: Router = field(default_factory=Router)
    weights: Weights = field(default_factory=Weights)
    inference_url: str | None = None  # vLLM endpoint for Harbor agents

    def state_id(self) -> StateID:
        return StateID.from_layers(self.harness, self.router, self.weights)

    def get_layers(
        self, active: list[str] | None = None,
    ) -> list[tuple[str, Any]]:
        """Return (name, layer) pairs, filtered by *active* if given."""
        all_layers = [(name, getattr(self, name)) for name in LAYER_NAMES]
        if active is None:
            return all_layers
        return [(n, l) for n, l in all_layers if n in active]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_layer_protocol.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add lfx/core/loop.py tests/test_layer_protocol.py
git commit -m "feat: add inference_url to AgentState"
```

---

## Chunk 3: SkyRLWeightsBackend

### Task 6: SkyRLWeightsBackend with mock backend

**Files:**
- Create: `lfx/backends/skyrl.py`
- Modify: `lfx/backends/__init__.py`
- Create: `tests/test_skyrl_backend.py`

This is the most complex task. The backend wraps SkyRL's `AbstractBackend` and translates Episode → GeneratorOutput → ForwardBackwardInput.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_skyrl_backend.py
"""Tests for SkyRLWeightsBackend using mock SkyRL backend."""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig
from lfx.backends.base import BackendError
from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.types import Datum, FBResult, Future, OptimResult, SampleContext


def _make_episode(task_id="t1", reward=0.8):
    return Episode(
        id=Episode.new_id(), state_id="abc", task_id=task_id, bench="test",
        messages=[
            Message(role="system", content="You are helpful."),
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ],
        step_boundaries=[0],
        steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
        summary=EpisodeSummary(total_reward=reward),
    )


class TestSkyRLWeightsConfig:
    def test_config_fields(self):
        cfg = SkyRLWeightsConfig(
            base_model="Qwen/Qwen3-8B",
            backend_type="jax",
            backend_config={"tensor_parallel_size": 4},
            lora_config={"rank": 32, "alpha": 16.0},
            training_config={"loss_fn": "ppo", "adam_params": {"learning_rate": 1e-6}},
            tokenizer_name="Qwen/Qwen3-8B",
        )
        assert cfg.base_model == "Qwen/Qwen3-8B"
        assert cfg.backend_type == "jax"


class TestSkyRLWeightsBackend:
    def test_forward_backward_calls_exporter_and_backend(self):
        """Verifies the Episode → GeneratorOutput → backend pipeline."""
        cfg = SkyRLWeightsConfig(
            base_model="test-model", backend_type="jax",
            backend_config={}, lora_config={"rank": 8, "alpha": 16.0, "seed": 42},
            training_config={"loss_fn": "cross_entropy", "adam_params": {}},
            tokenizer_name="test",
        )
        # Mock the SkyRL backend and tokenizer
        mock_backend = MagicMock()
        mock_backend.forward_backward.return_value = {
            "req1": MagicMock(loss_fn_outputs={"loss": 0.5}, metrics={"kl": 0.01})
        }

        backend = SkyRLWeightsBackend.__new__(SkyRLWeightsBackend)
        backend._config = cfg
        backend._backend = mock_backend
        backend._model_id = "model-1"
        backend._adapter_refs = []
        backend.inference_url = None
        # Use FakeTokenizer from test_skyrl_export
        from tests.test_skyrl_export import FakeTokenizer
        backend._exporter = __import__("lfx.exporters.skyrl", fromlist=["SkyRLExporter"]).SkyRLExporter(tokenizer=FakeTokenizer())

        datum = Datum(episodes=[_make_episode()])
        result = backend.forward_backward(datum).result()
        assert result.status == "ok"
        mock_backend.forward_backward.assert_called_once()

    def test_optim_step_calls_backend(self):
        cfg = SkyRLWeightsConfig(
            base_model="test", backend_type="jax", backend_config={},
            lora_config={}, training_config={"adam_params": {"learning_rate": 1e-5}},
            tokenizer_name="test",
        )
        mock_backend = MagicMock()
        mock_backend.optim_step.return_value = MagicMock(
            metrics={"grad_norm": 0.1, "learning_rate": 1e-5}
        )

        backend = SkyRLWeightsBackend.__new__(SkyRLWeightsBackend)
        backend._config = cfg
        backend._backend = mock_backend
        backend._model_id = "model-1"
        backend._adapter_refs = []

        result = backend.optim_step().result()
        assert result.status == "ok"
        assert result.updates_applied == 1
        mock_backend.optim_step.assert_called_once()

    def test_to_dict_includes_all_config(self):
        cfg = SkyRLWeightsConfig(
            base_model="model-a", backend_type="jax",
            backend_config={"tp": 4}, lora_config={"rank": 32},
            training_config={"loss_fn": "ppo"}, tokenizer_name="tok",
        )
        backend = SkyRLWeightsBackend.__new__(SkyRLWeightsBackend)
        backend._config = cfg
        backend._adapter_refs = ["lora-v1"]

        d = backend.to_dict()
        assert d["model_ref"] == "model-a"
        assert d["backend_type"] == "jax"
        assert d["adapter_refs"] == ["lora-v1"]
        assert d["training_config"] == {"loss_fn": "ppo"}

    def test_clear_pending_is_noop(self):
        backend = SkyRLWeightsBackend.__new__(SkyRLWeightsBackend)
        backend._backend = MagicMock()
        backend.clear_pending_state()  # Should not raise

    def test_from_exception_maps_errors(self):
        err = BackendError.from_exception(MemoryError("CUDA"))
        assert err.code == "gpu_oom"
        err2 = BackendError.from_exception(ImportError("no skyrl"))
        assert err2.code == "import_error"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skyrl_backend.py -v`
Expected: FAIL (ImportError — skyrl.py not created yet)

- [ ] **Step 3: Implement SkyRLWeightsBackend**

```python
# lfx/backends/skyrl.py
"""SkyRLWeightsBackend — delegates to SkyRL's AbstractBackend."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from lfx.backends.base import BackendError
from lfx.core.types import (
    Datum, FBResult, Future, LoadResult, OptimResult,
    SampleContext, SampleResult, SaveResult,
)
from lfx.exporters.skyrl import SkyRLExporter

log = logging.getLogger(__name__)


@dataclass
class SkyRLWeightsConfig:
    """Config for the SkyRL weights backend. Dicts pass through to SkyRL."""

    base_model: str = ""
    backend_type: str = "jax"  # "jax" or "skyrl_train"
    backend_config: dict[str, Any] = field(default_factory=dict)
    lora_config: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str = ""


class SkyRLWeightsBackend:
    """Wraps SkyRL AbstractBackend for weight training.

    All SkyRL config passes through as dicts — LfX does not interpret
    SkyRL's configuration knobs.
    """

    def __init__(self, config: SkyRLWeightsConfig):
        self._config = config
        self._adapter_refs: list[str] = []
        self.inference_url: str | None = None

        # 1. Validate SkyRL imports
        try:
            from skyrl.tinker.types import ForwardBackwardInput  # noqa: F401
            from skyrl.backends.backend import AbstractBackend  # noqa: F401
        except ImportError as e:
            raise BackendError(code="import_error", message=str(e), recoverable=False)

        # 2. Load tokenizer
        try:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
            # Validate chat template works
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}], tokenize=True,
            )
        except Exception as e:
            raise BackendError(
                code="tokenizer_mismatch", message=str(e), recoverable=False,
            )

        # 3. Instantiate SkyRL backend
        self._backend = self._create_backend(config)
        self._model_id = f"lfx-{config.base_model.replace('/', '-')}"

        # 4. Create model (LoRA adapter + optimizer)
        from skyrl.tinker.types import LoraConfig as SkyRLLoraConfig
        lora = SkyRLLoraConfig(**config.lora_config)
        self._backend.create_model(self._model_id, lora)

        # 5. Inference URL
        if config.backend_config.get("enable_http_endpoint"):
            self.inference_url = getattr(self._backend, "get_inference_url", lambda: None)()

        # 6. Exporter
        self._exporter = SkyRLExporter(tokenizer=self._tokenizer)

        log.info("SkyRLWeightsBackend ready: model=%s backend=%s",
                 config.base_model, config.backend_type)

    def _create_backend(self, config: SkyRLWeightsConfig):
        if config.backend_type == "jax":
            from skyrl.backends.jax import JaxBackend, JaxBackendConfig
            bc = JaxBackendConfig(**config.backend_config)
            return JaxBackend(config.base_model, bc)
        elif config.backend_type == "skyrl_train":
            from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend
            return SkyRLTrainBackend(config.base_model, config.backend_config)
        else:
            raise BackendError(
                code="invalid_config",
                message=f"Unknown backend_type: {config.backend_type}",
                recoverable=False,
            )

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        try:
            gen_output = self._exporter.export(data.episodes)
            prepared = self._to_forward_backward_input(gen_output)
            result = self._backend.forward_backward(prepared)
            metrics = {}
            for req_id, output in result.items():
                if hasattr(output, "metrics"):
                    metrics.update(output.metrics)
                if hasattr(output, "loss_fn_outputs"):
                    metrics.update(output.loss_fn_outputs)
            return Future.immediate(FBResult(status="ok", metrics=metrics))
        except Exception as e:
            error = BackendError.from_exception(e)
            return Future.immediate(FBResult(status="error", metrics={"error": error}))

    def optim_step(self) -> Future[OptimResult]:
        try:
            from skyrl.tinker.types import OptimStepInput, AdamParams
            adam_cfg = self._config.training_config.get("adam_params", {})
            optim_input = OptimStepInput(
                adam_params=AdamParams(**adam_cfg),
            )
            result = self._backend.optim_step(self._model_id, optim_input)
            metrics = result.metrics if hasattr(result, "metrics") else {}
            return Future.immediate(OptimResult(
                status="ok", updates_applied=1, metrics=metrics,
            ))
        except Exception as e:
            error = BackendError.from_exception(e)
            return Future.immediate(OptimResult(
                status="error", updates_applied=0, metrics={"error": error},
            ))

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        return Future.immediate(SampleResult(
            output=self._config.base_model,
            metadata={"active_adapter": self._adapter_refs[-1] if self._adapter_refs else None},
        ))

    def save_state(self, name: str) -> Future[SaveResult]:
        try:
            path = f"checkpoints/{self._model_id}/{name}"
            self._backend.save_checkpoint(path, self._model_id)
            self._adapter_refs.append(path)
            return Future.immediate(SaveResult(name=name, status="ok"))
        except Exception:
            return Future.immediate(SaveResult(name=name, status="error"))

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        adapter_refs = state.get("adapter_refs", [])
        if adapter_refs:
            try:
                self._backend.load_checkpoint(adapter_refs[-1], self._model_id)
                self._adapter_refs = list(adapter_refs)
            except Exception:
                return Future.immediate(LoadResult(status="error"))
        return Future.immediate(LoadResult(status="ok"))

    def clear_pending_state(self) -> None:
        pass  # SkyRL manages its own gradient buffers

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_ref": self._config.base_model,
            "backend_type": self._config.backend_type,
            "backend_config": self._config.backend_config,
            "lora_config": self._config.lora_config,
            "training_config": self._config.training_config,
            "adapter_refs": self._adapter_refs,
        }

    def _to_forward_backward_input(self, gen_output: dict[str, Any]) -> Any:
        """Translate GeneratorOutput → SkyRL PreparedModelPassBatch.

        This is the glue code between LfX's Episode format and SkyRL's
        training input format. Uses only public SkyRL types.
        """
        from skyrl.tinker.types import ForwardBackwardInput
        loss_fn = self._config.training_config.get("loss_fn", "cross_entropy")
        loss_fn_config = self._config.training_config.get("loss_fn_config", {})

        # Build per-sequence inputs from GeneratorOutput
        data = []
        prompt_ids_list = gen_output.get("prompt_token_ids", [])
        response_ids_list = gen_output.get("response_ids", [])
        loss_masks_list = gen_output.get("loss_masks", [])
        rewards_list = gen_output.get("rewards", [])
        logprobs_list = gen_output.get("rollout_logprobs") or [None] * len(prompt_ids_list)

        for i in range(len(prompt_ids_list)):
            full_ids = prompt_ids_list[i] + response_ids_list[i]
            target_ids = response_ids_list[i]
            weights = loss_masks_list[i] if i < len(loss_masks_list) else [1] * len(target_ids)
            advantages = [rewards_list[i]] * len(target_ids) if i < len(rewards_list) else [0.0] * len(target_ids)
            logprobs = logprobs_list[i]

            data.append({
                "model_input": {"chunks": [{"tokens": full_ids}]},
                "loss_fn_inputs": {
                    "target_tokens": {"data": target_ids},
                    "weights": {"data": weights},
                    "advantages": {"data": advantages},
                    "logprobs": {"data": logprobs} if logprobs else None,
                },
            })

        return ForwardBackwardInput(
            data=data,
            model_id=self._model_id,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
```

Update `lfx/backends/__init__.py` to add exports.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_skyrl_backend.py -v`
Expected: PASS (tests use mocks, no real SkyRL needed)

- [ ] **Step 5: Commit**

```bash
git add lfx/backends/skyrl.py lfx/backends/__init__.py tests/test_skyrl_backend.py
git commit -m "feat: add SkyRLWeightsBackend"
```

---

## Chunk 4: Harbor Environment

### Task 7: HarborTaskEnvironment and HarborAdapter

**Files:**
- Create: `lfx/envs/__init__.py`
- Create: `lfx/envs/harbor.py`
- Create: `tests/test_harbor_env.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_harbor_env.py
"""Tests for HarborTaskEnvironment and HarborAdapter using mocks."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lfx.core.episode import Episode, RewardSignal
from lfx.core.loop import AgentState
from lfx.utils.async_bridge import run_async


class TestHarborTaskEnvironment:
    def test_task_id_from_dir_name(self):
        from lfx.envs.harbor import HarborTaskEnvironment

        with patch("lfx.envs.harbor.Trial"), patch("lfx.envs.harbor.TrialConfig"):
            env = HarborTaskEnvironment(
                task_dir=Path("/data/tasks/code-contest-42"),
                trial_config={"agent": {"name": "terminus-2", "kwargs": {}}},
            )
            assert env.task_id == "code-contest-42"

    def test_run_episode_builds_episode(self):
        from lfx.envs.harbor import HarborTaskEnvironment

        mock_results = MagicMock()
        mock_results.verifier_result.rewards = {"reward": 0.75}
        mock_results.agent_result.metadata = {
            "all_messages": [
                {"role": "user", "content": "Write hello world"},
                {"role": "assistant", "content": "print('hello world')"},
            ]
        }

        with patch("lfx.envs.harbor.Trial") as MockTrial, \
             patch("lfx.envs.harbor.TrialConfig"):
            mock_trial_instance = MagicMock()
            mock_trial_instance.run = AsyncMock(return_value=mock_results)
            MockTrial.return_value = mock_trial_instance

            env = HarborTaskEnvironment(
                task_dir=Path("/data/tasks/test-task"),
                trial_config={"agent": {"name": "terminus-2", "kwargs": {}}},
            )
            ep = run_async(env.run_episode(AgentState()))

            assert isinstance(ep, Episode)
            assert ep.task_id == "test-task"
            assert len(ep.messages) == 2
            assert ep.summary.signals["outcome"].value == 0.75
            assert ep.summary.filtered is False

    def test_timeout_produces_filtered_episode(self):
        from lfx.envs.harbor import HarborTaskEnvironment

        with patch("lfx.envs.harbor.Trial") as MockTrial, \
             patch("lfx.envs.harbor.TrialConfig"):
            mock_trial = MagicMock()
            mock_trial.run = AsyncMock(side_effect=TimeoutError("agent timeout"))
            MockTrial.return_value = mock_trial

            env = HarborTaskEnvironment(
                task_dir=Path("/data/tasks/slow-task"),
                trial_config={"agent": {"name": "terminus-2", "kwargs": {}}},
            )
            ep = run_async(env.run_episode(AgentState()))

            assert ep.summary.filtered is True
            assert "outcome" not in ep.summary.signals

    def test_reward_transform(self):
        from lfx.envs.harbor import HarborTaskEnvironment

        mock_results = MagicMock()
        mock_results.verifier_result.rewards = {"reward": 0.5}
        mock_results.agent_result.metadata = {"all_messages": []}

        with patch("lfx.envs.harbor.Trial") as MockTrial, \
             patch("lfx.envs.harbor.TrialConfig"):
            mock_trial = MagicMock()
            mock_trial.run = AsyncMock(return_value=mock_results)
            MockTrial.return_value = mock_trial

            env = HarborTaskEnvironment(
                task_dir=Path("/data/tasks/t1"),
                trial_config={"agent": {"name": "t2", "kwargs": {}}},
                reward_transform=lambda r: r * 2 - 1,  # [0,1] → [-1,1]
            )
            ep = run_async(env.run_episode(AgentState()))
            assert ep.summary.signals["outcome"].value == 0.0  # 0.5*2-1 = 0

    def test_config_validation_missing_agent(self):
        from lfx.envs.harbor import HarborTaskEnvironment
        with patch("lfx.envs.harbor.Trial"), patch("lfx.envs.harbor.TrialConfig"):
            with pytest.raises(ValueError, match="agent"):
                HarborTaskEnvironment(
                    task_dir=Path("/data/t"), trial_config={},
                )


class TestHarborAdapter:
    def test_run_episode_delegates(self):
        from lfx.envs.harbor import HarborAdapter, HarborTaskEnvironment

        mock_env = MagicMock(spec=HarborTaskEnvironment)
        mock_env.task_id = "task-1"
        mock_ep = MagicMock(spec=Episode)
        mock_env.run_episode = AsyncMock(return_value=mock_ep)

        adapter = HarborAdapter([mock_env])
        result = adapter.run_episode("task-1", AgentState())
        assert result is mock_ep

    def test_run_batch_concurrent(self):
        from lfx.envs.harbor import HarborAdapter, HarborTaskEnvironment

        mock_env = MagicMock(spec=HarborTaskEnvironment)
        mock_env.task_id = "task-1"
        mock_ep = MagicMock(spec=Episode)
        mock_env.run_episode = AsyncMock(return_value=mock_ep)

        adapter = HarborAdapter([mock_env])
        results = adapter.run_batch(["task-1", "task-1"], AgentState(), n_per_task=1)
        assert len(results) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_harbor_env.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement HarborTaskEnvironment and HarborAdapter**

See spec sections 5 and 6 for the full implementation. Key: lazy import of `harbor.trial.Trial` and `harbor.models.trial.config.TrialConfig`. For tests, these are mocked.

```python
# lfx/envs/__init__.py
"""LfX environment adapters."""

# lfx/envs/harbor.py
"""Harbor environment adapter — runs Harbor trials, produces LfX Episodes."""
# ... (implementation per spec)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_harbor_env.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add lfx/envs/__init__.py lfx/envs/harbor.py tests/test_harbor_env.py
git commit -m "feat: add HarborTaskEnvironment and HarborAdapter"
```

---

## Chunk 5: Training Script + Config

### Task 8: TrainConfig and training entry point

**Files:**
- Create: `lfx/train.py`
- Create: `tests/test_train_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_train_config.py
"""Tests for TrainConfig and the train() entry point."""

import pytest
from lfx.train import TrainConfig, HarborConfig


class TestTrainConfig:
    def test_weight_mode(self):
        cfg = TrainConfig(
            mode="weight",
            env_type="harbor",
            harbor=HarborConfig(task_dirs=["/data/tasks"]),
            skyrl={"base_model": "Qwen/Qwen3-8B", "backend_type": "jax"},
        )
        assert cfg.mode == "weight"

    def test_harness_learning_mode(self):
        cfg = TrainConfig(
            mode="harness_learning",
            env_type="harbor",
            harbor=HarborConfig(task_dirs=["/data/tasks"]),
        )
        assert cfg.mode == "harness_learning"

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            TrainConfig(mode="invalid", env_type="harbor")

    def test_weight_mode_requires_skyrl(self):
        with pytest.raises(ValueError):
            TrainConfig(mode="weight", env_type="harbor")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_config.py -v`
Expected: FAIL (ImportError)

- [ ] **Step 3: Implement TrainConfig and train()**

```python
# lfx/train.py
"""Unified training entry point."""
# See spec section 9 for full implementation.
# TrainConfig: Pydantic BaseModel with mode, env_type, harbor, skyrl, harness fields.
# train(): builds envs, backends, agent_state, runs learning_loop.
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_train_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add lfx/train.py tests/test_train_config.py
git commit -m "feat: add unified training script and TrainConfig"
```

---

### Task 9: CI compat gate test

**Files:**
- Create: `tests/test_skyrl_compat.py`

- [ ] **Step 1: Write compat test**

```python
# tests/test_skyrl_compat.py
"""CI compat gate: validates SkyRL submodule types are importable
and the Episode → GeneratorOutput → ForwardBackwardInput path works."""

import pytest

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.exporters.skyrl import SkyRLExporter


@pytest.mark.skipif(
    not _skyrl_available(), reason="SkyRL submodule not available"
)
class TestSkyRLCompat:
    def test_tinker_types_importable(self):
        from skyrl.tinker.types import ForwardBackwardInput, OptimStepInput
        assert ForwardBackwardInput is not None
        assert OptimStepInput is not None

    def test_backend_importable(self):
        from skyrl.backends.backend import AbstractBackend
        assert AbstractBackend is not None

    def test_full_translation_path(self):
        """Episode → GeneratorOutput via SkyRLExporter."""
        from tests.test_skyrl_export import FakeTokenizer

        ep = Episode(
            id="test-ep", state_id="abc", task_id="t1", bench="test",
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi!"),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=1.0, done=True, timing_ms=100.0)],
            summary=EpisodeSummary(total_reward=1.0),
        )
        exporter = SkyRLExporter(tokenizer=FakeTokenizer())
        output = exporter.export([ep])

        assert "prompt_token_ids" in output
        assert "response_ids" in output
        assert "loss_masks" in output
        assert "rewards" in output
        assert len(output["prompt_token_ids"]) > 0


def _skyrl_available() -> bool:
    try:
        import skyrl.tinker.types  # noqa: F401
        return True
    except ImportError:
        return False
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_skyrl_compat.py -v`
Expected: PASS (or skipped if SkyRL not available)

- [ ] **Step 3: Commit**

```bash
git add tests/test_skyrl_compat.py
git commit -m "feat: add SkyRL compat gate test"
```

---

### Task 10: Run full test suite and verify

- [ ] **Step 1: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: ALL PASS (existing tests unchanged, new tests pass)

- [ ] **Step 2: Verify backward compatibility**

Run: `pytest tests/test_layer_protocol.py -v`
Expected: ALL existing tests still pass (Weights without backend = stub mode)

- [ ] **Step 3: Final commit if any fixups needed**

```bash
git add -A && git commit -m "fix: test suite cleanup"
```
