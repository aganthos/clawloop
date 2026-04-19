"""Unit tests for TinkerWeightsConfig dataclass and TinkerWeightsBackend.

These tests verify the config holds non-secret fields only, has the
preflight-verified defaults, and never accidentally serializes credentials.
The backend tests cover __init__ + current_sampling_client (Task 8) and
the Layer-protocol methods (Task 9); they must NOT hit the network — every
Tinker SDK call is monkey-patched.
"""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# tinker / tinker_cookbook are optional extras (pyproject.toml [games]).
# Skip the whole module on CI runs that don't install them.
pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook")

from clawloop.core.types import Datum, SampleContext
from clawloop.weight_backends._tinker_sdk import TinkerBackendError
from clawloop.weight_backends.base import BackendError
from clawloop.weight_backends.tinker import TinkerWeightsConfig


def test_config_defaults():
    cfg = TinkerWeightsConfig(base_model="Qwen/Qwen3-8B")
    assert cfg.lora_rank == 8
    assert cfg.seed == 42
    assert cfg.train_attn is True
    assert cfg.train_mlp is True
    assert cfg.train_unembed is False
    assert cfg.loss_fn == "importance_sampling"
    # Verify adam_params has exactly the 4 supported keys.
    assert set(cfg.adam_params.keys()) == {"learning_rate", "beta1", "beta2", "eps"}
    assert "weight_decay" not in cfg.adam_params


def test_config_merges_partial_adam_params_with_defaults():
    """User-supplied ``adam_params={"learning_rate": 5e-6}`` keeps the other
    three required keys (beta1, beta2, eps) — otherwise ``AdamParams(**...)``
    would TypeError at optim_step time."""
    from clawloop.weight_backends.tinker import TinkerWeightsConfig

    cfg = TinkerWeightsConfig(
        base_model="Qwen/Qwen3-8B",
        adam_params={"learning_rate": 5e-6},
    )
    # All four required AdamParams kwargs present after the merge.
    assert cfg.adam_params["learning_rate"] == 5e-6  # user override kept
    assert cfg.adam_params["beta1"] == 0.9  # default filled in
    assert cfg.adam_params["beta2"] == 0.999
    assert cfg.adam_params["eps"] == 1e-8


def test_config_has_no_secret_fields():
    cfg = TinkerWeightsConfig(base_model="Qwen/Qwen3-8B")
    assert not hasattr(cfg, "api_key")
    assert not hasattr(cfg, "tinker_api_key")


def test_config_serialization_has_no_secrets():
    cfg = TinkerWeightsConfig(base_model="Qwen/Qwen3-8B")
    serialized = str(asdict(cfg)).lower()
    for forbidden in ("api_key", "secret", "token", "bearer"):
        assert forbidden not in serialized, f"{forbidden} found in config dict"


# ---------------------------------------------------------------------------
# TinkerWeightsBackend (Task 8): __init__ + current_sampling_client
# ---------------------------------------------------------------------------


def test_init_fails_without_api_key(monkeypatch):
    """Constructor must raise TinkerBackendError(missing_api_key) when env is unset."""
    monkeypatch.delenv("TINKER_API_KEY", raising=False)
    # Block all three lookup paths in clawloop.config.load_env so the real
    # clawloop/.env (which exists locally) cannot satisfy the check.
    monkeypatch.setenv("CLAWLOOP_ENV_FILE", "/nonexistent/path")
    monkeypatch.chdir("/tmp")
    import clawloop.config

    clawloop.config._loaded = False
    # Belt-and-braces: also no-op the load_env reference inside tinker.py so
    # the package-scoped clawloop/.env can never be picked up.
    monkeypatch.setattr("clawloop.weight_backends.tinker.load_env", lambda: [])

    from clawloop.weight_backends._tinker_sdk import TinkerBackendError
    from clawloop.weight_backends.tinker import (
        TinkerWeightsBackend,
        TinkerWeightsConfig,
    )

    with pytest.raises(TinkerBackendError) as excinfo:
        TinkerWeightsBackend(TinkerWeightsConfig(base_model="Qwen/Qwen3-8B"))
    assert excinfo.value.code == "missing_api_key"


def test_init_creates_training_and_sampling_clients(monkeypatch):
    """Verify constructor wires service/training/sampling/tokenizer/renderer."""
    monkeypatch.setenv("TINKER_API_KEY", "fake")

    fake_service = SimpleNamespace(name="service")
    fake_tokenizer = SimpleNamespace(name="tokenizer")
    fake_training = SimpleNamespace(
        get_tokenizer=lambda: fake_tokenizer,
    )
    fake_sampling = SimpleNamespace(name="sampling")
    fake_renderer = SimpleNamespace(name="renderer")

    create_training_calls: list[dict] = []
    create_sampling_calls: list[dict] = []
    get_renderer_calls: list[str] = []
    recommended_calls: list[str] = []

    def _make_service():
        return fake_service

    def _create_training(service, **kwargs):
        assert service is fake_service
        create_training_calls.append(kwargs)
        return fake_training

    def _create_sampling(service, **kwargs):
        assert service is fake_service
        create_sampling_calls.append(kwargs)
        return fake_sampling

    def _get_renderer(name, tokenizer):
        get_renderer_calls.append(name)
        return fake_renderer

    def _recommended(model):
        recommended_calls.append(model)
        return "qwen3"

    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.make_service_client",
        _make_service,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.create_training",
        _create_training,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.create_sampling",
        _create_sampling,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.make_rest_client",
        lambda service: SimpleNamespace(name="rest"),
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.get_model_id",
        lambda training: "fake-model-id",
    )
    monkeypatch.setattr("clawloop.weight_backends.tinker.get_renderer", _get_renderer)
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker.get_recommended_renderer_name",
        _recommended,
    )

    from clawloop.weight_backends.tinker import (
        TinkerWeightsBackend,
        TinkerWeightsConfig,
    )

    cfg = TinkerWeightsConfig(
        base_model="Qwen/Qwen3-8B",
        lora_rank=16,
        seed=7,
        train_attn=True,
        train_mlp=False,
        train_unembed=True,
    )
    backend = TinkerWeightsBackend(cfg)

    # create_training was called with full kwargs.
    assert len(create_training_calls) == 1
    kw = create_training_calls[0]
    assert kw["base_model"] == "Qwen/Qwen3-8B"
    assert kw["rank"] == 16
    assert kw["seed"] == 7
    assert kw["train_attn"] is True
    assert kw["train_mlp"] is False
    assert kw["train_unembed"] is True

    # create_sampling was called with base_model= (not model_path=).
    assert len(create_sampling_calls) == 1
    skw = create_sampling_calls[0]
    assert skw.get("base_model") == "Qwen/Qwen3-8B"
    assert "model_path" not in skw

    # Renderer auto-selected via get_recommended_renderer_name.
    assert recommended_calls == ["Qwen/Qwen3-8B"]
    assert get_renderer_calls == ["qwen3"]

    # Accessors expose what __init__ cached.
    assert backend.current_sampling_client() is fake_sampling
    assert backend.renderer is fake_renderer
    assert backend.tokenizer is fake_tokenizer
    assert backend.config is cfg


def test_init_uses_explicit_renderer_name_when_provided(monkeypatch):
    """If config.renderer_name is set, it overrides the recommended lookup."""
    monkeypatch.setenv("TINKER_API_KEY", "fake")

    fake_service = SimpleNamespace()
    fake_training = SimpleNamespace(get_tokenizer=lambda: SimpleNamespace())
    fake_sampling = SimpleNamespace()

    recommended_calls: list[str] = []
    get_renderer_calls: list[str] = []

    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.make_service_client",
        lambda: fake_service,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.create_training",
        lambda service, **kw: fake_training,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.create_sampling",
        lambda service, **kw: fake_sampling,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.make_rest_client",
        lambda service: SimpleNamespace(name="rest"),
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.get_model_id",
        lambda training: "fake-model-id",
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker.get_recommended_renderer_name",
        lambda m: recommended_calls.append(m) or "should-not-be-used",
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker.get_renderer",
        lambda name, tokenizer: get_renderer_calls.append(name) or SimpleNamespace(),
    )

    from clawloop.weight_backends.tinker import (
        TinkerWeightsBackend,
        TinkerWeightsConfig,
    )

    cfg = TinkerWeightsConfig(base_model="Qwen/Qwen3-8B", renderer_name="custom-renderer")
    TinkerWeightsBackend(cfg)

    assert recommended_calls == []
    assert get_renderer_calls == ["custom-renderer"]


# ---------------------------------------------------------------------------
# TinkerWeightsBackend (Task 9): Layer-protocol methods
# ---------------------------------------------------------------------------


def _fake_backend(monkeypatch):
    """Build a TinkerWeightsBackend with every SDK call mocked.

    Returns ``(backend, fake_training, fake_sampling)``. All Tinker SDK
    entry points and the renderer/tokenizer factories are monkeypatched
    so no network or model files are touched.
    """
    monkeypatch.setenv("TINKER_API_KEY", "fake")
    # Don't let load_env clobber our env var.
    monkeypatch.setattr("clawloop.weight_backends.tinker.load_env", lambda: [])

    fake_service = SimpleNamespace(name="service")
    fake_tokenizer = SimpleNamespace(name="tokenizer")
    fake_training = SimpleNamespace(get_tokenizer=lambda: fake_tokenizer)
    fake_sampling = SimpleNamespace(name="sampling-base")
    fake_renderer = SimpleNamespace(name="renderer")

    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.make_service_client",
        lambda: fake_service,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.create_training",
        lambda service, **kw: fake_training,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.create_sampling",
        lambda service, **kw: fake_sampling,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.make_rest_client",
        lambda service: SimpleNamespace(name="rest"),
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.get_model_id",
        lambda training: "fake-model-id",
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker.get_renderer",
        lambda name, tokenizer: fake_renderer,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker.get_recommended_renderer_name",
        lambda model: "qwen3",
    )

    from clawloop.weight_backends.tinker import (
        TinkerWeightsBackend,
        TinkerWeightsConfig,
    )

    cfg = TinkerWeightsConfig(base_model="Qwen/Qwen3-8B")
    backend = TinkerWeightsBackend(cfg)
    return backend, fake_training, fake_sampling


# 1. forward_backward pipelines fb + optim, returns ok with n_datums
def test_forward_backward_pipelines_fb_and_optim(monkeypatch):
    backend, fake_training, _ = _fake_backend(monkeypatch)

    monkeypatch.setattr(
        "clawloop.weight_backends.tinker.episodes_to_tinker_datums",
        lambda episodes, *, loss_fn: [MagicMock(name="datum")],
    )

    fb_future = MagicMock()
    fb_future.result.return_value = {"loss": 0.5}
    opt_future = MagicMock()
    opt_future.result.return_value = {"step": 1}

    fb_calls: list[dict] = []

    def _fb(training, batch, *, loss_fn, loss_fn_config):
        assert training is fake_training
        fb_calls.append(
            {
                "batch_len": len(batch),
                "loss_fn": loss_fn,
                "loss_fn_config": loss_fn_config,
            }
        )
        return fb_future

    opt_calls: list[Any] = []

    def _opt(training, adam_params):
        assert training is fake_training
        opt_calls.append(adam_params)
        return opt_future

    monkeypatch.setattr("clawloop.weight_backends.tinker._tinker_sdk.forward_backward", _fb)
    monkeypatch.setattr("clawloop.weight_backends.tinker._tinker_sdk.optim_step", _opt)

    result = backend.forward_backward(Datum(episodes=[], loss_fn="importance_sampling")).result()

    assert result.status == "ok"
    assert result.metrics["n_datums"] == 1
    assert len(fb_calls) == 1
    assert fb_calls[0]["batch_len"] == 1
    assert len(opt_calls) == 1
    fb_future.result.assert_called_once()
    opt_future.result.assert_called_once()


# 2. forward_backward returns ok with no SDK call when exporter yields []
def test_forward_backward_skips_when_no_datums(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)

    monkeypatch.setattr(
        "clawloop.weight_backends.tinker.episodes_to_tinker_datums",
        lambda episodes, *, loss_fn: [],
    )

    fb_called = []
    opt_called = []
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.forward_backward",
        lambda *a, **kw: fb_called.append(1) or MagicMock(),
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.optim_step",
        lambda *a, **kw: opt_called.append(1) or MagicMock(),
    )

    result = backend.forward_backward(Datum(episodes=[])).result()

    assert result.status == "ok"
    assert result.metrics.get("n_datums") == 0
    assert result.metrics.get("dropped") is True
    assert fb_called == []
    assert opt_called == []


# 3. forward_backward catches TinkerBackendError and returns FBResult(error)
def test_forward_backward_wraps_backend_error(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)

    monkeypatch.setattr(
        "clawloop.weight_backends.tinker.episodes_to_tinker_datums",
        lambda episodes, *, loss_fn: [MagicMock(name="datum")],
    )

    err = TinkerBackendError(BackendError(code="rate_limit", message="slow", recoverable=True))

    def _raise(*a, **kw):
        raise err

    monkeypatch.setattr("clawloop.weight_backends.tinker._tinker_sdk.forward_backward", _raise)
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.optim_step",
        lambda *a, **kw: MagicMock(),
    )

    result = backend.forward_backward(Datum(episodes=[])).result()

    assert result.status == "error"
    assert "error" in result.metrics
    assert result.metrics["error"]["code"] == "rate_limit"
    assert result.metrics["error"]["recoverable"] is True


# 4. save_state swaps the SamplingClient and appends to _adapter_paths
def test_save_state_swaps_sampling_client_and_appends_path(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)

    new_client = SimpleNamespace(name="sampling-after-save")
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.save_weights_and_get_sampling_client",
        lambda training, name: new_client,
    )
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.save_state_durable",
        lambda training, name, ttl_seconds=None: f"tinker://ckpt/{name}",
    )

    out = backend.save_state("iter_0").result()

    assert out.status == "ok"
    assert out.name == "iter_0"
    assert backend.current_sampling_client() is new_client
    assert backend._adapter_paths == ["iter_0"]
    assert backend._durable_paths == ["tinker://ckpt/iter_0"]


# 5. load_state with no adapter_paths is a successful no-op
def test_load_state_with_no_adapter_paths_is_ok(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)

    load_calls: list[Any] = []
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.load_state_with_optimizer",
        lambda training, path: load_calls.append(path),
    )

    out = backend.load_state({}).result()
    assert out.status == "ok"
    assert load_calls == []


# 6. load_state restores from the LAST entry of adapter_paths
def test_load_state_restores_from_last_path(monkeypatch):
    backend, fake_training, _ = _fake_backend(monkeypatch)

    load_calls: list[tuple[Any, str]] = []
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.load_state_with_optimizer",
        lambda training, path: load_calls.append((training, path)),
    )

    new_client = SimpleNamespace(name="sampling-after-load")
    create_calls: list[dict] = []

    def _create_sampling(service, **kwargs):
        create_calls.append(kwargs)
        return new_client

    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.create_sampling",
        _create_sampling,
    )

    # load_state prefers durable_paths (tinker:// URIs) over ephemeral names.
    out = backend.load_state(
        {
            "adapter_paths": ["iter_0", "iter_1"],
            "durable_paths": ["tinker://ckpt-0", "tinker://ckpt-1"],
        }
    ).result()

    assert out.status == "ok"
    assert load_calls == [(fake_training, "tinker://ckpt-1")]
    assert len(create_calls) == 1
    assert create_calls[0].get("model_path") == "tinker://ckpt-1"
    assert backend._adapter_paths == ["iter_0", "iter_1"]
    assert backend._durable_paths == ["tinker://ckpt-0", "tinker://ckpt-1"]
    assert backend.current_sampling_client() is new_client


# 7. optim_step is a no-op (already submitted+awaited inside forward_backward)
def test_optim_step_is_noop(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)

    sdk_called = []
    monkeypatch.setattr(
        "clawloop.weight_backends.tinker._tinker_sdk.optim_step",
        lambda *a, **kw: sdk_called.append(1) or MagicMock(),
    )

    out = backend.optim_step().result()
    assert out.status == "ok"
    assert out.updates_applied == 1
    assert sdk_called == []


# 8. sample returns base_model when no adapter has been saved
def test_sample_returns_base_model_when_no_adapter(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)

    out = backend.sample(SampleContext(bench="test")).result()
    assert out.output == "Qwen/Qwen3-8B"


# 9. sample returns the LATEST adapter path when one is available
def test_sample_returns_latest_adapter_path_when_available(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)
    backend._adapter_paths = ["iter_0", "iter_1"]

    out = backend.sample(SampleContext(bench="test")).result()
    assert out.output == "iter_1"


# 10. to_dict has no secret keys (or values)
def test_to_dict_has_no_secret_keys(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)

    d = backend.to_dict()
    blob = str(d).lower()
    for forbidden in ("api_key", "secret", "bearer", "token"):
        assert forbidden not in blob, f"{forbidden} leaked into to_dict()"
        assert all(
            forbidden not in str(k).lower() for k in d.keys()
        ), f"{forbidden} appeared as a key in to_dict()"


# 11. to_dict contains the expected config + adapter_paths
def test_to_dict_contains_expected_config_and_adapter_paths(monkeypatch):
    backend, _, _ = _fake_backend(monkeypatch)
    backend._adapter_paths = ["iter_0"]

    d = backend.to_dict()
    assert d["base_model"] == "Qwen/Qwen3-8B"
    assert d["lora_rank"] == 8
    assert d["adapter_paths"] == ["iter_0"]
    # All required keys are present.
    for key in (
        "base_model",
        "lora_rank",
        "seed",
        "train_attn",
        "train_mlp",
        "train_unembed",
        "loss_fn",
        "loss_fn_config",
        "adam_params",
        "adapter_paths",
    ):
        assert key in d, f"missing key {key!r} in to_dict()"
