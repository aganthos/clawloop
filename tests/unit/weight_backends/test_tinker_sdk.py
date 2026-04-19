"""Unit tests for the Tinker SDK thin adapter.

Every Tinker SDK call is mocked — no network access is required.  Tests cover
signature correctness, error-taxonomy wrapping, and structural checks on the
``ModelInput`` / ``SamplingParams`` objects built by ``async_sample``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# tinker is an optional extra (see pyproject.toml [games]); skip the whole
# module in CI runs that don't install it.
tinker = pytest.importorskip("tinker")
tinker_types = pytest.importorskip("tinker.types")

from clawloop.weight_backends import _tinker_sdk
from clawloop.weight_backends._tinker_sdk import (
    TinkerBackendError,
    async_sample,
    create_sampling,
    create_training,
    forward_backward,
    load_state_with_optimizer,
    make_service_client,
    optim_step,
    save_weights_and_get_sampling_client,
)


# ---------------------------------------------------------------------------
# 1. make_service_client reads env
# ---------------------------------------------------------------------------

def test_make_service_client_reads_env(monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "test-key-123")
    fake_client = MagicMock()
    with patch.object(tinker, "ServiceClient", return_value=fake_client) as ctor:
        got = make_service_client()
    assert got is fake_client
    ctor.assert_called_once_with()


# ---------------------------------------------------------------------------
# 2. create_training forwards kwargs
# ---------------------------------------------------------------------------

def test_create_training_passes_kwargs():
    service = MagicMock()
    create_training(
        service,
        base_model="meta-llama/Llama-3.2-1B",
        rank=8,
        seed=42,
        train_attn=True,
        train_mlp=False,
        train_unembed=True,
    )
    service.create_lora_training_client.assert_called_once_with(
        base_model="meta-llama/Llama-3.2-1B",
        rank=8,
        seed=42,
        train_attn=True,
        train_mlp=False,
        train_unembed=True,
    )


# ---------------------------------------------------------------------------
# 3. create_sampling: exactly-one-of validation
# ---------------------------------------------------------------------------

def test_create_sampling_requires_exactly_one_ref():
    service = MagicMock()

    with pytest.raises(ValueError):
        create_sampling(service)  # neither

    with pytest.raises(ValueError):
        create_sampling(service, base_model="x", model_path="y")  # both


# ---------------------------------------------------------------------------
# 4. create_sampling forwards base_model + retry_config
# ---------------------------------------------------------------------------

def test_create_sampling_base_model_path():
    service = MagicMock()
    retry = object()
    create_sampling(service, base_model="meta-llama/Llama-3.2-1B", retry_config=retry)
    service.create_sampling_client.assert_called_once_with(
        base_model="meta-llama/Llama-3.2-1B",
        retry_config=retry,
    )


# ---------------------------------------------------------------------------
# 5. optim_step passes typed AdamParams positionally
# ---------------------------------------------------------------------------

def test_optim_step_passes_typed_adam_params():
    training = MagicMock()
    adam = tinker_types.AdamParams(learning_rate=1e-5)
    optim_step(training, adam)
    training.optim_step.assert_called_once_with(adam)
    # sanity: no kwargs used
    _, kwargs = training.optim_step.call_args
    assert kwargs == {}


# ---------------------------------------------------------------------------
# 6. save_weights_and_get_sampling_client returns result directly
# ---------------------------------------------------------------------------

def test_save_weights_returns_sampling_client_directly():
    training = MagicMock()
    sentinel = MagicMock(name="SamplingClient")
    training.save_weights_and_get_sampling_client.return_value = sentinel

    out = save_weights_and_get_sampling_client(training, "ckpt-0", retry_config=None)

    assert out is sentinel
    training.save_weights_and_get_sampling_client.assert_called_once_with(
        "ckpt-0", retry_config=None
    )


# ---------------------------------------------------------------------------
# Helper: build a fake exception class by name (for error-taxonomy tests)
# ---------------------------------------------------------------------------

def _make_exc(name: str) -> type[Exception]:
    return type(name, (Exception,), {})


# ---------------------------------------------------------------------------
# 7. forward_backward wraps RateLimitError as recoverable
# ---------------------------------------------------------------------------

def test_forward_backward_wraps_rate_limit_as_recoverable():
    RateLimitError = _make_exc("RateLimitError")
    training = MagicMock()
    training.forward_backward.side_effect = RateLimitError("slow down")

    with pytest.raises(TinkerBackendError) as exc_info:
        forward_backward(training, [], loss_fn="cross_entropy")

    assert exc_info.value.recoverable is True
    assert exc_info.value.error.recoverable is True


# ---------------------------------------------------------------------------
# 8. forward_backward wraps BadRequestError as non-recoverable
# ---------------------------------------------------------------------------

def test_forward_backward_wraps_badrequest_as_non_recoverable():
    BadRequestError = _make_exc("BadRequestError")
    training = MagicMock()
    training.forward_backward.side_effect = BadRequestError("bad input")

    with pytest.raises(TinkerBackendError) as exc_info:
        forward_backward(training, [], loss_fn="cross_entropy")

    assert exc_info.value.recoverable is False
    assert exc_info.value.error.recoverable is False


# ---------------------------------------------------------------------------
# 9. Error-taxonomy .code assertions + unknown-exception fallthrough
# ---------------------------------------------------------------------------

def test_forward_backward_wraps_rate_limit_has_backend_unreachable_code():
    training = MagicMock()
    exc_cls = type("RateLimitError", (Exception,), {})
    training.forward_backward.side_effect = exc_cls("429")
    with pytest.raises(TinkerBackendError) as exc_info:
        forward_backward(training, [], loss_fn="importance_sampling")
    assert exc_info.value.code == "backend_unreachable"
    assert exc_info.value.recoverable is True


def test_forward_backward_wraps_badrequest_has_invalid_config_code():
    training = MagicMock()
    exc_cls = type("BadRequestError", (Exception,), {})
    training.forward_backward.side_effect = exc_cls("bad")
    with pytest.raises(TinkerBackendError) as exc_info:
        forward_backward(training, [], loss_fn="importance_sampling")
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.recoverable is False


def test_forward_backward_unknown_exception_maps_to_unknown_non_recoverable():
    training = MagicMock()
    exc_cls = type("WeirdFutureSDKError", (Exception,), {})
    training.forward_backward.side_effect = exc_cls("surprise")
    with pytest.raises(TinkerBackendError) as exc_info:
        forward_backward(training, [], loss_fn="importance_sampling")
    assert exc_info.value.code == "unknown"
    assert exc_info.value.recoverable is False


# ---------------------------------------------------------------------------
# 10. async_sample builds ModelInput + SamplingParams correctly
# ---------------------------------------------------------------------------

def test_async_sample_builds_model_input_and_sampling_params():
    sampling_client = MagicMock()
    sampling_client.sample.return_value = MagicMock(name="ConcurrentFuture")

    out = async_sample(
        sampling_client,
        prompt_tokens=[1, 2, 3],
        num_samples=1,
        max_tokens=8,
        temperature=0.7,
        top_p=0.95,
        stop=None,
    )

    assert out is sampling_client.sample.return_value
    sampling_client.sample.assert_called_once()
    _, kwargs = sampling_client.sample.call_args

    # prompt is a ModelInput with chunks=[EncodedTextChunk(tokens=[1,2,3])]
    prompt = kwargs["prompt"]
    assert isinstance(prompt, tinker_types.ModelInput)
    assert len(prompt.chunks) == 1
    assert list(prompt.chunks[0].tokens) == [1, 2, 3]

    assert kwargs["num_samples"] == 1

    params = kwargs["sampling_params"]
    assert isinstance(params, tinker_types.SamplingParams)
    assert params.max_tokens == 8


# ---------------------------------------------------------------------------
# 11. load_state_with_optimizer forwards the path and unwraps APIFuture
# ---------------------------------------------------------------------------

def test_load_state_with_optimizer_forwards_path():
    training = MagicMock()
    fut = MagicMock()
    fut.result.return_value = "restored"
    training.load_state_with_optimizer.return_value = fut

    out = load_state_with_optimizer(training, "tinker://ckpt-7")

    training.load_state_with_optimizer.assert_called_once_with("tinker://ckpt-7")
    fut.result.assert_called_once_with()
    assert out == "restored"


def test_load_state_with_optimizer_passthrough_when_no_result_attr():
    training = MagicMock()
    sentinel = object()
    # Use SimpleNamespace-like object without .result attr.
    training.load_state_with_optimizer.return_value = sentinel

    # Patch hasattr behaviour by giving back something without result().
    class _Bare:
        pass

    bare = _Bare()
    training.load_state_with_optimizer.return_value = bare
    out = load_state_with_optimizer(training, "tinker://ckpt-1")
    assert out is bare


# ---------------------------------------------------------------------------
# 12. load_state_with_optimizer wraps exceptions via the error taxonomy
# ---------------------------------------------------------------------------

def test_load_state_with_optimizer_wraps_recoverable_exception():
    training = MagicMock()
    exc_cls = type("RateLimitError", (Exception,), {})
    training.load_state_with_optimizer.side_effect = exc_cls("slow down")

    with pytest.raises(TinkerBackendError) as exc_info:
        load_state_with_optimizer(training, "tinker://ckpt-0")
    assert exc_info.value.code == "backend_unreachable"
    assert exc_info.value.recoverable is True


def test_load_state_with_optimizer_wraps_non_recoverable_exception():
    training = MagicMock()
    exc_cls = type("BadRequestError", (Exception,), {})
    training.load_state_with_optimizer.side_effect = exc_cls("bad path")

    with pytest.raises(TinkerBackendError) as exc_info:
        load_state_with_optimizer(training, "tinker://ckpt-0")
    assert exc_info.value.code == "invalid_config"
    assert exc_info.value.recoverable is False
