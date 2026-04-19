"""Thin adapter over the Tinker SDK.

Every Tinker SDK call lives behind a typed function in this module so SDK
drift is isolated to a single file.  Tinker exceptions are translated
(by exception-class name, not isinstance — so we survive SDK version
shuffles) into :class:`TinkerBackendError`, which wraps the ClawLoop
:class:`BackendError` dataclass with the correct ``recoverable`` bit.

Signatures verified against ``tinker==0.18.1`` (preflight Task 2).

Notes on design
---------------
- ``BackendError`` in ``clawloop.weight_backends.base`` is a frozen dataclass
  (not an ``Exception``); it cannot be raised.  We therefore expose
  ``TinkerBackendError(Exception)`` which carries a ``BackendError`` on
  ``.error`` and mirrors ``recoverable``/``code``/``message`` as properties
  for ergonomic use at call sites.

- Futures (``ConcurrentFuture``) returned by ``async_sample`` and
  ``forward_backward`` are passed through unchanged so callers can submit
  multiple operations before awaiting any of them.
"""

from __future__ import annotations

from typing import Any

import tinker
from tinker import types as _tinker_types

from clawloop.weight_backends.base import BackendError


# ---------------------------------------------------------------------------
# Exception wrapper
# ---------------------------------------------------------------------------

class TinkerBackendError(Exception):
    """Raisable wrapper around a :class:`BackendError` descriptor.

    Exposes the underlying dataclass on ``.error`` and mirrors its
    ``recoverable``/``code``/``message`` fields as properties so callers can
    write ``except TinkerBackendError as e: if e.recoverable: ...`` without
    reaching into ``.error`` every time.
    """

    def __init__(self, error: BackendError) -> None:
        self.error = error
        super().__init__(f"[{error.code}] {error.message}")

    @property
    def recoverable(self) -> bool:
        return self.error.recoverable

    @property
    def code(self) -> str:
        return self.error.code

    @property
    def message(self) -> str:
        return self.error.message


# ---------------------------------------------------------------------------
# Error taxonomy
# ---------------------------------------------------------------------------

# Single source of truth: ``exc_name -> (code, recoverable)``. Matching is on
# ``type(exc).__name__`` (string) rather than ``isinstance`` so we survive SDK
# version shuffles where the class might live at a different import path but
# keep the same name. Unknown names fall through to ``("unknown", False)``
# (conservative — prefer fail-loud over silent retry on a genuinely broken call).
#
# Edit this table when the SDK adds new exception types.
_ERROR_TAXONOMY: dict[str, tuple[str, bool]] = {
    # Recoverable
    "RateLimitError":             ("backend_unreachable", True),
    "APIConnectionError":         ("backend_unreachable", True),
    "APITimeoutError":            ("backend_unreachable", True),
    "InternalServerError":        ("backend_unreachable", True),
    "RequestFailedError":         ("backend_unreachable", True),
    # Non-recoverable
    "BadRequestError":            ("invalid_config", False),
    "AuthenticationError":        ("invalid_config", False),
    "PermissionDeniedError":      ("invalid_config", False),
    "UnprocessableEntityError":   ("invalid_config", False),
    "ConflictError":              ("invalid_config", False),
    "APIResponseValidationError": ("schema_incompatible", False),
}


def _wrap(exc: Exception) -> TinkerBackendError:
    """Translate a raw Tinker exception into a :class:`TinkerBackendError`."""
    name = type(exc).__name__
    code, recoverable = _ERROR_TAXONOMY.get(name, ("unknown", False))
    return TinkerBackendError(
        BackendError(code=code, message=str(exc), recoverable=recoverable)
    )


# ---------------------------------------------------------------------------
# Thin adapter functions
# ---------------------------------------------------------------------------

def make_service_client() -> "tinker.ServiceClient":
    """Return a new Tinker :class:`ServiceClient`.

    Reads ``TINKER_API_KEY`` from the environment; no kwargs are passed.
    """
    try:
        return tinker.ServiceClient()
    except Exception as e:
        raise _wrap(e) from e


def create_training(
    service: "tinker.ServiceClient",
    *,
    base_model: str,
    rank: int,
    seed: int,
    train_attn: bool,
    train_mlp: bool,
    train_unembed: bool,
) -> Any:
    """Create a LoRA training client on *service*.

    All six kwargs are forwarded to ``service.create_lora_training_client``.
    There is NO ``alpha`` kwarg in tinker 0.18.1.
    """
    try:
        return service.create_lora_training_client(
            base_model=base_model,
            rank=rank,
            seed=seed,
            train_attn=train_attn,
            train_mlp=train_mlp,
            train_unembed=train_unembed,
        )
    except Exception as e:
        raise _wrap(e) from e


def create_sampling(
    service: "tinker.ServiceClient",
    *,
    base_model: str | None = None,
    model_path: str | None = None,
    retry_config: Any = None,
) -> Any:
    """Create a sampling client.

    Exactly one of ``base_model`` or ``model_path`` must be supplied.
    Passing both or neither raises :class:`ValueError`.
    """
    if (base_model is None) == (model_path is None):
        raise ValueError(
            "exactly one of base_model or model_path is required"
        )
    kwargs: dict[str, Any] = {"retry_config": retry_config}
    if base_model is not None:
        kwargs["base_model"] = base_model
    else:
        kwargs["model_path"] = model_path
    try:
        return service.create_sampling_client(**kwargs)
    except Exception as e:
        raise _wrap(e) from e


def async_sample(
    sampling_client: Any,
    *,
    prompt_tokens: list[int],
    num_samples: int = 1,
    max_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.95,
    stop: list[int] | None = None,
) -> Any:
    """Submit a sampling request and return the ``ConcurrentFuture`` unchanged.

    The caller can submit several sampling ops before awaiting any of them.
    """
    try:
        prompt = _tinker_types.ModelInput(
            chunks=[_tinker_types.EncodedTextChunk(tokens=prompt_tokens)]
        )
        params = _tinker_types.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        return sampling_client.sample(
            prompt=prompt,
            num_samples=num_samples,
            sampling_params=params,
        )
    except Exception as e:
        raise _wrap(e) from e


def forward_backward(
    training: Any,
    batch: list,
    *,
    loss_fn: str,
    loss_fn_config: dict | None = None,
) -> Any:
    """Submit a forward_backward op; return the future so callers can pipeline."""
    try:
        return training.forward_backward(
            data=batch,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
        )
    except Exception as e:
        raise _wrap(e) from e


def optim_step(training: Any, adam_params: "_tinker_types.AdamParams") -> Any:
    """Apply an optimizer step with a typed :class:`AdamParams` object.

    Tinker 0.18.1 takes the typed object positionally — **not** a dict,
    **not** loose kwargs.
    """
    try:
        return training.optim_step(adam_params)
    except Exception as e:
        raise _wrap(e) from e


def save_weights_and_get_sampling_client(
    training: Any,
    name: str,
    *,
    retry_config: Any = None,
) -> Any:
    """Persist a checkpoint and return the fresh :class:`SamplingClient`.

    The SDK returns the client directly — no tuple, no ttl.
    """
    try:
        return training.save_weights_and_get_sampling_client(
            name, retry_config=retry_config
        )
    except Exception as e:
        raise _wrap(e) from e


def save_state_durable(
    training: Any, name: str, *, ttl_seconds: int | None = None,
) -> str | None:
    """Write a durable training checkpoint; return its ``tinker://`` path.

    Unlike :func:`save_weights_and_get_sampling_client` (ephemeral — its ``name``
    is ignored, no enumerable path), ``save_state`` produces a checkpoint that
    :func:`list_checkpoints` can find and that ``create_training_client_from_state``
    can re-attach to.  We call both per iteration: one for the fresh
    SamplingClient next iter will use, one for audit/resume.
    """
    try:
        fut = training.save_state(name, ttl_seconds=ttl_seconds)
        resp = fut.result()
        return getattr(resp, "path", None) or getattr(resp, "tinker_path", None)
    except Exception as e:
        raise _wrap(e) from e


def get_model_id(training: Any) -> str:
    """Training-run identifier used by :func:`list_checkpoints`."""
    try:
        return training.model_id
    except Exception as e:
        raise _wrap(e) from e


def make_rest_client(service: Any) -> Any:
    try:
        return service.create_rest_client()
    except Exception as e:
        raise _wrap(e) from e


def list_checkpoints(rest: Any, training_run_id: str) -> list[dict[str, Any]]:
    """Return a JSON-serializable snapshot of the run's current checkpoints.

    Each entry is ``{"checkpoint_id", "checkpoint_type", "time", "tinker_path",
    "size_bytes", "expires_at", "public"}``.  Best-effort — never raises the
    run; on SDK failure returns ``[{"error": ...}]`` so the caller can still log.
    """
    try:
        fut = rest.list_checkpoints(training_run_id)
        resp = fut.result()
    except Exception as e:
        return [{"error": type(e).__name__, "message": str(e)}]

    def _coerce(v: Any) -> Any:
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        # datetime.datetime -> ISO 8601; pydantic enums / objects -> str.
        return str(v)

    out: list[dict[str, Any]] = []
    for ck in getattr(resp, "checkpoints", []) or []:
        out.append({
            "checkpoint_id":   _coerce(getattr(ck, "checkpoint_id", None)),
            "checkpoint_type": _coerce(getattr(ck, "checkpoint_type", None)),
            "time":            _coerce(getattr(ck, "time", None)),
            "tinker_path":     _coerce(getattr(ck, "tinker_path", None)),
            "size_bytes":      getattr(ck, "size_bytes", None),
            "expires_at":      _coerce(getattr(ck, "expires_at", None)),
            "public":          getattr(ck, "public", None),
        })
    return out


def load_state_with_optimizer(training: Any, path: str) -> Any:
    """Restore weights + optimizer state from ``path`` (``tinker://...`` URI).

    Returns whatever the SDK returns (likely an ``APIFuture``; await before
    use). Use this — not the weights-only ``load_state`` — when you need
    optimizer momentum/variance restored on resume.
    """
    try:
        fut = training.load_state_with_optimizer(path)
        return fut.result() if hasattr(fut, "result") else fut
    except Exception as e:
        raise _wrap(e) from e
