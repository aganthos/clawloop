"""Tinker weights backend configuration and backend class.

Holds the non-secret configuration for the Tinker weight backend. The
``TINKER_API_KEY`` is read from the environment at backend construction time
and is intentionally never stored on this dataclass — keeping secrets out of
serialized configs, logs, and archive records.

Field choices follow the v5.1 SDK preflight: Tinker 0.18.1 has no ``alpha``
LoRA kwarg, ``AdamParams`` exposes only ``learning_rate, beta1, beta2, eps``
(no ``weight_decay``), and ``save_weights_and_get_sampling_client`` does not
accept a ``ttl_seconds`` argument. See the design doc v5.1 SDK overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from tinker import types as _tinker_types
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer

from clawloop.config import load_env
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
from clawloop.weight_backends import _tinker_sdk
from clawloop.weight_backends._tinker_exporter import episodes_to_tinker_datums
from clawloop.weight_backends._tinker_sdk import TinkerBackendError
from clawloop.weight_backends.base import BackendError

LossFn = Literal["importance_sampling", "cross_entropy", "ppo", "cispo", "dro"]


@dataclass
class TinkerWeightsConfig:
    """Non-secret configuration. TINKER_API_KEY is read from env, never stored."""

    base_model: str
    lora_rank: int = 8
    seed: int = 42
    train_attn: bool = True
    train_mlp: bool = True
    train_unembed: bool = False
    renderer_name: str | None = None
    loss_fn: LossFn = "importance_sampling"
    loss_fn_config: dict[str, Any] = field(default_factory=dict)
    # Users can pass a partial dict (e.g. just ``learning_rate``). We merge
    # with defaults at construction time in ``__post_init__`` so a partial
    # override doesn't drop required AdamParams kwargs.
    adam_params: dict[str, Any] = field(default_factory=dict)
    # NOTE: Tinker 0.18.1 AdamParams has NO weight_decay field — do not include.
    # TTL applied to durable training.save_state checkpoints written each iter.
    # 1 hour by default; set None to retain (subject to account quotas).
    ttl_seconds_intermediate: int | None = 3600

    # Defaults that must appear on every AdamParams call; a partial user
    # override is merged over these in ``__post_init__``.
    _ADAM_DEFAULTS: "dict[str, Any]" = field(
        default_factory=lambda: {
            "learning_rate": 1e-5,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
        },
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        # Merge partial adam_params over defaults so users can write
        # ``adam_params={"learning_rate": 5e-6}`` without dropping the rest.
        merged = {**self._ADAM_DEFAULTS, **self.adam_params}
        # Validate: every required kwarg present after merge.
        required = {"learning_rate", "beta1", "beta2", "eps"}
        missing = required - set(merged)
        if missing:
            raise ValueError(f"adam_params missing required keys after merge: {sorted(missing)}")
        self.adam_params = merged


# ---------------------------------------------------------------------------
# TinkerWeightsBackend (Task 8: __init__ + current_sampling_client)
# ---------------------------------------------------------------------------


class TinkerWeightsBackend:
    """Tinker weight backend.

    Task 8 covers only the constructor and the accessor methods. The
    Layer-protocol methods (forward_backward, optim_step, save_state,
    load_state, sample, to_dict, clear_pending_state) arrive in Task 9.

    The base-model SamplingClient is created up-front so iter-0 rollouts
    have a valid client BEFORE any save_state has been called. After the
    first ``save_state``, ``current_sampling_client()`` will return the
    adapter-bound client instead.
    """

    def __init__(self, config: TinkerWeightsConfig) -> None:
        # 1. Load .env so TINKER_API_KEY can live in clawloop/.env.
        load_env()

        # 2. Fail fast if TINKER_API_KEY is still not in env.
        if "TINKER_API_KEY" not in os.environ:
            raise TinkerBackendError(
                BackendError(
                    code="missing_api_key",
                    message=(
                        "TINKER_API_KEY env var must be set (put it in "
                        "clawloop/.env or export it) before constructing "
                        "TinkerWeightsBackend."
                    ),
                    recoverable=False,
                )
            )

        self._config = config
        self._service = _tinker_sdk.make_service_client()

        # 3. Training client (LoRA).
        self._training = _tinker_sdk.create_training(
            self._service,
            base_model=config.base_model,
            rank=config.lora_rank,
            seed=config.seed,
            train_attn=config.train_attn,
            train_mlp=config.train_mlp,
            train_unembed=config.train_unembed,
        )

        # 4. Tokenizer = single source of truth from the training client.
        # We do NOT use HF AutoTokenizer anywhere.
        self._tokenizer = self._training.get_tokenizer()

        # 5. Renderer via tinker_cookbook — auto-select per model unless
        # the user pinned one explicitly.
        renderer_name = config.renderer_name or get_recommended_renderer_name(config.base_model)
        self._renderer = get_renderer(renderer_name, self._tokenizer)

        # 6. Base-model SamplingClient — so iter 0 rollouts have a valid
        # client BEFORE any save_state has been called.
        self._sampling = _tinker_sdk.create_sampling(
            self._service,
            base_model=config.base_model,
        )
        self._adapter_paths: list[str] = []
        # Durable tinker:// paths from training.save_state — enumerable via
        # RestClient.list_checkpoints. Populated alongside adapter_paths by
        # save_state() below.
        self._durable_paths: list[str] = []
        self._rest = _tinker_sdk.make_rest_client(self._service)
        self._model_id = _tinker_sdk.get_model_id(self._training)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        """Training-run identifier (used for RestClient.list_checkpoints)."""
        return self._model_id

    def list_tinker_checkpoints(self) -> list[dict[str, Any]]:
        """Enumerate this run's durable checkpoints.

        Ephemeral save_weights_and_get_sampling_client checkpoints are NOT
        included — those are not listed by the Tinker REST API. Only the
        ``save_state``-produced checkpoints we wrote explicitly show up.
        """
        return _tinker_sdk.list_checkpoints(self._rest, self._model_id)

    def current_sampling_client(self) -> Any:
        """Latest :class:`SamplingClient`.

        Base-model client at iter 0; adapter-bound after the first
        successful ``save_state`` (added in Task 9).
        """
        return self._sampling

    @property
    def renderer(self) -> Any:
        return self._renderer

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def config(self) -> TinkerWeightsConfig:
        return self._config

    # ------------------------------------------------------------------
    # Layer protocol — forward_backward / optim_step
    # ------------------------------------------------------------------

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Run one fwd/bwd + optim_step on *data.episodes*.

        Pipelining: we submit ``forward_backward`` and ``optim_step``
        back-to-back BEFORE awaiting either. The Tinker SDK guarantees
        ordering by submission, so the optimizer will see this step's
        gradients. Awaiting both before returning keeps the public verb
        simple — ``optim_step`` is a no-op so callers can still use the
        two-phase Layer contract.
        """
        try:
            tinker_datums = episodes_to_tinker_datums(data.episodes, loss_fn=self._config.loss_fn)
            if not tinker_datums:
                return Future.immediate(
                    FBResult(
                        status="ok",
                        metrics={"n_datums": 0, "dropped": True},
                    )
                )

            fb_future = _tinker_sdk.forward_backward(
                self._training,
                tinker_datums,
                loss_fn=self._config.loss_fn,
                loss_fn_config=self._config.loss_fn_config,
            )
            adam_params = _tinker_types.AdamParams(**self._config.adam_params)
            opt_future = _tinker_sdk.optim_step(self._training, adam_params)

            # Await both — order: fb first to surface fb errors first.
            fb_out = fb_future.result() if hasattr(fb_future, "result") else fb_future
            opt_out = opt_future.result() if hasattr(opt_future, "result") else opt_future

            # Extract JSON-safe scalar metrics. The SDK's ForwardBackwardOutput
            # / OptimStepResponse pack their stats in a `.metrics: dict[str, Any]`;
            # reach into that rather than serializing the whole pydantic object.
            fb_metrics = getattr(fb_out, "metrics", {}) or {}
            opt_metrics = getattr(opt_out, "metrics", {}) or {}

            metrics: dict[str, Any] = {
                "n_datums": len(tinker_datums),
                **{f"fb.{k}": v for k, v in fb_metrics.items()},
                **{f"optim.{k}": v for k, v in opt_metrics.items()},
            }
            return Future.immediate(FBResult(status="ok", metrics=metrics))
        except TinkerBackendError as e:
            return Future.immediate(
                FBResult(
                    status="error",
                    metrics={
                        "error": {
                            "code": e.code,
                            "message": e.message,
                            "recoverable": e.recoverable,
                        }
                    },
                )
            )

    def optim_step(self) -> Future[OptimResult]:
        """No-op — the optimizer step was already submitted+awaited inside
        :meth:`forward_backward`. Returns a successful result so the
        two-phase Layer contract is still satisfied for callers."""
        return Future.immediate(OptimResult(status="ok", updates_applied=1, metrics={}))

    # ------------------------------------------------------------------
    # Layer protocol — sample
    # ------------------------------------------------------------------

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:  # noqa: ARG002
        """Return the latest adapter path identifier (or base model name).

        The SamplingClient itself is accessed via
        :meth:`current_sampling_client`; this method's return value is a
        lightweight string identifier suitable for logging and state
        hashing.
        """
        if self._adapter_paths:
            output = self._adapter_paths[-1]
        else:
            output = self._config.base_model
        return Future.immediate(SampleResult(output=output, metadata={}))

    # ------------------------------------------------------------------
    # Layer protocol — save_state / load_state
    # ------------------------------------------------------------------

    def save_state(self, name: str) -> Future[SaveResult]:
        """Persist a checkpoint twice: ephemeral (→ fresh SamplingClient) +
        durable (→ enumerable ``tinker://`` path).

        Why both:
        - ``save_weights_and_get_sampling_client`` gives us the client the
          next iter's rollouts need, but is ephemeral and invisible to
          ``list_checkpoints``.
        - ``training.save_state(name)`` writes a durable, listable training
          checkpoint we can resume from with ``load_state_with_optimizer``.
        """
        try:
            new_sampling = _tinker_sdk.save_weights_and_get_sampling_client(self._training, name)
            self._sampling = new_sampling
            self._adapter_paths.append(name)
            # Best-effort durable save. Failure here must not abort training —
            # catch locally and surface via SaveResult.status with a hint.
            try:
                path = _tinker_sdk.save_state_durable(
                    self._training,
                    name,
                    ttl_seconds=self._config.ttl_seconds_intermediate,
                )
                if path:
                    self._durable_paths.append(path)
            except TinkerBackendError as e:
                return Future.immediate(
                    SaveResult(name=name, status=f"ok_ephemeral_only: {e.code}")
                )
            return Future.immediate(SaveResult(name=name, status="ok"))
        except TinkerBackendError as e:
            return Future.immediate(SaveResult(name=name, status=f"error: {e.code}"))

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        """Restore weights + optimizer from the last durable checkpoint.

        Prefers ``durable_paths`` (real ``tinker://`` URIs) over
        ``adapter_paths`` (ephemeral names). If neither is usable, no-op and
        keep the base-model sampling client from __init__.
        """
        if not state:
            return Future.immediate(LoadResult(status="ok"))
        durable = state.get("durable_paths") or []
        adapter = state.get("adapter_paths") or []
        # Pick the real path if we have one; ephemeral names aren't reloadable.
        last_path = durable[-1] if durable else (adapter[-1] if adapter else None)
        if not last_path or not str(last_path).startswith("tinker://"):
            return Future.immediate(LoadResult(status="ok"))
        try:
            _tinker_sdk.load_state_with_optimizer(self._training, last_path)
            self._sampling = _tinker_sdk.create_sampling(self._service, model_path=last_path)
            self._adapter_paths = list(adapter)
            self._durable_paths = list(durable)
            return Future.immediate(LoadResult(status="ok"))
        except TinkerBackendError as e:
            return Future.immediate(LoadResult(status=f"error: {e.code}"))

    # ------------------------------------------------------------------
    # Layer protocol — clear_pending_state / to_dict
    # ------------------------------------------------------------------

    def clear_pending_state(self) -> None:
        """No-op — Tinker manages its own pending-gradient buffers."""
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return a non-secret view of this backend.

        Excludes the API key (which is read from env, not stored on the
        config) and any other credential-like fields.
        """
        cfg = self._config
        return {
            "base_model": cfg.base_model,
            "lora_rank": cfg.lora_rank,
            "seed": cfg.seed,
            "train_attn": cfg.train_attn,
            "train_mlp": cfg.train_mlp,
            "train_unembed": cfg.train_unembed,
            "loss_fn": cfg.loss_fn,
            "loss_fn_config": dict(cfg.loss_fn_config),
            "adam_params": dict(cfg.adam_params),
            "adapter_paths": list(self._adapter_paths),
            "durable_paths": list(self._durable_paths),
            "model_id": self._model_id,
        }
