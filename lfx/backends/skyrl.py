"""SkyRLWeightsBackend — wraps SkyRL's AbstractBackend for LoRA/GRPO training.

All SkyRL config passes through as dicts — LfX does not interpret SkyRL's
configuration knobs.  The ``_to_forward_backward_input`` translation passes
raw rewards and trajectory_ids through; SkyRL computes advantages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lfx.backends.base import BackendError, SkyRLBackendInitError
from lfx.core.types import (
    Datum,
    FBResult,
    Future,
    LoadResult,
    OptimResult,
    SampleContext,
    SampleResult,
    SaveResult,
)
from lfx.exporters.skyrl import SkyRLExporter


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SkyRLWeightsConfig:
    """Configuration for the SkyRL weights backend.

    All SkyRL-specific knobs live inside ``backend_config``, ``lora_config``,
    and ``training_config`` as opaque dicts — LfX passes them through without
    interpretation.
    """

    base_model: str = ""
    backend_type: str = "jax"  # "jax" or "skyrl_train"
    backend_config: dict[str, Any] = field(default_factory=dict)
    lora_config: dict[str, Any] = field(default_factory=dict)
    training_config: dict[str, Any] = field(default_factory=dict)
    tokenizer_name: str = ""


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

_REQUIRED_STATE_KEYS = frozenset(
    ["model_ref", "backend_type", "backend_config", "lora_config", "training_config", "adapter_refs"]
)


class SkyRLWeightsBackend:
    """Wraps SkyRL's AbstractBackend for real LoRA/GRPO training.

    The ``__init__`` lifecycle requires real SkyRL + a tokenizer, so tests
    bypass it with ``__new__`` and mock the internals.
    """

    def __init__(self, config: SkyRLWeightsConfig) -> None:
        self._config = config
        self._adapter_refs: list[str] = []
        self.inference_url: str | None = None

        # 1. Validate SkyRL imports
        try:
            import skyrl  # noqa: F401
        except ImportError as e:
            raise SkyRLBackendInitError(BackendError.from_exception(e)) from e

        # 2. Load tokenizer and validate apply_chat_template
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                config.tokenizer_name or config.base_model
            )
            # Smoke-test the chat template
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "test"}],
                tokenize=True,
                add_generation_prompt=True,
            )
        except Exception as e:
            raise SkyRLBackendInitError(
                BackendError(
                    code="tokenizer_mismatch",
                    message=f"Tokenizer validation failed: {e}",
                    recoverable=False,
                )
            ) from e

        # 3. Instantiate SkyRL backend
        try:
            if config.backend_type == "jax":
                from skyrl.backends.jax_backend import JaxBackend

                self._backend = JaxBackend(**config.backend_config)
            elif config.backend_type == "skyrl_train":
                from skyrl.backends.skyrl_train_backend import SkyRLTrainBackend

                self._backend = SkyRLTrainBackend(**config.backend_config)
            else:
                raise ValueError(f"Unknown backend_type: {config.backend_type!r}")
        except Exception as e:
            if isinstance(e, SkyRLBackendInitError):
                raise
            raise SkyRLBackendInitError(BackendError.from_exception(e)) from e

        # 4. Create model with LoRA config
        self._model_id = f"lfx-{config.base_model}"
        try:
            self._backend.create_model(self._model_id, config.lora_config)
        except Exception as e:
            raise SkyRLBackendInitError(BackendError.from_exception(e)) from e

        # 5. Store inference_url if configured
        if config.backend_config.get("enable_http_endpoint"):
            self.inference_url = getattr(self._backend, "inference_url", None)

        # 6. Create exporter
        self._exporter = SkyRLExporter(tokenizer=tokenizer)

    # -- Layer protocol -----------------------------------------------------

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Export episodes → GeneratorOutput → backend.forward_backward."""
        try:
            gen_output = self._exporter.export(data.episodes)
            fb_input = self._to_forward_backward_input(gen_output)
            result = self._backend.forward_backward(fb_input)
            metrics = result if isinstance(result, dict) else {}
            return Future.immediate(FBResult(status="ok", metrics=metrics))
        except Exception as e:
            err = BackendError.from_exception(e)
            return Future.immediate(FBResult(status="error", metrics={"error": err}))

    def optim_step(self) -> Future[OptimResult]:
        """Build OptimStepInput from training_config and call backend."""
        try:
            # Pass full training_config so loss_fn, scheduler, etc. take effect
            optim_input = dict(self._config.training_config)
            result = self._backend.optim_step(self._model_id, optim_input)
            metrics = result if isinstance(result, dict) else {}
            return Future.immediate(
                OptimResult(status="ok", updates_applied=1, metrics=metrics)
            )
        except Exception as e:
            err = BackendError.from_exception(e)
            return Future.immediate(
                OptimResult(status="error", updates_applied=0, metrics={"error": err})
            )

    def sample(self, ctx: SampleContext) -> Future[SampleResult]:
        """Return the base model reference."""
        return Future.immediate(
            SampleResult(output=self._config.base_model)
        )

    def save_state(self, name: str) -> Future[SaveResult]:
        """Save a checkpoint and record the adapter reference."""
        try:
            self._backend.save_checkpoint(self._model_id, name)
            self._adapter_refs.append(name)
            return Future.immediate(SaveResult(name=name, status="ok"))
        except Exception as e:
            return Future.immediate(SaveResult(name=name, status=f"error: {e}"))

    def load_state(self, state: dict[str, Any]) -> Future[LoadResult]:
        """Restore backend state from a serialized dict."""
        missing = _REQUIRED_STATE_KEYS - set(state.keys())
        if missing:
            return Future.immediate(
                LoadResult(status=f"error: missing keys {sorted(missing)}")
            )

        adapter_refs = state.get("adapter_refs", [])
        self._adapter_refs = list(adapter_refs)

        if adapter_refs:
            try:
                self._backend.load_checkpoint(self._model_id, adapter_refs[-1])
            except Exception as e:
                return Future.immediate(LoadResult(status=f"error: {e}"))

        return Future.immediate(LoadResult(status="ok"))

    def clear_pending_state(self) -> None:
        """No-op — SkyRL manages its own buffers."""
        pass

    def to_dict(self) -> dict[str, Any]:
        """Serialize the backend configuration."""
        return {
            "model_ref": self._config.base_model,
            "backend_type": self._config.backend_type,
            "backend_config": self._config.backend_config,
            "lora_config": self._config.lora_config,
            "training_config": self._config.training_config,
            "adapter_refs": list(self._adapter_refs),
        }

    # -- Internal -----------------------------------------------------------

    @staticmethod
    def _to_forward_backward_input(gen_output: dict[str, Any]) -> dict[str, Any]:
        """Translate GeneratorOutput → ForwardBackwardInput for SkyRL.

        CRITICAL: passes raw rewards and trajectory_ids through.
        LfX does NOT compute advantages — SkyRL's configured advantage
        estimator handles it.
        """
        return {
            "prompt_token_ids": gen_output["prompt_token_ids"],
            "response_ids": gen_output["response_ids"],
            "rewards": gen_output["rewards"],
            "loss_masks": gen_output["loss_masks"],
            "trajectory_ids": gen_output["trajectory_ids"],
            "is_last_step": gen_output["is_last_step"],
            "rollout_logprobs": gen_output.get("rollout_logprobs"),
        }
