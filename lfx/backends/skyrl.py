"""SkyRLWeightsBackend — wraps SkyRL's AbstractBackend for LoRA/GRPO training.

All SkyRL config passes through as dicts — LfX does not interpret SkyRL's
configuration knobs.

The AbstractBackend.forward_backward takes PreparedModelPassBatch which
requires pre-computed advantages.  We compute GRPO advantages (group-mean
subtraction across rollouts sharing the same task_id) before calling
the backend.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

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
                from skyrl.backends.jax import JaxBackend, JaxBackendConfig

                jax_cfg = JaxBackendConfig(**config.backend_config)
                self._backend = JaxBackend(config.base_model, jax_cfg)
            elif config.backend_type == "skyrl_train":
                from skyrl.backends.skyrl_train_backend import (
                    SkyRLTrainBackend,
                    SkyRLTrainBackendOverrides,
                )

                train_cfg = SkyRLTrainBackendOverrides(**config.backend_config)
                self._backend = SkyRLTrainBackend(config.base_model, train_cfg)
            else:
                raise ValueError(f"Unknown backend_type: {config.backend_type!r}")
        except Exception as e:
            if isinstance(e, SkyRLBackendInitError):
                raise
            raise SkyRLBackendInitError(BackendError.from_exception(e)) from e

        # 4. Create model with LoRA config
        self._model_id = f"lfx-{config.base_model.replace('/', '-')}"
        try:
            from skyrl.tinker.types import LoraConfig

            lora = LoraConfig(
                rank=config.lora_config.get("rank", 8),
                alpha=config.lora_config.get("alpha", 16.0),
                seed=config.lora_config.get("seed", 42),
                train_attn=config.lora_config.get("train_attn", True),
                train_mlp=config.lora_config.get("train_mlp", True),
                train_unembed=config.lora_config.get("train_unembed", False),
            )
            self._backend.create_model(self._model_id, lora)
        except Exception as e:
            raise SkyRLBackendInitError(BackendError.from_exception(e)) from e

        # 5. Store inference_url if configured
        if config.backend_config.get("enable_http_endpoint"):
            self.inference_url = getattr(self._backend, "inference_url", None)

        # 6. Create exporter
        self._exporter = SkyRLExporter(tokenizer=tokenizer)

    # -- Layer protocol -----------------------------------------------------

    def forward_backward(self, data: Datum) -> Future[FBResult]:
        """Export episodes → GeneratorOutput → PreparedModelPassBatch → backend."""
        try:
            gen_output = self._exporter.export(data.episodes)
            prepared = self._to_prepared_batch(gen_output)
            result = self._backend.forward_backward(prepared)
            # result is dict[str, ForwardBackwardOutput | ErrorResponse]
            metrics: dict[str, Any] = {}
            if isinstance(result, dict):
                for _req_id, output in result.items():
                    # Check for ErrorResponse
                    if hasattr(output, "error") and hasattr(output, "status"):
                        return Future.immediate(FBResult(
                            status="error",
                            metrics={"error": BackendError(
                                code="backend_unreachable",
                                message=output.error,
                                recoverable=True,
                            )},
                        ))
                    if hasattr(output, "metrics") and output.metrics:
                        metrics.update(output.metrics)
                    if hasattr(output, "loss_fn_outputs"):
                        metrics["loss_fn_outputs"] = output.loss_fn_outputs
            return Future.immediate(FBResult(status="ok", metrics=metrics))
        except Exception as e:
            err = BackendError.from_exception(e)
            return Future.immediate(FBResult(status="error", metrics={"error": err}))

    def optim_step(self) -> Future[OptimResult]:
        """Build OptimStepInput from training_config and call backend."""
        try:
            from skyrl.tinker.types import AdamParams, OptimStepInput

            adam_cfg = self._config.training_config.get("adam_params", {})
            optim_input = OptimStepInput(
                adam_params=AdamParams(
                    learning_rate=adam_cfg.get("learning_rate", 1e-5),
                    beta1=adam_cfg.get("beta1", 0.9),
                    beta2=adam_cfg.get("beta2", 0.999),
                    eps=adam_cfg.get("eps", 1e-8),
                    weight_decay=adam_cfg.get("weight_decay", 0.0),
                ),
            )
            result = self._backend.optim_step(self._model_id, optim_input)
            metrics = result.metrics if result.metrics else {}
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
            # AbstractBackend signature: save_checkpoint(output_path, model_id)
            self._backend.save_checkpoint(name, self._model_id)
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
                # AbstractBackend signature: load_checkpoint(checkpoint_path, model_id)
                self._backend.load_checkpoint(adapter_refs[-1], self._model_id)
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

    def _to_prepared_batch(self, gen_output: dict[str, Any]) -> Any:
        """Translate GeneratorOutput → PreparedModelPassBatch for SkyRL.

        Constructs the real Pydantic type that AbstractBackend.forward_backward
        expects.  Computes GRPO advantages (group-mean subtraction across
        rollouts sharing the same task_id via trajectory_ids).
        """
        from skyrl.tinker.types import PreparedModelPassBatch

        prompt_ids_list = gen_output["prompt_token_ids"]
        response_ids_list = gen_output["response_ids"]
        loss_masks_list = gen_output["loss_masks"]
        rewards_list = gen_output["rewards"]
        trajectory_ids = gen_output.get("trajectory_ids", [])
        logprobs_list = gen_output.get("rollout_logprobs") or [None] * len(prompt_ids_list)
        loss_fn = self._config.training_config.get("loss_fn", "cross_entropy")
        loss_fn_config = self._config.training_config.get("loss_fn_config")

        n = len(prompt_ids_list)

        # -- Compute GRPO advantages (group-mean subtraction) ----------------
        # Group rewards by instance_id (task_id), compute mean per group,
        # then advantage = reward - group_mean.
        group_rewards: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i in range(n):
            instance_id = trajectory_ids[i].instance_id if i < len(trajectory_ids) else "default"
            group_rewards[instance_id].append((i, rewards_list[i]))

        advantages_per_seq: list[float] = [0.0] * n
        for _instance_id, entries in group_rewards.items():
            group_mean = sum(r for _, r in entries) / len(entries) if entries else 0.0
            for idx, reward in entries:
                advantages_per_seq[idx] = reward - group_mean

        # -- Build per-sequence arrays ---------------------------------------
        all_input_ids: list[list[int]] = []
        all_targets: list[list[int]] = []
        all_token_weights: list[list[float]] = []
        all_sampling_logprobs: list[list[float]] = []
        all_advantages: list[list[float]] = []

        for i in range(n):
            full_ids = prompt_ids_list[i] + response_ids_list[i]
            resp_ids = response_ids_list[i]
            mask = loss_masks_list[i] if i < len(loss_masks_list) else [1.0] * len(resp_ids)

            all_input_ids.append(full_ids)
            all_targets.append(resp_ids)
            all_token_weights.append([float(w) for w in mask])

            # Logprobs: use rollout logprobs if available, else empty
            # (zeros would mean P=1.0 which distorts IS ratios)
            lp = logprobs_list[i] if logprobs_list[i] is not None else []
            all_sampling_logprobs.append(lp)

            # Broadcast sequence-level advantage to all response tokens
            all_advantages.append([advantages_per_seq[i]] * len(resp_ids))

        # -- Build request_batch_slices (one slice per sequence) -------------
        request_id = uuid4().hex
        request_batch_slices = [
            (request_id, self._model_id, i, i + 1) for i in range(n)
        ]

        return PreparedModelPassBatch(
            all_input_ids=all_input_ids,
            all_targets=all_targets,
            all_token_weights=all_token_weights,
            all_sampling_logprobs=all_sampling_logprobs,
            all_advantages=all_advantages,
            all_model_ids=[self._model_id] * n,
            all_loss_fns=[loss_fn] * n,
            all_loss_fn_configs=[loss_fn_config] * n,
            request_batch_slices=request_batch_slices,
        )
