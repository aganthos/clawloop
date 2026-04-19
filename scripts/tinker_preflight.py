"""Tinker SDK preflight validation.

Validates that the real Tinker SDK surface matches the assumptions made in
the v5 design doc. Does not swallow exceptions — the whole point of this
spike is to surface SDK drift loudly.

Run:
    python clawloop/scripts/tinker_preflight.py

Requires `TINKER_API_KEY` (loaded from `clawloop/.env` via `load_env`).
"""
from __future__ import annotations

import os
import sys

import numpy as np

import tinker
from tinker import types

from clawloop.config import load_env


def main() -> int:
    load_env()

    if "TINKER_API_KEY" not in os.environ:
        raise SystemExit(
            "TINKER_API_KEY is not set. Put it in clawloop/.env "
            "(or export it) and rerun. load_env() did not find it."
        )

    # 2. ServiceClient
    service = tinker.ServiceClient()
    print("ok: ServiceClient()")

    # 3. LoRA training client
    training = service.create_lora_training_client(
        base_model="Qwen/Qwen3-8B",
        rank=8,
        seed=42,
        train_attn=True,
        train_mlp=True,
        train_unembed=False,
    )
    print("ok: create_lora_training_client(..., rank, seed, train_*)")

    # 4. Tokenizer
    tokenizer = training.get_tokenizer()
    print(f"ok: training.get_tokenizer -> {type(tokenizer).__name__}")

    # 5. Base-model sampling client
    sampling = service.create_sampling_client(base_model="Qwen/Qwen3-8B")
    assert hasattr(sampling, "sample"), "sampling client missing .sample()"
    print("ok: create_sampling_client(base_model=...)")

    # 6. Sample once (futures-based). sample() takes a ModelInput prompt
    # + num_samples + SamplingParams (NOT prompt_tokens=list[int] kwargs).
    prompt_text = "Hello"
    prompt_ids = list(tokenizer.encode(prompt_text))
    prompt_input = types.ModelInput(
        chunks=[types.EncodedTextChunk(tokens=prompt_ids)]
    )
    sampling_params = types.SamplingParams(
        max_tokens=8, temperature=1.0, top_p=0.95,
    )
    fut = sampling.sample(
        prompt=prompt_input,
        num_samples=1,
        sampling_params=sampling_params,
    )
    resp = fut.result()
    assert hasattr(resp, "sequences"), "SampleResponse missing .sequences"
    assert len(resp.sequences) == 1, f"expected 1 sample, got {len(resp.sequences)}"
    seq = resp.sequences[0]
    assert hasattr(seq, "tokens"), "SampledSequence missing .tokens"
    assert hasattr(seq, "logprobs"), "SampledSequence missing .logprobs"
    assert seq.logprobs is not None and len(seq.tokens) == len(seq.logprobs), (
        f"sampled_tokens/logprobs alignment broken: "
        f"{len(seq.tokens)} tokens vs {len(seq.logprobs) if seq.logprobs else 'None'} logprobs"
    )
    print(f"ok: sample returns sequences[0].tokens+logprobs len {len(seq.tokens)} (stop={seq.stop_reason})")

    # 7. Build minimal Datum using the sampled tokens.
    sampled_tokens = list(seq.tokens)
    sampled_logprobs = list(seq.logprobs)
    tokens = prompt_ids + sampled_tokens
    n_prompt = len(prompt_ids)
    n_comp = len(sampled_tokens)
    datum = types.Datum(
        model_input=types.ModelInput(chunks=[types.EncodedTextChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": np.array(
                [0] * n_prompt + sampled_tokens, dtype=np.int64
            ),
            "logprobs": np.array(
                [0.0] * n_prompt + sampled_logprobs, dtype=np.float32
            ),
            "advantages": np.array(
                [0.0] * n_prompt + [1.0] * n_comp, dtype=np.float32
            ),
        },
    )

    # 8. Pipelined forward_backward + optim_step.
    # optim_step takes a typed AdamParams object (not a dict).
    fb_fut = training.forward_backward(
        data=[datum],
        loss_fn="importance_sampling",
        loss_fn_config=None,
    )
    opt_fut = training.optim_step(
        types.AdamParams(
            learning_rate=1e-5, beta1=0.9, beta2=0.999, eps=1e-8,
        )
    )
    fb_out = fb_fut.result()
    opt_out = opt_fut.result()  # noqa: F841 -- force resolution
    print("ok: forward_backward + optim_step pipelined")
    fb_metrics = getattr(fb_out, "metrics", None) or {}
    if isinstance(fb_metrics, dict):
        first_keys = list(fb_metrics.keys())[:5]
    else:
        first_keys = []
    print(f"    fb metrics keys: {first_keys}")

    # 9. Atomic save + fresh sampling client.
    # Real signature: save_weights_and_get_sampling_client(name, retry_config) -> SamplingClient.
    # No ttl_seconds. Returns SamplingClient directly.
    new_sampling = training.save_weights_and_get_sampling_client("preflight_iter_0")
    assert hasattr(new_sampling, "sample"), (
        "save_weights_and_get_sampling_client must return a SamplingClient"
    )
    print("ok: save_weights_and_get_sampling_client -> SamplingClient")

    print("\nALL PREFLIGHT CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
