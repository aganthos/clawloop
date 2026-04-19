"""Tinker weight-tuning demo — 1 iter of RL against an OpenSpiel game.

**What this demonstrates**

A minimal, runnable example of how ClawLoop wraps Thinking Machines Lab's
managed [Tinker](https://thinkingmachines.ai) LoRA-training service for
reinforcement learning. Runs Qwen3-8B for ONE iter (8 Blackjack episodes)
against Tinker, producing per-iter metrics + a durable adapter checkpoint.

**What it's not**

A benchmark. One iter is for plumbing validation. For real runs see
``scripts/run_pilot.py`` + the YAMLs alongside this file.

**Prerequisites**

- ``TINKER_API_KEY`` in ``clawloop/.env`` (or as env var).
- Install the games extra:    ``uv sync --extra games``

**Run**

    uv run python examples/tinker_weight_demo.py

**What you'll see**

- Rich console table per iter (``avg_reward``, ``loss:sum``, ``n_datums``).
- ``pilot_runs/tinker_demo/experiment.jsonl`` with full trace.
- A durable Tinker checkpoint at ``tinker://.../weights/iter_0`` enumerable
  via ``backend.list_tinker_checkpoints()``.

**Adapting this to another env**

The pieces you'd change are marked with ``[ADAPT]`` comments. The short
version: your env's ``Episode``s must carry ``prompt_tokens``,
``sampled_tokens``, and ``sampling_logprobs`` in ``StepMeta.info`` — those
are the alignment payload the exporter reads directly so it never has to
re-tokenize anything.
"""
from __future__ import annotations

import logging
import sys

from clawloop.config import load_env
from clawloop.train import TrainConfig, train


def main() -> int:
    # 1. Bring TINKER_API_KEY into the process env.  Works from clawloop/.env
    #    (preferred — gitignored) or a process-level export.
    load_env()

    # 2. Define the training config.  This is the same shape you'd put in
    #    a YAML and pass to `scripts/run_pilot.py`; we inline it here so
    #    the moving parts are visible on one screen.
    config = TrainConfig(
        mode="weight",                 # [ADAPT] "weight" trains via a weights backend
        env_type="openspiel",          # [ADAPT] swap for your env_type once registered
        weight_backend="tinker",       # [ADAPT] stays "tinker" for Tinker-backed training

        # [ADAPT] env-specific config — put anything your `_build_<env>` needs here.
        # `episodes_per_iter` is derived automatically from `seeds × episodes_per_seed`
        # by `effective_episodes_per_iter(config)` — here: 4 seeds * 2 = 8 episodes/iter.
        openspiel={
            "game_name": "blackjack",
            "seeds": [0, 1, 2, 3],
            "episodes_per_seed": 2,    # GRPO needs K >= 2 per scenario for variance
            "prompt_style": "canonical",
            "rethink_k": 3,
            "max_turns": 10,
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 128,
        },

        # Tinker LoRA training knobs.  `base_model` must be in
        # `service.get_server_capabilities().supported_models`; run
        # `scripts/tinker_preflight.py` to see the live list for your account.
        tinker={
            "base_model": "Qwen/Qwen3-8B",
            "lora_rank": 8,
            "seed": 42,
            "train_attn": True,
            "train_mlp": True,
            "train_unembed": False,
            "loss_fn": "importance_sampling",
            "adam_params": {
                "learning_rate": 1.0e-5,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1.0e-8,
            },
        },

        n_iterations=1,                # Smoke — one iter. Bump for real runs.
        output_dir="pilot_runs/tinker_demo",

        # Optional: mirror metrics to wandb.  Requires `WANDB_API_KEY` in
        # env or .env.  Disabled by default so the demo runs without signup.
        # wandb_project="clawloop-tinker-demo",
    )

    # 3. Run. This builds the ClawLoop layers, wires TinkerWeightsBackend,
    #    fans out 8 Blackjack rollouts concurrently against Tinker's
    #    SamplingClient (see `asyncio.gather` + `asyncio.wrap_future` in
    #    `OpenSpielGameAdapter.run_episodes_batch`), aggregates per-task
    #    GRPO advantages, calls `forward_backward` + `optim_step` as
    #    pipelined futures, and finally writes a durable Tinker checkpoint
    #    via `save_state` + `save_weights_and_get_sampling_client`.
    agent_state, state_id = train(config)

    # 4. Inspect what landed on Tinker's side.  The backend exposes
    #    `list_tinker_checkpoints()` which calls RestClient.list_checkpoints
    #    under the hood — this is how you'd enumerate, delete, or reload
    #    saved LoRA adapters programmatically.
    backend = getattr(agent_state.weights, "_backend", None)
    if backend is not None and hasattr(backend, "list_tinker_checkpoints"):
        checkpoints = backend.list_tinker_checkpoints()
        print()
        print(f"Tinker model_id: {backend.model_id}")
        print(f"Durable checkpoints ({len(checkpoints)}):")
        for ck in checkpoints:
            print(
                f"  {ck.get('checkpoint_type')}  "
                f"{ck.get('tinker_path')}  "
                f"({ck.get('size_bytes')} bytes, "
                f"expires {ck.get('expires_at')})"
            )

    print()
    print(f"Final state_id: {state_id.combined_hash[:12]}")
    print(f"Logs + reward curves: {config.output_dir}/experiment.jsonl")
    print()
    print("Open the live viewer:")
    print("  python -m http.server 8770 --bind 127.0.0.1")
    print(
        "  then http://localhost:8770/clawloop/static/learning_viewer.html"
        f"?exp=/{config.output_dir}/experiment.jsonl"
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main())
