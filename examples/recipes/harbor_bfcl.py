#!/usr/bin/env python3
"""ClawLoop recipe: Harbor BFCL (Berkeley Function Calling Leaderboard).

Trains an agent to make correct function calls using Harbor's sandboxed
execution environment + BFCL tasks.

Two modes:
  weight           SkyRL/Tinker GRPO — model generates rollouts via vLLM,
                   Harbor verifies in Docker, GRPO trains the weights.
  harness_learning ClawLoop harness — an API model runs the trials, reflector
                   analyzes failures and evolves the system prompt.

Weight mode (GPU + Docker):
    # Download BFCL tasks (small parity subset)
    harbor run -d "bfcl_parity@1.0" --download-only -p ~/data/bfcl_parity
    # Train
    python examples/recipes/harbor_bfcl.py --mode weight --task-dir ~/data/bfcl_parity

Harness mode (no GPU, needs API key + Docker):
    python examples/recipes/harbor_bfcl.py --mode harness_learning --task-dir ~/data/bfcl_parity
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger("clawloop.recipe.harbor_bfcl")


# ---------------------------------------------------------------------------
# Harness learning — ClawLoop loop with Harbor trials
# ---------------------------------------------------------------------------

def run_harness_learning(args):
    """Prompt optimization via reflector. Harbor runs real agent trials."""
    from pathlib import Path

    from clawloop.core.intensity import AdaptiveIntensity
    from clawloop.core.loop import AgentState, learning_loop
    from clawloop.core.reflector import Reflector
    from clawloop.envs.harbor import HarborAdapter, HarborTaskEnvironment
    from clawloop.layers.harness import Harness
    from clawloop.layers.router import Router
    from clawloop.layers.weights import Weights
    from clawloop.llm import LiteLLMClient

    task_dirs = _find_task_dirs(args.task_dir)
    if not task_dirs:
        print(f"ERROR: No Harbor tasks found in {args.task_dir}")
        sys.exit(1)
    log.info("Found %d Harbor tasks in %s", len(task_dirs), args.task_dir)

    harness = Harness(system_prompts={
        "harbor": (
            "You are a function-calling assistant. When given a function schema "
            "and a natural language request, write the correct JSON function call. "
            "Write your result to /app/result.json as a JSON array."
        ),
    })
    harness.reflector = Reflector(client=LiteLLMClient(
        model=args.reflector_model, api_key=args.api_key, api_base=args.api_base,
    ))

    trial_config = {
        "agent": {
            "name": args.agent,
            "kwargs": {"store_all_messages": True},
        },
        "environment": {"type": "docker"},
    }
    if args.agent != "oracle":
        trial_config["agent"]["model_name"] = args.task_model
        trial_config["agent"]["kwargs"].update({
            "max_turns": 16, "temperature": 0.7,
            "api_base": args.api_base, "api_key": args.api_key,
        })

    envs = [
        HarborTaskEnvironment(
            task_dir=Path(d), trial_config=trial_config, train_on_truncated=True,
        )
        for d in task_dirs[:args.max_tasks]
    ]
    adapter = HarborAdapter(envs)
    tasks = [e.task_id for e in envs]

    state, sid = learning_loop(
        adapter=adapter,
        agent_state=AgentState(harness=harness, router=Router(), weights=Weights()),
        tasks=tasks, n_episodes=args.episodes, n_iterations=args.iterations,
        active_layers=["harness", "router"],
        intensity=AdaptiveIntensity(),
    )
    print(f"\nDone. State: {sid.combined_hash[:12]}")
    if harness.playbook.entries:
        print(f"Playbook entries: {len(harness.playbook.entries)}")
        for e in harness.playbook.entries[:3]:
            print(f"  - {e.content[:80]}")


# ---------------------------------------------------------------------------
# Weight training — real Tinker via SkyRL Harbor integration
# ---------------------------------------------------------------------------

def run_weight_training(args):
    """GRPO weight training. SkyRL serves the model, Harbor runs trials."""
    import ray
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.entrypoints.main_base import validate_cfg
    from skyrl.train.utils.utils import initialize_ray

    task_dirs = _find_task_dirs(args.task_dir)
    if not task_dirs:
        print(f"ERROR: No Harbor tasks found in {args.task_dir}")
        sys.exit(1)
    log.info("Found %d Harbor tasks in %s", len(task_dirs), args.task_dir)

    # Use SkyRL's Harbor integration entry point
    overrides = {
        "data.train_data": f"['{args.task_dir}']",
        "data.val_data": f"['{args.task_dir}']",
        "trainer.algorithm.advantage_estimator": "grpo",
        "trainer.policy.model.path": args.model,
        "trainer.placement.colocate_all": True,
        "trainer.policy.model.lora.rank": args.lora_rank,
        "trainer.policy.model.lora.alpha": args.lora_rank,
        "trainer.strategy": "fsdp2",
        "trainer.placement.policy_num_gpus_per_node": 1,
        "trainer.placement.ref_num_gpus_per_node": 1,
        "generator.inference_engine.num_engines": 1,
        "generator.inference_engine.tensor_parallel_size": 1,
        "generator.inference_engine.enable_http_endpoint": True,
        "trainer.epochs": args.iterations,
        "trainer.eval_before_train": False,
        "trainer.eval_interval": 999,
        "trainer.update_epochs_per_batch": 1,
        "trainer.train_batch_size": 8,
        "trainer.policy_mini_batch_size": 4,
        "trainer.micro_forward_batch_size_per_gpu": 2,
        "trainer.micro_train_batch_size_per_gpu": 2,
        "trainer.max_prompt_length": 4096,
        "generator.sampling_params.max_generate_length": 4096,
        "trainer.policy.optimizer_config.lr": 3e-5,
        "trainer.algorithm.use_kl_loss": True,
        "generator.inference_engine.backend": "vllm",
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.weight_sync_backend": "nccl",
        "generator.inference_engine.async_engine": True,
        "generator.batched": False,
        "generator.n_samples_per_prompt": 2,
        "generator.inference_engine.gpu_memory_utilization": 0.5,
        "trainer.use_sample_packing": False,
        "trainer.logger": "console",
        "trainer.project_name": "clawloop_harbor_bfcl",
        "trainer.run_name": f"bfcl_{args.model.split('/')[-1]}",
        "trainer.resume_mode": "none",
        "trainer.ckpt_interval": 999,
        "trainer.ckpt_path": os.path.expanduser("~/ckpts/harbor_bfcl"),
    }

    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    validate_cfg(cfg)

    @ray.remote(num_cpus=1)
    def entrypoint(cfg, task_dir):
        # Use SkyRL's Harbor integration
        from examples.train_integrations.harbor.entrypoints.main_harbor import HarborExp
        exp = HarborExp(cfg, harbor_task_dir=task_dir)
        exp.run()

    initialize_ray(cfg)
    ray.get(entrypoint.remote(cfg, args.task_dir))
    print("\nWeight training done.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_task_dirs(base_dir: str) -> list[str]:
    """Find Harbor task directories (contain instruction.md)."""
    base = os.path.expanduser(base_dir)
    tasks = []
    for md in glob.glob(os.path.join(base, "**/instruction.md"), recursive=True):
        tasks.append(os.path.dirname(md))
    return sorted(tasks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ClawLoop Harbor BFCL — Tinker-compatible")
    p.add_argument("--mode", choices=["weight", "harness_learning"], required=True)
    p.add_argument("--task-dir", required=True, help="Path to Harbor BFCL task directory")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--max-tasks", type=int, default=20, help="Max tasks to use")
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--agent", default="oracle", help="Harbor agent (oracle, terminus-2)")
    p.add_argument("--api-base", default=os.environ.get("CLAWLOOP_API_BASE", "http://localhost:11434/v1"))
    p.add_argument("--api-key", default=os.environ.get("CLAWLOOP_API_KEY", ""))
    p.add_argument("--task-model", default="gemini/gemini-2.0-flash-lite")
    p.add_argument("--reflector-model", default="openai/claude-sonnet-4-5-20250929")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("mode=%s model=%s task_dir=%s", args.mode, args.model, args.task_dir)

    if args.mode == "weight":
        run_weight_training(args)
    else:
        run_harness_learning(args)


if __name__ == "__main__":
    main()
