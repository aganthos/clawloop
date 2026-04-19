#!/usr/bin/env python3
"""ClawLoop recipe: A2A CRMArena (Entropic).

Trains on Salesforce CRMArenaPro tasks — an A2A benchmark where a purple
agent (ClawLoop-controlled) interacts with a green evaluator agent to solve CRM
tasks. 7-dimension scoring: functional, drift_adaptation, token_efficiency,
query_efficiency, error_recovery, trajectory_efficiency, hallucination_rate.

Two learning modes via --mode:
  weight           SkyRL/Tinker GRPO on episodes collected from CRMArena trials.
                   Purple agent generates responses, green agent scores them,
                   SkyRL trains the weights.
  harness_learning ClawLoop harness layer — reflector analyzes failures across the
                   7 reward dimensions and evolves the system prompt.

Harness mode (no GPU, needs LLM API):
    python examples/recipes/a2a_crmarena.py --mode harness_learning \
        --task-ids 0 1 2 --iterations 3

Weight mode (GPU):
    python examples/recipes/a2a_crmarena.py --mode weight \
        --task-ids 0 1 2 --iterations 1
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger("clawloop.recipe.a2a_crmarena")


# ---------------------------------------------------------------------------
# Harness learning — prompt optimization via reflector
# ---------------------------------------------------------------------------


def run_harness_learning(args):
    from clawloop.core.intensity import AdaptiveIntensity
    from clawloop.core.loop import AgentState, learning_loop
    from clawloop.environments.entropic import EntropicAdapter
    from clawloop.learning_layers.harness import Harness
    from clawloop.learning_layers.router import Router
    from clawloop.learning_layers.weights import Weights
    from examples.recipes.common import build_local_evolver

    harness = Harness(
        system_prompts={
            "entropic": (
                "You are a CRM assistant. Help users with service requests accurately. "
                "Verify information before making changes. Handle schema drift gracefully."
            ),
        },
        evolver=build_local_evolver(args.reflector_model, args.api_key, args.api_base),
    )

    adapter = EntropicAdapter()
    adapter.setup(
        {
            "model": args.task_model,
            "entropic_bench_path": args.bench_path,
            "task_ids": args.task_ids,
            "task_limit": len(args.task_ids) if args.task_ids else 3,
            "api_base": args.api_base,
            "api_key": args.api_key,
        }
    )

    agent_state = AgentState(harness=harness, router=Router(), weights=Weights())
    tasks = [f"base_{i}" for i in range(len(args.task_ids) if args.task_ids else 3)]

    log.info("Running harness learning: %d tasks, %d iterations", len(tasks), args.iterations)
    agent_state, state_id = learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=tasks,
        n_episodes=len(tasks),
        n_iterations=args.iterations,
        active_layers=["harness", "router"],
        intensity=AdaptiveIntensity(),
    )
    print(f"\nDone. State: {state_id.combined_hash[:12]}")
    if harness.playbook.entries:
        print(f"Playbook entries: {len(harness.playbook.entries)}")
        for e in harness.playbook.entries[:3]:
            print(f"  - {e.content[:80]}")


# ---------------------------------------------------------------------------
# Weight training — SkyRL GRPO on CRMArena episodes
# ---------------------------------------------------------------------------


def run_weight_training(args):
    from clawloop.core.loop import AgentState, learning_loop
    from clawloop.environments.entropic import EntropicAdapter
    from clawloop.learning_layers.harness import Harness
    from clawloop.learning_layers.router import Router
    from clawloop.learning_layers.weights import Weights
    from clawloop.weight_backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig

    harness = Harness(
        system_prompts={
            "entropic": (
                "You are a CRM assistant. Help users with service requests accurately. "
                "Verify information before making changes. Handle schema drift gracefully."
            ),
        }
    )

    # SkyRL backend for weight training
    log.info("Initializing SkyRL backend with %s...", args.model)
    backend = SkyRLWeightsBackend(
        SkyRLWeightsConfig(
            base_model=args.model,
            backend_type="skyrl_train",
            backend_config={
                "strategy": "fsdp2",
                "trainer.placement.colocate_all": True,
                "trainer.placement.policy_num_gpus_per_node": 1,
                "trainer.placement.ref_num_gpus_per_node": 1,
                "generator.inference_engine.num_engines": 1,
                "generator.inference_engine.tensor_parallel_size": 1,
                "trainer.train_batch_size": 4,
                "trainer.policy_mini_batch_size": 4,
                "trainer.micro_forward_batch_size_per_gpu": 2,
                "trainer.micro_train_batch_size_per_gpu": 2,
                "trainer.max_prompt_length": 4096,
                "generator.sampling_params.max_generate_length": 2048,
                "generator.inference_engine.gpu_memory_utilization": 0.4,
                "trainer.use_sample_packing": False,
            },
            lora_config={"rank": args.lora_rank, "alpha": args.lora_rank * 2.0},
            training_config={"loss_fn": "cross_entropy", "adam_params": {"learning_rate": 1e-5}},
        )
    )
    weights = Weights(model_ref=args.model, _backend=backend)
    log.info("SkyRL backend ready")

    # Entropic adapter collects episodes from CRMArena
    adapter = EntropicAdapter()
    adapter.setup(
        {
            "model": args.task_model,
            "entropic_bench_path": args.bench_path,
            "task_ids": args.task_ids,
            "task_limit": len(args.task_ids) if args.task_ids else 3,
            "api_base": args.api_base,
            "api_key": args.api_key,
        }
    )

    agent_state = AgentState(
        harness=harness,
        router=Router(),
        weights=weights,
        inference_url=getattr(backend, "inference_url", None),
    )
    tasks = [f"base_{i}" for i in range(len(args.task_ids) if args.task_ids else 3)]

    log.info("Running weight training: %d tasks, %d iterations", len(tasks), args.iterations)
    agent_state, state_id = learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=tasks,
        n_episodes=len(tasks),
        n_iterations=args.iterations,
        active_layers=["weights"],
    )
    print(f"\nDone. State: {state_id.combined_hash[:12]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="ClawLoop A2A CRMArena — Tinker-compatible")
    p.add_argument("--mode", choices=["weight", "harness_learning"], required=True)
    p.add_argument(
        "--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model for weight training"
    )
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument(
        "--task-ids", type=int, nargs="+", default=[0, 1, 2], help="CRMArena task indices"
    )
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--bench-path", default="benchmarks/a2a/entropic-crmarenapro")
    p.add_argument(
        "--api-base", default=os.environ.get("CLAWLOOP_API_BASE", "http://localhost:11434/v1")
    )
    p.add_argument("--api-key", default=os.environ.get("CLAWLOOP_API_KEY", ""))
    p.add_argument("--task-model", default="openai/claude-haiku-4-5-20251001")
    p.add_argument("--reflector-model", default="openai/claude-sonnet-4-5-20250929")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    log.info("mode=%s model=%s tasks=%s", args.mode, args.model, args.task_ids)

    if args.mode == "weight":
        run_weight_training(args)
    else:
        run_harness_learning(args)


if __name__ == "__main__":
    main()
