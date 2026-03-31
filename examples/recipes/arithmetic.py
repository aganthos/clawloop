#!/usr/bin/env python3
"""ClawLoop recipe: Arithmetic RL.

Mirrors Tinker cookbook math_rl/arithmetic — trains to solve x + y = ?.

Two learning modes via --mode:
  weight           Uses SkyRL/Tinker natively — model generates its own rollouts
                   via vLLM, environment scores them, GRPO trains. Real Tinker.
  harness_learning Uses ClawLoop harness layer — an external LLM generates responses,
                   reflector analyzes failures, playbook evolves the prompt.

Same environment, same scoring, different learning space.

Weight mode (GPU, real Tinker):
    python examples/recipes/arithmetic_dataset.py --output_dir ~/data/arithmetic
    python examples/recipes/arithmetic.py --mode weight

Harness mode (no GPU, prompt optimization):
    python examples/recipes/arithmetic.py --mode harness_learning
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger("clawloop.recipe.arithmetic")


# ---------------------------------------------------------------------------
# Harness learning path — ClawLoop learning loop
# ---------------------------------------------------------------------------

def run_harness_learning(args):
    """Prompt optimization via reflector LLM. No GPU needed."""
    from clawloop.core.episode import Episode, EpisodeSummary, Message, StepMeta
    from clawloop.core.intensity import AdaptiveIntensity
    from clawloop.core.loop import AgentState, learning_loop
    from clawloop.core.reflector import Reflector
    from clawloop.core.reward import RewardSignal
    from clawloop.core.types import SampleContext
    from clawloop.learning_layers.harness import Harness
    from clawloop.learning_layers.router import Router
    from clawloop.learning_layers.weights import Weights
    from clawloop.llm import LiteLLMClient

    harness = Harness(system_prompts={
        "arithmetic": "Solve arithmetic problems step by step. Put your final answer in \\boxed{} notation.",
    })
    harness.reflector = Reflector(client=LiteLLMClient(
        model=args.reflector_model, api_key=args.api_key, api_base=args.api_base,
    ))
    task_client = LiteLLMClient(model=args.task_model, api_key=args.api_key, api_base=args.api_base)

    problems = [(random.randint(1, 100), random.randint(1, 100)) for _ in range(200)]

    class Adapter:
        def run_episode(self, task, agent_state):
            a, b = task
            expected = a + b
            try:
                prompt = agent_state.harness.sample(SampleContext(bench="arithmetic")).result().output
            except Exception:
                prompt = "Solve and put your answer in \\boxed{}."
            try:
                response = str(task_client.complete([
                    {"role": "system", "content": prompt or "Solve and put your answer in \\boxed{}."},
                    {"role": "user", "content": f"What is {a} + {b}?"},
                ]))
            except Exception as e:
                log.warning("LLM failed: %s", e)
                return Episode(id=Episode.new_id(), state_id="", task_id=f"{a}+{b}",
                    bench="arithmetic", messages=[], step_boundaries=[], steps=[],
                    summary=EpisodeSummary(filtered=True), metadata={"error": str(e)})

            m = re.search(r"\\boxed\{(\-?\d+)\}", response)
            answer = int(m.group(1)) if m else None
            reward = 1.0 if answer == expected else 0.0
            summary = EpisodeSummary(total_reward=reward)
            summary.signals["outcome"] = RewardSignal(name="outcome", value=reward * 2 - 1, confidence=1.0)
            sid = ""
            try: sid = agent_state.state_id().combined_hash
            except Exception: pass
            return Episode(id=Episode.new_id(), state_id=sid, task_id=f"{a}+{b}",
                bench="arithmetic",
                messages=[Message(role="system", content=prompt or ""),
                          Message(role="user", content=f"What is {a} + {b}?"),
                          Message(role="assistant", content=response)],
                step_boundaries=[0],
                steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=0.0)],
                summary=summary, metadata={"expected": expected, "correct": answer == expected})

    state, sid = learning_loop(
        adapter=Adapter(), agent_state=AgentState(harness=harness, router=Router(), weights=Weights()),
        tasks=problems, n_episodes=args.episodes, n_iterations=args.iterations,
        active_layers=["harness", "router"],
        intensity=AdaptiveIntensity(),
    )
    print(f"\nDone. State: {sid.combined_hash[:12]}")
    if harness.playbook.entries:
        print(f"Playbook entries: {len(harness.playbook.entries)}")
        for e in harness.playbook.entries[:3]:
            print(f"  - {e.content[:80]}")


# ---------------------------------------------------------------------------
# Weight training path — real Tinker via SkyRL
# ---------------------------------------------------------------------------

def run_weight_training(args):
    """GRPO weight training via SkyRL. Model generates its own rollouts."""
    import ray
    from skyrl.train.config import SkyRLTrainConfig
    from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
    from skyrl.train.utils.utils import initialize_ray
    from skyrl_gym.envs import register

    data_dir = os.path.expanduser(args.data_dir)
    if not os.path.exists(os.path.join(data_dir, "train.parquet")):
        print(f"ERROR: {data_dir}/train.parquet not found.")
        print(f"Run: python examples/recipes/arithmetic_dataset.py --output_dir {data_dir}")
        sys.exit(1)

    overrides = {
        "data.train_data": f"['{data_dir}/train.parquet']",
        "data.val_data": f"['{data_dir}/validation.parquet']",
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
        "trainer.epochs": args.iterations,
        "trainer.eval_before_train": False,
        "trainer.eval_interval": 999,
        "trainer.update_epochs_per_batch": 1,
        "trainer.train_batch_size": 16,
        "trainer.policy_mini_batch_size": 8,
        "trainer.micro_forward_batch_size_per_gpu": 4,
        "trainer.micro_train_batch_size_per_gpu": 4,
        "trainer.max_prompt_length": 128,
        "generator.sampling_params.max_generate_length": 64,
        "trainer.policy.optimizer_config.lr": 1e-4,
        "trainer.algorithm.use_kl_loss": True,
        "generator.inference_engine.backend": "vllm",
        "generator.inference_engine.run_engines_locally": True,
        "generator.inference_engine.weight_sync_backend": "nccl",
        "generator.inference_engine.async_engine": True,
        "generator.batched": True,
        "environment.env_class": "arithmetic",
        "generator.n_samples_per_prompt": 2,
        "generator.inference_engine.gpu_memory_utilization": 0.5,
        "trainer.use_sample_packing": False,
        "trainer.logger": "console",
        "trainer.project_name": "clawloop_arithmetic",
        "trainer.run_name": f"arithmetic_{args.model.split('/')[-1]}",
        "trainer.resume_mode": "none",
        "trainer.ckpt_interval": 999,
        "trainer.ckpt_path": os.path.expanduser("~/ckpts/arithmetic"),
    }

    cfg = SkyRLTrainConfig.from_cli_overrides(overrides)
    validate_cfg(cfg)

    @ray.remote(num_cpus=1)
    def entrypoint(cfg):
        register(id="arithmetic", entry_point="examples.recipes.arithmetic_env:ArithmeticEnv")
        exp = BasePPOExp(cfg)
        exp.run()

    initialize_ray(cfg)
    ray.get(entrypoint.remote(cfg))
    print("\nWeight training done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ClawLoop Arithmetic RL — Tinker-compatible")
    p.add_argument("--mode", choices=["weight", "harness_learning"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--data-dir", default="~/data/arithmetic")
    p.add_argument("--api-base", default=os.environ.get("CLAWLOOP_API_BASE", "http://localhost:11434/v1"))
    p.add_argument("--api-key", default=os.environ.get("CLAWLOOP_API_KEY", ""))
    p.add_argument("--task-model", default="openai/claude-haiku-4-5-20251001")
    p.add_argument("--reflector-model", default="openai/claude-sonnet-4-5-20250929")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log.info("mode=%s model=%s", args.mode, args.model)

    if args.mode == "weight":
        run_weight_training(args)
    else:
        run_harness_learning(args)


if __name__ == "__main__":
    main()
