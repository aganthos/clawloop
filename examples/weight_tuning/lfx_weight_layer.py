#!/usr/bin/env python3
"""Weight tuning via the LfX weight layer (SkyRLWeightsBackend).

Demonstrates the full LfX integration path:
  Episode -> SkyRLExporter -> PreparedModelPassBatch -> SkyRLTrainBackend

This is the LfX-native way to do weight training — episodes flow through
the LfX type system and get translated to SkyRL's training format. Use this
when you want LfX to manage the learning loop (harness + weights together).

For standalone SkyRL training (Tinker-native), see gsm8k_lora.sh instead.

Requires GPU + SkyRL[fsdp] installed.

    python examples/weight_tuning/lfx_weight_layer.py
    python examples/weight_tuning/lfx_weight_layer.py --model Qwen/Qwen2.5-1.5B-Instruct
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("lfx.example.weights")


def make_math_episodes(n: int = 4):
    """Synthetic math episodes for demonstration."""
    from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
    from lfx.core.reward import RewardSignal

    problems = [
        ("What is 2 + 3?", "Let me solve: 2 + 3 = 5. #### 5", 1.0),
        ("What is 7 * 8?", "7 * 8 = 56. #### 56", 1.0),
        ("What is 100 / 4?", "100 / 4 = 25. #### 25", 1.0),
        ("What is 15 - 9?", "15 - 9 = 7. #### 7", 0.0),
    ]

    episodes = []
    for i in range(n):
        q, a, reward = problems[i % len(problems)]
        summary = EpisodeSummary(total_reward=reward)
        summary.signals["outcome"] = RewardSignal(
            name="outcome", value=reward * 2 - 1, confidence=1.0,
        )
        episodes.append(Episode(
            id=Episode.new_id(), state_id="demo", task_id=f"math-{i % len(problems)}",
            bench="math",
            messages=[
                Message(role="system", content="You are a math tutor."),
                Message(role="user", content=q),
                Message(role="assistant", content=a),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)],
            summary=summary,
        ))
    return episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2, help="Number of forward_backward + optim_step cycles")
    args = parser.parse_args()

    from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig
    from lfx.core.types import Datum

    config = SkyRLWeightsConfig(
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
            "trainer.max_prompt_length": 256,
            "generator.sampling_params.max_generate_length": 256,
            "generator.inference_engine.gpu_memory_utilization": 0.4,
            "trainer.use_sample_packing": False,
        },
        lora_config={"rank": args.lora_rank, "alpha": args.lora_rank * 2.0},
        training_config={
            "loss_fn": "cross_entropy",
            "adam_params": {"learning_rate": args.lr},
        },
    )

    log.info("Initializing SkyRLWeightsBackend with %s (LoRA rank=%d)...", args.model, args.lora_rank)
    t0 = time.time()
    backend = SkyRLWeightsBackend(config)
    log.info("Backend ready in %.1fs", time.time() - t0)

    for step in range(args.steps):
        episodes = make_math_episodes(args.episodes)
        datum = Datum(episodes=episodes)

        log.info("Step %d/%d: forward_backward (%d episodes)...", step + 1, args.steps, len(episodes))
        fb = backend.forward_backward(datum).result()
        log.info("  fb: status=%s", fb.status)

        log.info("Step %d/%d: optim_step...", step + 1, args.steps)
        opt = backend.optim_step().result()
        log.info("  optim: status=%s updates=%d", opt.status, opt.updates_applied)

    log.info("Done. %d steps completed.", args.steps)


if __name__ == "__main__":
    main()
