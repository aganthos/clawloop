#!/usr/bin/env python3
"""LfX recipe: Arithmetic RL.

Mirrors Tinker cookbook math_rl/arithmetic — trains an LLM to solve x + y = ?.
Tinker-compatible: uses the same forward_backward/optim_step protocol.

LfX addition: flip --mode to switch between weight training (SkyRL GRPO)
and harness learning (prompt optimization via reflector LLM). Or use "full"
for both simultaneously — failures improve prompts, successes train weights.

    # Harness learning (no GPU, uses LLM for prompt optimization):
    python examples/recipes/arithmetic.py --mode harness_learning

    # Weight training (GPU, SkyRL GRPO LoRA):
    python examples/recipes/arithmetic.py --mode weight

    # Full multi-layer (GPU + LLM):
    python examples/recipes/arithmetic.py --mode full
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lfx.core.episode import Episode, EpisodeSummary, Message, StepMeta
from lfx.core.intensity import AdaptiveIntensity
from lfx.core.loop import AgentState, learning_loop
from lfx.core.reward import RewardSignal
from lfx.core.types import SampleContext
from lfx.layers.harness import Harness
from lfx.layers.router import Router
from lfx.layers.weights import Weights
from lfx.train import MODE_LAYERS

log = logging.getLogger("lfx.recipe.arithmetic")

# ---------------------------------------------------------------------------
# Arithmetic environment — mirrors Tinker's ArithmeticEnv
# ---------------------------------------------------------------------------

def _make_problems(n: int = 500, max_val: int = 100):
    """Generate n random addition problems."""
    return [(random.randint(1, max_val), random.randint(1, max_val)) for _ in range(n)]


def _check_answer(response: str, expected: int) -> tuple[float, str]:
    """Extract answer and score. Returns (reward, feedback)."""
    import re
    # Try \boxed{} first, then last number
    m = re.search(r"\\boxed\{(\-?\d+)\}", response)
    if m:
        answer = int(m.group(1))
    else:
        nums = re.findall(r"\-?\d+", response)
        answer = int(nums[-1]) if nums else None

    if answer == expected:
        return 1.0, f"Correct: {expected}"
    return 0.0, f"Wrong: got {answer}, expected {expected}"


class ArithmeticAdapter:
    """AdapterLike for arithmetic RL — mirrors Tinker's ArithmeticDatasetBuilder."""

    def __init__(self, client, problems=None):
        self._client = client
        self._problems = problems or _make_problems()

    def run_episode(self, task, agent_state) -> Episode:
        a, b = task
        question = f"What is {a} + {b}?"
        expected = a + b

        # Get prompt from harness (includes playbook if harness is active)
        try:
            prompt = agent_state.harness.sample(
                SampleContext(bench="arithmetic")
            ).result().output or "Solve and put your answer in \\boxed{}."
        except Exception:
            prompt = "Solve and put your answer in \\boxed{}."

        try:
            response = str(self._client.complete([
                {"role": "system", "content": prompt},
                {"role": "user", "content": question},
            ]))
        except Exception as e:
            log.warning("LLM call failed: %s", e)
            return Episode(
                id=Episode.new_id(), state_id="", task_id=f"{a}+{b}",
                bench="arithmetic", messages=[], step_boundaries=[], steps=[],
                summary=EpisodeSummary(filtered=True),
                metadata={"error": type(e).__name__},
            )

        reward, feedback = _check_answer(response, expected)
        summary = EpisodeSummary(total_reward=reward)
        summary.signals["outcome"] = RewardSignal(
            name="outcome", value=reward * 2 - 1, confidence=1.0,
        )

        state_id = ""
        try:
            state_id = agent_state.state_id().combined_hash
        except Exception:
            pass

        return Episode(
            id=Episode.new_id(), state_id=state_id, task_id=f"{a}+{b}",
            bench="arithmetic",
            messages=[
                Message(role="system", content=prompt),
                Message(role="user", content=question),
                Message(role="assistant", content=response),
            ],
            step_boundaries=[0],
            steps=[StepMeta(t=0, reward=reward, done=True, timing_ms=0.0)],
            summary=summary,
            metadata={"expected": expected, "feedback": feedback},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LfX Arithmetic RL (Tinker-compatible)")
    p.add_argument("--mode", choices=["weight", "harness_learning", "full"], default="harness_learning")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model for weight training")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--lora-rank", type=int, default=8)
    # LLM config (for harness reflector and task inference)
    p.add_argument("--api-base", default=os.environ.get("LFX_API_BASE", "http://127.0.0.1:8317/v1"))
    p.add_argument("--api-key", default=os.environ.get("LFX_API_KEY", "kuhhandel-bench-key"))
    p.add_argument("--task-model", default="openai/claude-haiku-4-5-20251001")
    p.add_argument("--reflector-model", default="openai/claude-sonnet-4-5-20250929")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    layers = MODE_LAYERS[args.mode]
    log.info("mode=%s layers=%s model=%s", args.mode, layers, args.model)

    # 1. Harness
    harness = Harness(system_prompts={
        "arithmetic": "Solve arithmetic problems step by step. Put your final answer in \\boxed{} notation.",
    })
    if "harness" in layers:
        from lfx.core.reflector import Reflector
        from lfx.llm import LiteLLMClient
        harness.reflector = Reflector(client=LiteLLMClient(
            model=args.reflector_model, api_key=args.api_key, api_base=args.api_base,
        ))

    # 2. Weights — SkyRL backend (Tinker-compatible)
    backend = None
    if "weights" in layers:
        from lfx.backends.skyrl import SkyRLWeightsBackend, SkyRLWeightsConfig
        backend = SkyRLWeightsBackend(SkyRLWeightsConfig(
            base_model=args.model,
            backend_type="skyrl_train",
            backend_config={
                "strategy": "fsdp2",
                "trainer.placement.colocate_all": True,
                "trainer.placement.policy_num_gpus_per_node": 1,
                "trainer.placement.ref_num_gpus_per_node": 1,
                "generator.inference_engine.num_engines": 1,
                "generator.inference_engine.tensor_parallel_size": 1,
                "trainer.train_batch_size": 8,
                "trainer.policy_mini_batch_size": 4,
                "trainer.micro_forward_batch_size_per_gpu": 2,
                "trainer.micro_train_batch_size_per_gpu": 2,
                "trainer.max_prompt_length": 128,
                "generator.sampling_params.max_generate_length": 64,
                "generator.inference_engine.gpu_memory_utilization": 0.4,
                "trainer.use_sample_packing": False,
            },
            lora_config={"rank": args.lora_rank, "alpha": args.lora_rank * 2.0},
            training_config={"loss_fn": "cross_entropy", "adam_params": {"learning_rate": 1e-4}},
        ))
        weights = Weights(model_ref=args.model, _backend=backend)
    else:
        weights = Weights()

    # 3. Task LLM + environment
    from lfx.llm import LiteLLMClient
    task_client = LiteLLMClient(model=args.task_model, api_key=args.api_key, api_base=args.api_base)
    problems = _make_problems(200)
    adapter = ArithmeticAdapter(client=task_client, problems=problems)

    # 4. Run
    agent_state = AgentState(
        harness=harness, router=Router(), weights=weights,
        inference_url=getattr(backend, "inference_url", None) if backend else None,
    )

    agent_state, state_id = learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=problems,
        n_episodes=args.episodes,
        n_iterations=args.iterations,
        active_layers=layers,
        intensity=AdaptiveIntensity() if "harness" in layers else None,
    )

    print(f"\nDone. Final state: {state_id.combined_hash[:12]}")
    if hasattr(harness, 'playbook') and harness.playbook.entries:
        print(f"Playbook entries learned: {len(harness.playbook.entries)}")
        for e in harness.playbook.entries[:3]:
            print(f"  - {e.content[:80]}")


if __name__ == "__main__":
    main()
