#!/usr/bin/env python3
"""LfX recipe: Guess the Number (multi-turn RL).

Mirrors Tinker cookbook multiplayer_rl/guess_number — LLM guesses an integer
0-1024 via binary search. Gets "Too high"/"Too low"/"Correct" feedback.
Max 10 turns per episode.

Tinker-compatible: uses the same forward_backward/optim_step protocol.

LfX addition: flip --mode to switch between weight training (SkyRL GRPO)
and harness learning (prompt optimization via reflector LLM).

    # Harness learning (no GPU):
    python examples/recipes/guess_number.py --mode harness_learning

    # Weight training (GPU, SkyRL):
    python examples/recipes/guess_number.py --mode weight

"""
from __future__ import annotations

import argparse
import logging
import os
import random
import re
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

log = logging.getLogger("lfx.recipe.guess_number")

MAX_TURNS = 10
MAX_VAL = 1024


# ---------------------------------------------------------------------------
# Guess-the-number environment — mirrors Tinker's GuessNumberEnv
# ---------------------------------------------------------------------------

class GuessNumberGame:
    """One game instance. Tracks conversation and scoring."""

    def __init__(self, target: int):
        self.target = target
        self.turns = 0
        self.done = False
        self.reward = 0.0
        self.messages: list[Message] = []

    def step(self, guess_text: str) -> str:
        """Process a guess. Returns feedback string."""
        self.turns += 1
        m = re.search(r"\d+", guess_text)
        if m is None:
            feedback = f"Invalid guess. Please guess a number between 0 and {MAX_VAL}."
            if self.turns >= MAX_TURNS:
                self.done = True
                self.reward = 0.0
                feedback += f" Game over. The number was {self.target}."
            return feedback

        guess = int(m.group())
        if guess == self.target:
            self.done = True
            # Reward based on how quickly they found it (fewer turns = higher)
            self.reward = 1.0 - (self.turns - 1) / MAX_TURNS
            return f"Correct! The number was {self.target}. Found in {self.turns} turns."
        elif guess < self.target:
            feedback = f"{guess} is too low."
        else:
            feedback = f"{guess} is too high."

        if self.turns >= MAX_TURNS:
            self.done = True
            self.reward = 0.0
            feedback += f" Game over. The number was {self.target}."
        return feedback


class GuessNumberAdapter:
    """AdapterLike for guess-the-number multi-turn RL."""

    def __init__(self, client):
        self._client = client

    def run_episode(self, task, agent_state) -> Episode:
        target = task  # task is just an int

        # Get prompt from harness
        try:
            prompt = agent_state.harness.sample(
                SampleContext(bench="guess_number")
            ).result().output or self._default_prompt()
        except Exception:
            prompt = self._default_prompt()

        game = GuessNumberGame(target)
        messages = [Message(role="system", content=prompt)]
        step_boundaries = []
        steps = []

        # Initial user message
        user_msg = f"I'm thinking of a number between 0 and {MAX_VAL}. Can you guess it?"
        messages.append(Message(role="user", content=user_msg))
        step_boundaries.append(len(messages) - 1)

        for turn in range(MAX_TURNS):
            # LLM guesses
            try:
                conv = [{"role": m.role, "content": m.content} for m in messages]
                response = str(self._client.complete(conv))
            except Exception as e:
                log.warning("LLM call failed at turn %d: %s", turn, e)
                return self._error_episode(target, str(e))

            messages.append(Message(role="assistant", content=response))

            # Environment step
            feedback = game.step(response)
            steps.append(StepMeta(
                t=turn, reward=game.reward if game.done else 0.0,
                done=game.done, timing_ms=0.0,
            ))

            if game.done:
                break

            # Feedback as next user message
            messages.append(Message(role="user", content=feedback))
            step_boundaries.append(len(messages) - 1)

        summary = EpisodeSummary(total_reward=game.reward)
        summary.signals["outcome"] = RewardSignal(
            name="outcome", value=game.reward * 2 - 1, confidence=1.0,
        )

        state_id = ""
        try:
            state_id = agent_state.state_id().combined_hash
        except Exception:
            pass

        return Episode(
            id=Episode.new_id(), state_id=state_id,
            task_id=f"guess_{target}", bench="guess_number",
            messages=messages, step_boundaries=step_boundaries,
            steps=steps, summary=summary,
            metadata={"target": target, "turns": game.turns, "found": game.reward > 0},
        )

    def _default_prompt(self):
        return (
            "You are playing a number guessing game. Guess an integer between "
            f"0 and {MAX_VAL}. Use binary search: start at the midpoint, then "
            "go higher or lower based on feedback. Reply with just your guess number."
        )

    def _error_episode(self, target, error):
        return Episode(
            id=Episode.new_id(), state_id="", task_id=f"guess_{target}",
            bench="guess_number", messages=[], step_boundaries=[], steps=[],
            summary=EpisodeSummary(filtered=True),
            metadata={"error": error},
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LfX Guess the Number (Tinker-compatible)")
    p.add_argument("--mode", choices=["weight", "harness_learning"], default="harness_learning")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--api-base", default=os.environ.get("LFX_API_BASE", "http://127.0.0.1:8317/v1"))
    p.add_argument("--api-key", default=os.environ.get("LFX_API_KEY", ""))
    p.add_argument("--task-model", default="openai/claude-haiku-4-5-20251001")
    p.add_argument("--reflector-model", default="openai/claude-sonnet-4-5-20250929")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    layers = MODE_LAYERS[args.mode]
    log.info("mode=%s layers=%s", args.mode, layers)

    # 1. Harness
    harness = Harness(system_prompts={
        "guess_number": (
            "You are playing a number guessing game. Use binary search strategy. "
            f"The number is between 0 and {MAX_VAL}. Reply with just your guess."
        ),
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
                "trainer.max_prompt_length": 512,
                "generator.sampling_params.max_generate_length": 64,
                "generator.inference_engine.gpu_memory_utilization": 0.4,
                "trainer.use_sample_packing": False,
            },
            lora_config={"rank": args.lora_rank, "alpha": args.lora_rank * 2.0},
            training_config={"loss_fn": "cross_entropy", "adam_params": {"learning_rate": 3e-5}},
        ))
        weights = Weights(model_ref=args.model, _backend=backend)
    else:
        weights = Weights()

    # 3. Task LLM + environment
    from lfx.llm import LiteLLMClient
    task_client = LiteLLMClient(model=args.task_model, api_key=args.api_key, api_base=args.api_base)
    adapter = GuessNumberAdapter(client=task_client)

    # Tasks = random target numbers
    targets = [random.randint(0, MAX_VAL) for _ in range(50)]

    # 4. Run
    agent_state = AgentState(
        harness=harness, router=Router(), weights=weights,
        inference_url=getattr(backend, "inference_url", None) if backend else None,
    )

    agent_state, state_id = learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=targets,
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
