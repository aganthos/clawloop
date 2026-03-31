#!/usr/bin/env python3
"""ClawLoop Demo — Learn to solve math problems via the ClawLoop learning loop.

Run in dry-run mode (no API calls, finishes in seconds):
    python examples/demo_math.py --dry-run

Run with real LLMs (requires ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY):
    ANTHROPIC_API_KEY=... python examples/demo_math.py

Environment variables:
    CLAWLOOP_TASK_MODEL       (default: anthropic/claude-haiku-4-5-20251001)
    CLAWLOOP_REFLECTOR_MODEL  (default: anthropic/claude-sonnet-4-5-20250929)
    CLAWLOOP_API_BASE         (optional — for proxy setups, omit to let litellm route natively)
    CLAWLOOP_API_KEY          (optional — for proxy setups, omit to use provider env vars)
    CLAWLOOP_ITERATIONS       (default: 5)
    CLAWLOOP_EPISODES         (default: 5)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

# ---------------------------------------------------------------------------
# Local dev: allow running from the repo root without pip install
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clawloop.agent import ClawLoopAgent
from clawloop.environments.math import MathEnvironment, _BUILTIN_PROBLEMS
from clawloop.llm import LiteLLMClient, MockLLMClient

log = logging.getLogger("clawloop.demo")

# ---------------------------------------------------------------------------
# Base system prompt for the math domain
# ---------------------------------------------------------------------------
BASE_SYSTEM_PROMPT = (
    "You are a math competition solver. Solve problems step by step and "
    "always present your final answer in \\boxed{} notation.\n"
    "For example: \\boxed{42}"
)


# ---------------------------------------------------------------------------
# Mock clients for --dry-run mode
# ---------------------------------------------------------------------------

# Map question text -> ground-truth answer for lookup by the mock task client
_QUESTION_TO_ANSWER: dict[str, str] = {
    p["question"]: p["answer"] for p in _BUILTIN_PROBLEMS
}

# Questions that the mock "gets wrong" — roughly 40% wrong to start, improving
# over iterations as call_count grows.
_HARD_QUESTIONS = {
    "What is 15 * 13?",
    "Solve for x: x^2 - 5x + 6 = 0. Give the larger root.",
    "How many positive divisors does 60 have?",
    "What is the remainder when 2^10 is divided by 7?",
    "How many ways can you choose 3 items from a set of 5? (i.e., C(5,3))",
    "How many subsets does a set with 4 elements have?",
    "What is the sum of the interior angles (in degrees) of a hexagon?",
    "What is the value of the sum 1 + 2 + 3 + ... + 20?",
}


class MockTaskClient:
    """Mock task LLM that returns boxed answers, some correct and some wrong.

    Simulates gradual improvement: in early iterations more answers are wrong;
    in later iterations more flip to correct.  The improvement is driven by
    the internal call counter — as more calls happen (later iterations), the
    "hard" questions progressively become correct.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def complete(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        self._call_count += 1

        # Extract the question from the user message
        question = ""
        for msg in messages:
            if msg["role"] == "user":
                question = msg["content"]
                break

        gt = _QUESTION_TO_ANSWER.get(question)
        if gt is None:
            # Fallback for unknown questions
            return "I'm not sure. \\boxed{0}"

        is_hard = question in _HARD_QUESTIONS

        # Improvement schedule: hard questions become correct over time.
        # calls 1-10: hard questions wrong; calls 11-15: half flip; 16+: mostly correct
        if is_hard:
            if self._call_count <= 10:
                give_correct = False
            elif self._call_count <= 15:
                give_correct = (self._call_count % 2 == 0)
            else:
                give_correct = True
        else:
            give_correct = True

        if give_correct:
            return (
                f"Let me work through this step by step.\n"
                f"After careful calculation, the answer is \\boxed{{{gt}}}."
            )
        else:
            wrong = str(int(gt) + 1) if gt.isdigit() else "999"
            return f"Hmm, I think the answer is \\boxed{{{wrong}}}."


def _build_mock_reflector_responses() -> list[str]:
    """Build reflector LLM responses that produce progressive insights.

    The reflector is called once per iteration when should_reflect() is True.
    We provide enough responses for the full learning run:
      Iter 1: "Always show intermediate calculation steps"
      Iter 2: "For combinatorics, use the multiplication principle"
      Iter 3: empty (no new insights)
      Iter 4: "Verify by plugging the answer back into the original equation"
      Iter 5: empty
    Plus extras in case of additional reflect calls (paradigm breakthrough, etc.)
    """
    def _insight_json(content: str) -> str:
        return json.dumps([{
            "action": "add",
            "content": content,
            "target_entry_id": None,
            "tags": ["strategy"],
            "source_episode_ids": [],
        }])

    return [
        _insight_json("Always show intermediate calculation steps"),
        _insight_json("For combinatorics, use the multiplication principle"),
        "[]",
        _insight_json("Verify by plugging the answer back into the original equation"),
        "[]",
        # Extra responses for paradigm breakthrough or additional reflect calls
        _insight_json("Estimate the answer magnitude before calculating"),
        _insight_json("Break multi-step problems into smaller sub-problems"),
        _insight_json("Double-check by working the problem in reverse"),
        "[]",
        "[]",
        "[]",
        "[]",
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ClawLoop demo: learn to solve math problems"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use mock LLM clients (no API calls, fast)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of learning iterations",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Override episodes per iteration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="playbook.json",
        help="Path to save the learned playbook (default: playbook.json)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Resolve configuration
    iterations = args.iterations or int(os.environ.get("CLAWLOOP_ITERATIONS", "5"))
    episodes = args.episodes or int(os.environ.get("CLAWLOOP_EPISODES", "5"))

    if args.dry_run:
        log.info("=== DRY-RUN MODE (mock LLM clients, no API calls) ===")
        task_client = MockTaskClient()
        reflector_client = MockLLMClient(responses=_build_mock_reflector_responses())
    else:
        task_model = os.environ.get("CLAWLOOP_TASK_MODEL", "anthropic/claude-haiku-4-5-20251001")
        reflector_model = os.environ.get("CLAWLOOP_REFLECTOR_MODEL", "anthropic/claude-sonnet-4-5-20250929")
        api_base = os.environ.get("CLAWLOOP_API_BASE", "")
        api_key = os.environ.get("CLAWLOOP_API_KEY", "")

        log.info("=== REAL LLM MODE ===")
        log.info("  Task model:      %s", task_model)
        log.info("  Reflector model: %s", reflector_model)
        if api_base:
            log.info("  API base:        %s", api_base)

        task_client = LiteLLMClient(
            model=task_model, api_key=api_key or None, api_base=api_base or None,
            temperature=0.7, max_tokens=1024,
        )
        reflector_client = LiteLLMClient(
            model=reflector_model, api_key=api_key or None, api_base=api_base or None,
            temperature=0.7, max_tokens=2000,
        )

    log.info("  Iterations:      %d", iterations)
    log.info("  Episodes/iter:   %d", episodes)

    # -- Step 1: Create the math environment --
    env = MathEnvironment()
    n_tasks = len(env.get_tasks())
    log.info("MathEnvironment loaded with %d problems", n_tasks)

    # -- Step 2: Create the ClawLoopAgent --
    agent = ClawLoopAgent(
        task_client=task_client,
        reflector_client=reflector_client,
        bench="math",
        base_system_prompt=BASE_SYSTEM_PROMPT,
    )

    # -- Step 3: Run the learning loop --
    print("\n" + "=" * 60)
    print("  ClawLoop Learning Loop -- Math Demo")
    print("=" * 60 + "\n")

    results = agent.learn(env, iterations=iterations, episodes_per_iter=episodes)

    # -- Step 4: Print results --
    print("\n" + "-" * 60)
    print("  RESULTS")
    print("-" * 60)

    # Reward curve
    print("\nReward curve per iteration:")
    for i, reward in enumerate(results["rewards"]):
        bar = "#" * int(reward * 40)
        print(f"  Iter {i + 1}: {reward:.4f}  {bar}")

    # Playbook entries
    n_entries = results["n_entries"]
    print(f"\nPlaybook entries learned: {n_entries}")

    # Full system prompt
    final_prompt = agent.get_system_prompt()
    print("\nFinal system prompt:")
    print("-" * 40)
    print(final_prompt)
    print("-" * 40)

    # Save playbook
    agent.save_playbook(args.output)
    print(f"\nPlaybook saved to: {args.output}")

    # Show playbook JSON
    with open(args.output) as f:
        playbook_data = json.load(f)
    print("\nPlaybook JSON:")
    print(json.dumps(playbook_data, indent=2))

    print("\n" + "=" * 60)
    print("  Demo complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
