#!/usr/bin/env python3
"""Harness tuning example: learn better system prompts for math via the reflector.

No GPU needed — this tunes the prompt/playbook layer only.

    # Dry run (no API calls):
    python examples/harness_tuning/math_prompts.py --dry-run

    # With CLIProxyAPI:
    python examples/harness_tuning/math_prompts.py

    # With direct API key:
    ANTHROPIC_API_KEY=sk-... python examples/harness_tuning/math_prompts.py
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lfx.agent import LfXAgent
from lfx.envs.math import MathEnvironment
from lfx.llm import LiteLLMClient, MockLLMClient

log = logging.getLogger("lfx.example.harness")

BASE_PROMPT = (
    "You are a math competition solver. Solve problems step by step and "
    "always present your final answer in \\boxed{} notation."
)


def _mock_reflector_responses():
    def _j(c):
        return json.dumps([{"action": "add", "content": c, "target_entry_id": None,
                            "tags": ["strategy"], "source_episode_ids": []}])
    return [
        _j("Always show intermediate calculation steps"),
        _j("For combinatorics, use the multiplication principle"),
        "[]",
        _j("Verify by plugging the answer back into the original equation"),
        "[]", "[]", "[]",
    ]


class _MockTask:
    """Mock task client that simulates improvement over time."""
    def __init__(self):
        self._n = 0

    def complete(self, messages, **kw):
        self._n += 1
        q = next((m["content"] for m in messages if m["role"] == "user"), "")
        # Simple mock: progressively better answers
        if self._n > 10:
            return f"Step by step: The answer is \\boxed{{42}}."
        return f"I think it's \\boxed{{0}}."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="playbook_harness.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if args.dry_run:
        task_client = _MockTask()
        reflector_client = MockLLMClient(responses=_mock_reflector_responses())
    else:
        api_base = os.environ.get("LFX_API_BASE", "http://127.0.0.1:8317/v1")
        api_key = os.environ.get("LFX_API_KEY", "kuhhandel-bench-key")
        task_client = LiteLLMClient(
            model="openai/claude-haiku-4-5-20251001",
            api_key=api_key, api_base=api_base,
        )
        reflector_client = LiteLLMClient(
            model="openai/claude-sonnet-4-5-20250929",
            api_key=api_key, api_base=api_base,
        )

    env = MathEnvironment()
    agent = LfXAgent(
        task_client=task_client,
        reflector_client=reflector_client,
        bench="math",
        base_system_prompt=BASE_PROMPT,
    )

    results = agent.learn(env, iterations=args.iterations, episodes_per_iter=args.episodes)

    print(f"\nReward curve: {[f'{r:.3f}' for r in results['rewards']]}")
    print(f"Playbook entries: {results['n_entries']}")
    print(f"\nFinal prompt:\n{agent.get_system_prompt()}")

    agent.save_playbook(args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
