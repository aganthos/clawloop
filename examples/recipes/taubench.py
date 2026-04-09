#!/usr/bin/env python3
"""ClawLoop recipe: tau-bench 3 harness learning.

Runs harness learning on tau-bench retail or airline tasks. The ClawLoop
Harness layer evolves the agent system prompt based on per-episode reward
signals from tau-bench's domain-specific verifier.

Requirements:
    pip install "clawloop[taubench]"
    export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY, etc.

Usage (retail, 3 tasks, 3 iterations — quick smoke test):
    python examples/recipes/taubench.py \\
        --domain retail \\
        --task-ids retail_0 retail_1 retail_2 \\
        --iterations 3

Usage (airline, 5 tasks, 5 iterations):
    python examples/recipes/taubench.py \\
        --domain airline \\
        --task-ids airline_0 airline_1 airline_2 airline_3 airline_4 \\
        --iterations 5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger("clawloop.recipe.taubench")

_RETAIL_SYSTEM_PROMPT = (
    "You are a retail customer service agent. Help users with orders, returns, "
    "and product questions. Follow the policy exactly. Be concise and accurate."
)

_AIRLINE_SYSTEM_PROMPT = (
    "You are an airline customer service agent. Help users with flight bookings, "
    "cancellations, and seat upgrades. Follow the policy exactly. Be concise and accurate."
)

_DOMAIN_PROMPTS: dict[str, str] = {
    "retail": _RETAIL_SYSTEM_PROMPT,
    "airline": _AIRLINE_SYSTEM_PROMPT,
}


def run_harness_learning(args: argparse.Namespace) -> None:
    from clawloop.environments.taubench import TauBenchAdapter
    from clawloop.core.intensity import AdaptiveIntensity
    from clawloop.core.loop import AgentState, learning_loop
    from clawloop.learning_layers.harness import Harness
    from clawloop.learning_layers.router import Router
    from clawloop.learning_layers.weights import Weights

    from examples.recipes.common import build_local_evolver

    starter_prompt = _DOMAIN_PROMPTS.get(args.domain, _RETAIL_SYSTEM_PROMPT)

    harness = Harness(
        system_prompts={"taubench": starter_prompt},
        evolver=build_local_evolver(
            reflector_model=args.reflector_model,
            api_key=args.api_key,
            api_base=args.api_base,
            reflection_batch_size=args.reflection_batch_size,
        ),
    )

    adapter = TauBenchAdapter()
    adapter.setup(
        {
            "domain": args.domain,
            "llm_agent": args.task_model,
            "llm_user": args.task_model,
            "max_steps": args.max_steps,
            "max_concurrency": args.max_concurrency,
            "task_split": args.task_split,
        }
    )

    # Resolve task IDs: explicit list or auto-discover from split
    if args.task_ids:
        tasks = args.task_ids
    else:
        tasks = adapter.list_tasks(args.task_split)[: args.num_tasks]
        log.info("Auto-discovered %d tasks from split %r", len(tasks), args.task_split)

    agent_state = AgentState(harness=harness, router=Router(), weights=Weights())

    log.info(
        "Starting harness learning: domain=%s tasks=%d iterations=%d",
        args.domain, len(tasks), args.iterations,
    )

    agent_state, state_id = learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=tasks,
        n_episodes=len(tasks),
        n_iterations=args.iterations,
        active_layers=["harness", "router"],
        intensity=AdaptiveIntensity(reflect_every_n=args.reflect_every),
    )

    print(f"\nDone. State: {state_id.combined_hash[:12]}")
    print(f"Domain: {args.domain} | Iterations: {args.iterations} | Tasks: {len(tasks)}")

    if harness.playbook and harness.playbook.entries:
        print(f"\nPlaybook entries after learning: {len(harness.playbook.entries)}")
        for entry in harness.playbook.entries[:5]:
            print(f"  - {entry.content[:100]}")
    else:
        print("\nNo playbook entries yet (may need more iterations or failures to trigger learning).")

    print(f"\nFinal system prompt (first 300 chars):")
    print(harness.system_prompt("taubench")[:300])


def main() -> None:
    p = argparse.ArgumentParser(description="ClawLoop tau-bench 3 harness learning recipe")
    p.add_argument("--domain", choices=["retail", "airline"], default="retail",
                   help="tau-bench domain to run")
    p.add_argument("--task-ids", nargs="*", default=None,
                   help="Explicit task IDs to run (e.g. retail_0 retail_1). "
                        "If omitted, auto-discovers from --task-split up to --num-tasks.")
    p.add_argument("--num-tasks", type=int, default=5,
                   help="Number of tasks to auto-discover when --task-ids is not set")
    p.add_argument("--task-split", default="test",
                   help="tau-bench split to use (test, dev, train)")
    p.add_argument("--iterations", type=int, default=3,
                   help="Number of harness learning iterations")
    p.add_argument("--max-steps", type=int, default=30,
                   help="Max conversation steps per episode")
    p.add_argument("--max-concurrency", type=int, default=4,
                   help="Max parallel episodes per batch")
    p.add_argument("--task-model", default="gemini/gemini-2.0-flash-lite",
                   help="Model for agent and user simulator (default: gemini-2.0-flash-lite)")
    p.add_argument("--reflector-model", default="gemini/gemini-2.5-flash-lite",
                   help="Model for the ClawLoop reflector — runs once per iteration "
                        "(default: gemini-2.5-flash-lite)")
    p.add_argument("--reflect-every", type=int, default=1,
                   help="Reflect every N iterations (1=every iteration, 3=default adaptive)")
    p.add_argument("--reflection-batch-size", type=int, default=4,
                   help="Episodes per Reflector LLM call — higher enables contrastive learning")
    p.add_argument("--api-base", default=os.environ.get("CLAWLOOP_API_BASE"),
                   help="API base URL override (e.g. for local proxy)")
    p.add_argument("--api-key", default=os.environ.get("CLAWLOOP_API_KEY", ""),
                   help="API key override")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log.info("domain=%s model=%s iterations=%d", args.domain, args.task_model, args.iterations)

    run_harness_learning(args)


if __name__ == "__main__":
    main()
