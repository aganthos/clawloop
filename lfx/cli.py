"""LfX CLI — entry point for run, eval, compare, and gate commands."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from lfx.core.loop import AgentState, learning_loop

log = logging.getLogger("lfx")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lfx",
        description="LfX — Learning from Experience unified learning API",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    run_p = sub.add_parser("run", help="Run the learning loop")
    run_p.add_argument("--bench", required=True, help="Benchmark name")
    run_p.add_argument(
        "--iterations", type=int, default=1, help="Number of learning iterations"
    )
    run_p.add_argument(
        "--episodes", type=int, default=10, help="Episodes per iteration"
    )
    run_p.add_argument("--config", type=str, default=None, help="Config JSON file")
    run_p.add_argument("--model", type=str, default=None, help="LLM model (litellm format)")
    run_p.add_argument("--task-type", type=str, default="base",
                       help="Task type: base, hallucination, disambiguation")
    run_p.add_argument("--task-split", type=str, default="test",
                       help="Data split: train, test")
    run_p.add_argument("--output", type=str, default=None, help="Output directory")
    run_p.add_argument("--seed", type=int, default=None, help="Random seed")

    # -- eval --
    eval_p = sub.add_parser("eval", help="Evaluate current state (no learning)")
    eval_p.add_argument("--bench", required=True, help="Benchmark name")
    eval_p.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes"
    )
    eval_p.add_argument("--config", type=str, default=None, help="Config JSON file")

    # -- compare --
    cmp_p = sub.add_parser(
        "compare", help="Compare two states on a benchmark"
    )
    cmp_p.add_argument("--bench", required=True)
    cmp_p.add_argument("--state-a", required=True, help="First state config JSON")
    cmp_p.add_argument("--state-b", required=True, help="Second state config JSON")
    cmp_p.add_argument("--episodes", type=int, default=10)

    # -- gate --
    gate_p = sub.add_parser("gate", help="Run deploy-time regression gate")
    gate_p.add_argument("--candidate", required=True, help="Candidate episodes JSON")
    gate_p.add_argument(
        "--production", required=True, help="Production episodes JSON"
    )
    gate_p.add_argument("--threshold", type=float, default=0.0)

    return parser


ADAPTER_REGISTRY: dict[str, tuple[str, str]] = {
    "entropic": ("lfx.adapters.entropic", "EntropicAdapter"),
    "car": ("lfx.adapters.car", "CARAdapter"),
    "tau2": ("lfx.adapters.tau2", "Tau2Adapter"),
}


def _get_adapter(bench: str) -> Any:
    """Resolve a benchmark name to its adapter (lazy import)."""
    import importlib

    if bench not in ADAPTER_REGISTRY:
        print(f"Unknown benchmark: {bench}", file=sys.stderr)
        sys.exit(1)

    module_name, class_name = ADAPTER_REGISTRY[bench]
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    return adapter_class()


def _load_config(path: str | None) -> dict[str, Any]:
    """Load a JSON config file, returning {} if no path given."""
    if not path:
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found: {path}", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file {path}: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    # CLI args override config file
    if args.model:
        config["model"] = args.model
    if args.output:
        config["output"] = args.output
    if hasattr(args, "task_type"):
        config["task_type"] = args.task_type
    if hasattr(args, "task_split"):
        config["task_split"] = args.task_split
    if hasattr(args, "seed") and args.seed is not None:
        config["seed"] = args.seed

    adapter = _get_adapter(args.bench)
    adapter.setup(config)

    if args.seed is not None:
        import random
        random.seed(args.seed)

    agent_state = AgentState()
    try:
        tasks = adapter.list_tasks()
    except NotImplementedError:
        tasks = None

    if tasks is not None:
        _, state_id = learning_loop(
            adapter=adapter,
            agent_state=agent_state,
            tasks=tasks,
            n_episodes=args.episodes,
            n_iterations=args.iterations,
        )
    else:
        # Batch-oriented adapter (e.g. CAR) — generate task IDs from config
        task_type = config.get("task_type", "base")
        task_ids = [f"{task_type}_{i}" for i in range(args.episodes)]
        _, state_id = learning_loop(
            adapter=adapter,
            agent_state=agent_state,
            tasks=task_ids,
            n_episodes=args.episodes,
            n_iterations=args.iterations,
        )
    print(f"Final state: {state_id.combined_hash}")


def cmd_eval(args: argparse.Namespace) -> None:
    adapter = _get_adapter(args.bench)
    adapter.setup(_load_config(args.config))
    agent_state = AgentState()
    tasks = adapter.list_tasks()

    episodes = []
    for task in tasks[: args.episodes]:
        ep = adapter.run_episode(task, agent_state)
        episodes.append(ep)

    if episodes:
        avg = sum(e.summary.total_reward for e in episodes) / len(episodes)
        print(f"Evaluated {len(episodes)} episodes — avg reward: {avg:.4f}")
    else:
        print("No episodes collected.")


def cmd_compare(args: argparse.Namespace) -> None:
    print("Compare command: not yet implemented", file=sys.stderr)
    sys.exit(1)


def cmd_gate(args: argparse.Namespace) -> None:
    print("Gate command: not yet implemented", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    handlers = {
        "run": cmd_run,
        "eval": cmd_eval,
        "compare": cmd_compare,
        "gate": cmd_gate,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
