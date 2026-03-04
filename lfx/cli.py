"""LfX CLI — entry point for run, eval, compare, and gate commands."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from lfx.core.gate import gate_for_deploy
from lfx.core.loop import AgentState, learning_loop
from lfx.core.state import StateID

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

    # -- eval --
    eval_p = sub.add_parser("eval", help="Evaluate current state (no learning)")
    eval_p.add_argument("--bench", required=True, help="Benchmark name")
    eval_p.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes"
    )

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


def _get_adapter(bench: str) -> Any:
    """Resolve a benchmark name to its adapter (lazy import)."""
    if bench == "entropic":
        from lfx.adapters.entropic import EntropicAdapter
        return EntropicAdapter()
    elif bench == "car":
        from lfx.adapters.car import CARAdapter
        return CARAdapter()
    elif bench == "tau2":
        from lfx.adapters.tau2 import Tau2Adapter
        return Tau2Adapter()
    else:
        print(f"Unknown benchmark: {bench}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    adapter = _get_adapter(args.bench)
    config: dict[str, Any] = {}
    if args.config:
        with open(args.config) as f:
            config = json.load(f)
    adapter.setup(config)

    agent_state = AgentState()
    tasks = adapter.list_tasks()

    result_state, state_id = learning_loop(
        adapter=adapter,
        agent_state=agent_state,
        tasks=tasks,
        n_episodes=args.episodes,
        n_iterations=args.iterations,
    )
    print(f"Final state: {state_id.combined_hash}")


def cmd_eval(args: argparse.Namespace) -> None:
    adapter = _get_adapter(args.bench)
    adapter.setup({})
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
