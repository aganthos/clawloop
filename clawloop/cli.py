"""ClawLoop CLI — entry point for run, eval, compare, and gate commands."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from typing import Any

from clawloop.core.loop import AgentState, learning_loop

log = logging.getLogger("clawloop")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawloop",
        description="ClawLoop — Learning from Experience unified learning API",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # -- run --
    run_p = sub.add_parser("run", help="Run the learning loop")
    run_p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    run_p.add_argument("--bench", required=True, help="Benchmark name")
    run_p.add_argument(
        "--iterations", type=int, default=1, help="Number of learning iterations"
    )
    run_p.add_argument(
        "--episodes", type=int, default=10, help="Episodes per iteration"
    )
    run_p.add_argument("--config", type=str, default=None, help="Config JSON file")
    run_p.add_argument("--model", type=str, default=None, help="LLM model (litellm format)")
    run_p.add_argument("--api-base", type=str, default=None, help="LLM API base URL (e.g. CLIProxyAPI)")
    run_p.add_argument("--task-type", type=str, default="base",
                       help="Task type: base, hallucination, disambiguation")
    run_p.add_argument("--task-split", type=str, default="test",
                       help="Data split: train, test")
    run_p.add_argument("--output", type=str, default=None, help="Output directory")
    run_p.add_argument("--seed", type=int, default=None, help="Random seed")

    # -- eval --
    eval_p = sub.add_parser("eval", help="Evaluate current state (no learning)")
    eval_p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
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

    # -- setup-bench --
    setup_p = sub.add_parser("setup-bench", help="Install benchmark dependencies")
    setup_p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    setup_p.add_argument("--bench", required=True, help="Benchmark name")

    return parser


ADAPTER_REGISTRY: dict[str, tuple[str, str]] = {
    "entropic": ("clawloop.adapters.entropic", "EntropicAdapter"),
    "car": ("clawloop.adapters.car", "CARAdapter"),
    "tau2": ("clawloop.adapters.tau2", "Tau2Adapter"),
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


def _build_reflector(config: dict[str, Any]) -> Any | None:
    """Create a Reflector with LiteLLMClient if api_base is configured."""
    api_base = config.get("api_base")
    if not api_base:
        log.warning("No api_base in config — Reflector disabled (no learning)")
        return None

    from clawloop.core.reflector import Reflector, ReflectorConfig
    from clawloop.llm import LiteLLMClient

    model = config.get("reflector_model", config.get("model", "anthropic/claude-haiku-4-5-20251001"))
    client = LiteLLMClient(
        model=model,
        api_base=api_base,
        api_key=config.get("api_key"),
    )
    log.info("Reflector enabled: model=%s via %s", model, api_base)
    rbs = config.get("reflection_batch_size", 1)
    return Reflector(client=client, config=ReflectorConfig(reflection_batch_size=rbs))


def _ensure_output_dir(config: dict[str, Any], bench: str) -> None:
    """Set output dir if not configured. Convention: runs/<bench>/<timestamp>."""
    import time
    if "output" not in config or not config["output"]:
        config["output"] = f"./runs/{bench}/{int(time.time())}"


def cmd_run(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    # CLI args override config file
    if args.model:
        config["model"] = args.model
    if getattr(args, "api_base", None):
        config["api_base"] = args.api_base
    if args.output:
        config["output"] = args.output
    if hasattr(args, "task_type"):
        config["task_type"] = args.task_type
    if hasattr(args, "task_split"):
        config["task_split"] = args.task_split
    if hasattr(args, "seed") and args.seed is not None:
        config["seed"] = args.seed

    _ensure_output_dir(config, args.bench)

    adapter = _get_adapter(args.bench)
    adapter.setup(config)

    if args.seed is not None:
        random.seed(args.seed)

    # Wire Reflector into harness for ICL learning
    reflector = _build_reflector(config)
    agent_state = AgentState()
    agent_state.harness.reflector = reflector

    try:
        tasks = adapter.list_tasks()
    except NotImplementedError:
        tasks = None

    output_dir = config.get("output")

    if tasks is not None:
        _, state_id = learning_loop(
            adapter=adapter,
            agent_state=agent_state,
            tasks=tasks,
            n_episodes=args.episodes,
            n_iterations=args.iterations,
            output_dir=output_dir,
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
            output_dir=output_dir,
        )
    print(f"Final state: {state_id.combined_hash}")


def cmd_eval(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    adapter = _get_adapter(args.bench)
    adapter.setup(config)
    agent_state = AgentState()

    try:
        tasks = adapter.list_tasks()
        selected = tasks[: args.episodes]
    except NotImplementedError:
        task_type = config.get("task_type", "base")
        selected = [f"{task_type}_{i}" for i in range(args.episodes)]

    episodes = []
    for task in selected:
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


# Benchmark setup registry: bench -> (setup_script_path, uv_sync_extras)
BENCH_SETUP: dict[str, dict[str, Any]] = {
    "car": {
        "bench_dir": "benchmarks/a2a/car-bench",
        "data_setup": "scenarios/car-bench/setup.sh",
        "uv_sync_cmd": ["uv", "sync", "--extra", "car-bench-agent", "--extra", "car-bench-evaluator"],
    },
    "entropic": {
        "bench_dir": "benchmarks/a2a/entropic-crmarenapro",
        "data_setup": None,
        "uv_sync_cmd": ["uv", "sync"],
    },
    # "tau2": {
    #     "bench_dir": "benchmarks/tau-bench",
    #     "data_setup": None,
    #     "uv_sync_cmd": ["uv", "sync"],
    # },
}


def cmd_setup_bench(args: argparse.Namespace) -> None:
    """Install benchmark external dependencies."""
    import subprocess
    from pathlib import Path

    bench = args.bench
    if bench not in BENCH_SETUP:
        print(f"No setup defined for benchmark: {bench}", file=sys.stderr)
        print(f"Available: {', '.join(BENCH_SETUP.keys())}", file=sys.stderr)
        sys.exit(1)

    setup = BENCH_SETUP[bench]
    bench_dir = Path(setup["bench_dir"])

    if not bench_dir.exists():
        print(f"Benchmark dir not found: {bench_dir}", file=sys.stderr)
        sys.exit(1)

    # Run data setup script if defined
    data_setup = setup.get("data_setup")
    if data_setup:
        script = bench_dir / data_setup
        if script.exists():
            print(f"Running data setup: {script}")
            subprocess.run(["bash", str(script)], check=True)

    # Install dependencies via uv
    uv_cmd = setup.get("uv_sync_cmd")
    if uv_cmd:
        print(f"Installing dependencies in {bench_dir}...")
        subprocess.run(uv_cmd, cwd=str(bench_dir), check=True)

    # Also sync clawloop extras for this bench
    print(f"Syncing clawloop extras: --extra {bench}")
    subprocess.run(["uv", "sync", "--extra", bench, "--extra", "dev"], check=True)

    print(f"Setup complete for {bench}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    handlers = {
        "run": cmd_run,
        "eval": cmd_eval,
        "compare": cmd_compare,
        "gate": cmd_gate,
        "setup-bench": cmd_setup_bench,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
