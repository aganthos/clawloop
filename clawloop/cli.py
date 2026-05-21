"""ClawLoop CLI — entry points for demo and benchmark setup commands.

`clawloop run <config.json>` is a thin wrapper over the unified ``TrainConfig``
runner: load JSON, validate via Pydantic, dispatch to ``train()``. The
``--dry-run`` flag swaps real LLM clients for mocks so smoke tests work
without API keys.

`clawloop eval` is still disabled; legacy invocations get a truthful redirect
to ``clawloop run`` and ``clawloop demo math --dry-run``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("clawloop")

_EVAL_DISABLED_MSG = (
    "`clawloop eval` is disabled. Use one of:\n"
    "  - Real benchmark:  uv run clawloop run examples/configs/math_harness.json\n"
    "  - Other configs:   examples/configs/  (math, harbor, entropic, openclaw, taubench)\n"
    "  - No-key demo:     uv run clawloop demo math --dry-run"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawloop",
        description="ClawLoop — Learning from Experience unified learning API",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run a TrainConfig JSON via train()")
    run_p.add_argument("config", type=Path, help="Path to TrainConfig JSON")
    run_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Swap real LLM clients for mocks (no API calls)",
    )

    # Eval stays disabled. add_help=False so `eval --help` hits the redirect.
    sub.add_parser("eval", help="(disabled) use `clawloop run` instead", add_help=False)

    setup_p = sub.add_parser("setup-bench", help="Install benchmark dependencies")
    setup_p.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    setup_p.add_argument("--bench", required=True, help="Benchmark name")

    demo_p = sub.add_parser("demo", help="Run built-in demos")
    demo_sub = demo_p.add_subparsers(dest="demo_name", required=True)

    math_p = demo_sub.add_parser("math", help="Math learning loop demo")
    math_p.add_argument("--dry-run", action="store_true", help="Use mock LLMs (no API calls)")
    math_p.add_argument(
        "--iterations", type=int, default=None, help="Number of learning iterations"
    )
    math_p.add_argument("--episodes", type=int, default=None, help="Episodes per iteration")
    math_p.add_argument("--output", type=str, default="playbook.json", help="Playbook output path")

    return parser


BENCH_SETUP: dict[str, dict[str, Any]] = {
    "car": {
        "bench_dir": "benchmarks/a2a/car-bench",
        "data_setup": "scenarios/car-bench/setup.sh",
        "uv_sync_cmd": [
            "uv",
            "sync",
            "--extra",
            "car-bench-agent",
            "--extra",
            "car-bench-evaluator",
        ],
    },
    "entropic": {
        "bench_dir": "benchmarks/a2a/entropic-crmarenapro",
        "data_setup": None,
        "uv_sync_cmd": ["uv", "sync"],
    },
}


def cmd_setup_bench(args: argparse.Namespace) -> None:
    """Install benchmark external dependencies."""
    import subprocess

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

    data_setup = setup.get("data_setup")
    if data_setup:
        script = bench_dir / data_setup
        if script.exists():
            print(f"Running data setup: {script}")
            subprocess.run(["bash", str(script)], check=True)

    uv_cmd = setup.get("uv_sync_cmd")
    if uv_cmd:
        print(f"Installing dependencies in {bench_dir}...")
        subprocess.run(uv_cmd, cwd=str(bench_dir), check=True)

    print(f"Syncing clawloop extras: --extra {bench}")
    subprocess.run(["uv", "sync", "--extra", bench, "--extra", "dev"], check=True)

    print(f"Setup complete for {bench}")


def cmd_demo(args: argparse.Namespace) -> None:
    """Dispatch to the requested built-in demo."""
    if args.demo_name == "math":
        from clawloop.demo_math import main as demo_math_main

        argv: list[str] = []
        if getattr(args, "dry_run", False):
            argv.append("--dry-run")
        if getattr(args, "iterations", None) is not None:
            argv += ["--iterations", str(args.iterations)]
        if getattr(args, "episodes", None) is not None:
            argv += ["--episodes", str(args.episodes)]
        if getattr(args, "output", None):
            argv += ["--output", args.output]
        demo_math_main(argv)
    else:
        print(f"Unknown demo: {args.demo_name}", file=sys.stderr)
        sys.exit(1)


def cmd_run(args: argparse.Namespace) -> None:
    """Load a TrainConfig JSON and dispatch to train()."""
    from clawloop.train import MODE_LAYERS, TrainConfig, train

    raw = json.loads(args.config.read_text())
    config = TrainConfig(**raw)  # Pydantic ValidationError surfaces fail-fast

    log.info(
        "mode=%s env=%s layers=%s",
        config.mode,
        config.env_type,
        MODE_LAYERS[config.mode],
    )

    if args.dry_run:
        _install_dry_run_clients(config)

    train(config)


def _install_dry_run_clients(config: "Any") -> None:
    """Patch ``clawloop.train._make_llm_client`` to return mock clients.

    Identifies the role (reflector / task / other) by matching the cfg
    object identity against ``config.llm_clients``. Falls back to a generic
    ``MockLLMClient`` for any unknown role so unfamiliar envs still run.
    """
    import clawloop.train as _train
    from clawloop.demo_math import MockTaskClient, _build_mock_reflector_responses
    from clawloop.llm import MockLLMClient

    role_by_id = {id(v): k for k, v in config.llm_clients.items()}
    original = _train._make_llm_client

    def _mock_make(cfg):
        role = role_by_id.get(id(cfg))
        if role == "reflector":
            return MockLLMClient(responses=_build_mock_reflector_responses())
        if role == "task":
            return MockTaskClient()
        return MockLLMClient(responses=["[]"])

    _train._make_llm_client = _mock_make
    log.info("dry-run: LLM clients patched to mocks (original=%r)", original.__name__)


def main() -> None:
    parser = _build_parser()
    # parse_known_args lets the disabled `eval` subcommand swallow legacy flags
    # (`clawloop eval --bench entropic`) and fall through to the redirect.
    args, _unknown = parser.parse_known_args()

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "eval":
        print(_EVAL_DISABLED_MSG, file=sys.stderr)
        sys.exit(2)

    # For active subcommands, re-parse strictly so typos still error.
    args = parser.parse_args()
    handlers = {
        "run": cmd_run,
        "setup-bench": cmd_setup_bench,
        "demo": cmd_demo,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
