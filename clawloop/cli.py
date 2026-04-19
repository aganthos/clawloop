"""ClawLoop CLI — entry points for demo and benchmark setup commands.

The legacy ``run`` and ``eval`` subcommands are disabled: they only wired a
subset of environments and drifted from the unified ``TrainConfig`` runner.
They remain in the parser so stale muscle memory gets a truthful redirect
instead of a misleading ``Unknown benchmark`` failure.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

log = logging.getLogger("clawloop")

_DISABLED_MSG = (
    "`clawloop {cmd}` is temporarily disabled. Use one of:\n"
    "  - Real benchmark:  uv run python examples/train_runner.py \\\n"
    "                         examples/configs/entropic_harness.json\n"
    "  - Other configs:   examples/configs/  (math, harbor, entropic, openclaw, taubench)\n"
    "  - No-key demo:     uv run clawloop demo math --dry-run\n"
    "The config-driven runner covers every supported environment; "
    "reintroduction of `{cmd}` as a thin wrapper is tracked upstream."
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawloop",
        description="ClawLoop — Learning from Experience unified learning API",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    # Disabled subcommands. add_help=False so `run --help` hits the redirect
    # rather than argparse's auto-generated help output. Any legacy flags land
    # in `unknown` via parse_known_args() in main() and are ignored.
    sub.add_parser("run", help="(disabled) use examples/train_runner.py", add_help=False)
    sub.add_parser("eval", help="(disabled) use examples/train_runner.py", add_help=False)

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


def main() -> None:
    parser = _build_parser()
    # Use parse_known_args so disabled subcommands can ignore legacy flags
    # (`clawloop run --bench entropic`) and fall through to the redirect.
    args, _unknown = parser.parse_known_args()

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command in {"run", "eval"}:
        print(_DISABLED_MSG.format(cmd=args.command), file=sys.stderr)
        sys.exit(2)

    # For active subcommands, re-parse strictly so typos still error.
    args = parser.parse_args()
    handlers = {
        "setup-bench": cmd_setup_bench,
        "demo": cmd_demo,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
