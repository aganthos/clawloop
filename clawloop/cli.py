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

from clawloop.train import LLMClientConfig

log = logging.getLogger("clawloop")


class _DryRunLLMClientConfig(LLMClientConfig):
    """Private subclass that tags an LLM client config with its dry-run role.

    Used only by ``cmd_run --dry-run``: ``_install_dry_run_clients`` swaps
    each entry in ``config.llm_clients`` for an instance of this class, and
    the patched ``_make_llm_client`` picks the right mock via ``isinstance``.

    Why a subclass instead of a field on ``LLMClientConfig``:
      - The public ``LLMClientConfig`` schema stays free of testing
        vocabulary — no ``dry_run_role: null`` in JSON dumps or generated
        JSON Schema.
      - ``model`` is left untouched, so downstream code that reads it
        verbatim (e.g. ``_build_entropic`` propagating ``tc.model`` into
        ``entropic_cfg``) is unaffected.
      - Pydantic ``model_copy()`` preserves the runtime class, so the tag
        survives copies just like a field would.
    """

    dry_run_role: str


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


def _install_dry_run_clients(config: Any) -> None:
    """Wire `--dry-run`: guarantee no real LLM calls regardless of env_type.

    Two parts:
      1. Replace each ``LLMClientConfig`` in ``config.llm_clients`` with a
         ``_DryRunLLMClientConfig`` instance carrying the role, and patch
         ``clawloop.train._make_llm_client`` to switch on ``isinstance``.
         The private subclass keeps the public schema clean and survives
         ``model_copy()``.
      2. For envs whose adapter bypasses ``_make_llm_client`` (per
         ``train.ENVS_USING_MAKE_LLM_CLIENT``), swap the registered builder
         with a stub that returns a no-I/O ``_StubAdapter``.
    """
    import clawloop.train as _train
    from clawloop.demo_math import MockTaskClient, _build_mock_reflector_responses
    from clawloop.llm import MockLLMClient

    # Part 1: replace each cfg with a subclass instance carrying the role,
    # then route _make_llm_client through a mock factory that reads the role.
    for role, cfg in list(config.llm_clients.items()):
        config.llm_clients[role] = _DryRunLLMClientConfig(**cfg.model_dump(), dry_run_role=role)

    original_make = _train._make_llm_client

    def _mock_make(cfg):
        if isinstance(cfg, _DryRunLLMClientConfig):
            role = cfg.dry_run_role
            if role == "reflector":
                return MockLLMClient(responses=_build_mock_reflector_responses())
            if role == "task":
                return MockTaskClient()
            return MockLLMClient(responses=["[]"])
        return original_make(cfg)

    _train._make_llm_client = _mock_make

    # Part 2: for env_types that bypass _make_llm_client, replace the
    # registered builder with one that returns a stub adapter. Without this,
    # --dry-run on (e.g.) taubench / entropic / openclaw would still hit
    # real endpoints.
    env_type = config.env_type
    uses_make_llm_client = env_type in _train.ENVS_USING_MAKE_LLM_CLIENT
    if not uses_make_llm_client and env_type in _train.ENV_BUILDERS:
        # Floor at 1 to keep the task list non-empty even if a config sets
        # episodes_per_iter to 0; the learning loop samples from it.
        n_tasks = max(1, config.episodes_per_iter)
        stub_tasks = [f"dry_run_{env_type}_{i}" for i in range(n_tasks)]

        def _stub_builder(_cfg: Any, _clients: Any) -> tuple[Any, list[str]]:
            return _StubAdapter(env_type), list(stub_tasks)

        _train.ENV_BUILDERS[env_type] = _stub_builder

    log.info(
        "dry-run: LLM clients mocked; env=%r %s",
        env_type,
        "uses _make_llm_client" if uses_make_llm_client else "stubbed",
    )


class _StubAdapter:
    """Adapter that yields canned episodes — no network, no LLM calls.

    Used by --dry-run for env_types whose real adapter would otherwise
    drive external services (tau2, CRMArena, OpenClaw, OpenSpiel).
    """

    def __init__(self, env_type: str) -> None:
        self._env_type = env_type

    def run_episode(self, task: Any, agent_state: Any) -> Any:
        from uuid import uuid4

        from clawloop.core.episode import Episode, EpisodeSummary, StepMeta

        state_id = ""
        try:
            state_id = agent_state.state_id().combined_hash
        except (AttributeError, TypeError):
            # AttributeError: agent_state has no `state_id` or the result
            # lacks `combined_hash`. TypeError: `state_id` is not callable.
            # Any other exception is a real bug and should propagate.
            pass

        return Episode(
            id=uuid4().hex,
            state_id=state_id,
            task_id=f"{self._env_type}:{task}",
            bench=self._env_type,
            messages=[],
            step_boundaries=[],
            steps=[StepMeta(t=0, reward=1.0, done=True, timing_ms=0.0)],
            summary=EpisodeSummary(total_reward=1.0),
            metadata={"dry_run": True},
        )

    def run_batch(self, agent_state: Any, task_ids: list[Any]) -> list[Any]:
        return [self.run_episode(t, agent_state) for t in task_ids]

    def get_traces(self, episode: Any) -> dict[str, Any]:
        return {"bench": self._env_type, "episode_id": episode.id, "dry_run": True}


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
