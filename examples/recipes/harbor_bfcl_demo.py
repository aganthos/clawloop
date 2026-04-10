#!/usr/bin/env python3
"""ClawLoop demo: Harbor BFCL harness learning with train/eval split.

Downloads BFCL tasks, splits into train/eval, runs harness learning with
eval after each iteration. Logs everything for the live viewer.

Usage:
    python examples/recipes/harbor_bfcl_demo.py [--output-dir runs/bfcl-demo]

Opens the live viewer automatically in your browser.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

log = logging.getLogger("clawloop.demo.bfcl")

# ---------------------------------------------------------------------------
# BFCL task specs — mix of difficulties
# ---------------------------------------------------------------------------

# Real BFCL task names from Harbor registry
# 5 live-simple (easy baseline — flash lite gets most right)
BFCL_SIMPLE = [
    "bfcl-live-simple-0-0-0", "bfcl-live-simple-1-1-0", "bfcl-live-simple-10-3-6",
    "bfcl-live-simple-100-59-1", "bfcl-live-simple-101-60-0",
]

# 10 live-multiple (harder — multiple function calls needed)
BFCL_MULTIPLE = [
    "bfcl-live-multiple-0-0-0", "bfcl-live-multiple-1-0-1", "bfcl-live-multiple-10-4-2",
    "bfcl-live-multiple-100-42-4", "bfcl-live-multiple-1000-231-0",
    "bfcl-live-multiple-101-42-5", "bfcl-live-multiple-102-43-0",
    "bfcl-live-multiple-103-43-1", "bfcl-live-multiple-104-43-2",
    "bfcl-live-multiple-105-43-3",
]

# 5 live-parallel (parallel calls — even harder)
BFCL_PARALLEL = [
    "bfcl-live-parallel-0-0-0", "bfcl-live-parallel-1-0-1", "bfcl-live-parallel-10-6-0",
    "bfcl-live-parallel-11-7-0", "bfcl-live-parallel-12-8-0",
]

ALL_TASK_NAMES = BFCL_SIMPLE + BFCL_MULTIPLE + BFCL_PARALLEL
BFCL_GIT_URL = "https://github.com/laude-institute/harbor-datasets.git"
BFCL_GIT_COMMIT = "6bedd7878dc5d6f3456b4d80b781eb3c2d84f262"


# ---------------------------------------------------------------------------
# Download tasks
# ---------------------------------------------------------------------------

def download_tasks(output_dir: Path) -> list[Path]:
    """Download BFCL tasks via Harbor's TaskClient."""
    from harbor.models.task.id import GitTaskId
    from harbor.tasks.client import TaskClient

    download_dir = output_dir / "tasks"
    download_dir.mkdir(parents=True, exist_ok=True)

    async def _download():
        client = TaskClient()
        task_ids = [
            GitTaskId(
                git_url=BFCL_GIT_URL,
                git_commit_id=BFCL_GIT_COMMIT,
                path=Path(f"datasets/bfcl/{name}"),
            )
            for name in ALL_TASK_NAMES
        ]
        result = await client.download_tasks(task_ids=task_ids, output_dir=download_dir)
        return result.paths

    log.info("Downloading %d BFCL tasks...", len(ALL_TASK_NAMES))
    paths = asyncio.run(_download())
    log.info("Downloaded %d tasks to %s", len(paths), download_dir)
    return paths


# ---------------------------------------------------------------------------
# Eval logger
# ---------------------------------------------------------------------------

class EvalLog:
    """Logs eval results to eval.jsonl for the viewer."""

    def __init__(self, output_dir: Path):
        self._path = output_dir / "eval.jsonl"

    def log_eval(self, iteration: int, episodes: list, playbook_size: int):
        rewards = [ep.summary.total_reward for ep in episodes]
        per_task = {}
        for ep in episodes:
            per_task[ep.task_id] = {
                "reward": ep.summary.total_reward,
                "filtered": ep.summary.filtered,
                "n_messages": len(ep.messages),
            }
        entry = {
            "iteration": iteration,
            "timestamp": time.time(),
            "n_episodes": len(episodes),
            "avg_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "per_task": per_task,
            "playbook_size": playbook_size,
        }
        with open(self._path, "a") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()
        log.info(
            "  [eval] iter=%d avg=%.4f min=%.4f max=%.4f",
            iteration, entry["avg_reward"], entry["min_reward"], entry["max_reward"],
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="ClawLoop BFCL Demo — live harness learning")
    p.add_argument("--output-dir", default="runs/bfcl-demo", help="Output directory for logs")
    p.add_argument("--iterations", type=int, default=10, help="Number of learning iterations")
    p.add_argument("--episodes", type=int, default=3, help="Episodes per iteration (train)")
    p.add_argument("--task-model", default="gemini/gemini-2.0-flash-lite",
                    help="Model for Harbor agent (terminus-2)")
    p.add_argument("--reflector-model", default="gemini/gemini-2.5-flash-lite",
                    help="Model for reflector")
    p.add_argument("--n-train", type=int, default=15, help="Number of train tasks")
    p.add_argument("--n-eval", type=int, default=5, help="Number of eval tasks")
    p.add_argument("--no-viewer", action="store_true", help="Don't open viewer in browser")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # -- Setup output dir --
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("Output: %s", output_dir.resolve())

    # -- Download tasks --
    task_paths = download_tasks(output_dir)

    # -- Split train/eval --
    # Eval should be hard tasks (multiple/parallel) so the curve starts low.
    # Simple tasks go to train only.
    hard_paths = [p for p in task_paths if "simple" not in p.name]
    simple_paths = [p for p in task_paths if "simple" in p.name]
    random.seed(42)
    random.shuffle(hard_paths)
    eval_paths = hard_paths[:args.n_eval]
    train_paths = hard_paths[args.n_eval:] + simple_paths
    random.shuffle(train_paths)
    log.info("Train: %d tasks (%d hard, %d simple), Eval: %d tasks (all hard)",
             len(train_paths), len([p for p in train_paths if "simple" not in p.name]),
             len(simple_paths), len(eval_paths))

    # -- Build environments --
    from clawloop.core.loop import AgentState, learning_loop
    from clawloop.environments.harbor import HarborAdapter, HarborTaskEnvironment
    from clawloop.learning_layers.harness import Harness
    from examples.recipes.common import build_local_evolver

    trial_config = {
        "agent": {
            "name": "terminus-2",
            "model_name": args.task_model,
            "kwargs": {
                "store_all_messages": True,
                "max_turns": 16,
                "temperature": 0.7,
            },
        },
        "task": {},
        "trials_dir": str(output_dir / "trials"),
    }

    train_envs = [
        HarborTaskEnvironment(task_dir=p, trial_config=trial_config)
        for p in train_paths
    ]
    eval_envs = [
        HarborTaskEnvironment(task_dir=p, trial_config=trial_config)
        for p in eval_paths
    ]
    train_adapter = HarborAdapter(train_envs)
    eval_adapter = HarborAdapter(eval_envs)
    train_task_ids = [e.task_id for e in train_envs]
    eval_task_ids = [e.task_id for e in eval_envs]

    # -- Build harness --
    # Start with a minimal prompt — intentionally bare so there's room to learn.
    # The reflector will discover strategies from failures and add them.
    evolver = build_local_evolver(args.reflector_model)
    harness = Harness(
        system_prompts={
            "harbor": (
                "You are an assistant that can execute shell commands. "
                "Complete the task described below."
            ),
        },
        evolver=evolver,
    )

    agent_state = AgentState(harness=harness)

    # -- Save config --
    config = {
        "task_model": args.task_model,
        "reflector_model": args.reflector_model,
        "iterations": args.iterations,
        "episodes_per_iter": args.episodes,
        "n_train": len(train_task_ids),
        "n_eval": len(eval_task_ids),
        "train_tasks": train_task_ids,
        "eval_tasks": eval_task_ids,
        "started_at": time.time(),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    # -- Eval callback --
    eval_log = EvalLog(output_dir)

    def after_iteration(iteration, agent_state, train_episodes):
        """Run eval episodes and log results."""
        log.info("  Running eval on %d tasks...", len(eval_task_ids))
        try:
            eval_episodes = eval_adapter.run_batch(agent_state, eval_task_ids)
        except Exception:
            log.exception("Eval batch failed")
            eval_episodes = []
        playbook_size = len(agent_state.harness.playbook.entries)
        eval_log.log_eval(iteration, eval_episodes, playbook_size)

    # -- Open viewer --
    if not args.no_viewer:
        viewer_path = Path(__file__).resolve().parent.parent.parent / "clawloop" / "static" / "learning_viewer.html"
        if viewer_path.exists():
            url = f"file://{viewer_path}?dir={output_dir.resolve()}"
            log.info("Opening viewer: %s", url)
            webbrowser.open(url)
        else:
            log.warning("Viewer not found at %s", viewer_path)

    # -- Run learning loop --
    log.info(
        "Starting: %d iterations, %d episodes/iter, %d train tasks, %d eval tasks",
        args.iterations, args.episodes, len(train_task_ids), len(eval_task_ids),
    )

    agent_state, state_id = learning_loop(
        adapter=train_adapter,
        agent_state=agent_state,
        tasks=train_task_ids,
        n_episodes=args.episodes,
        n_iterations=args.iterations,
        active_layers=["harness"],
        output_dir=str(output_dir),
        after_iteration=after_iteration,
    )

    # -- Summary --
    playbook = agent_state.harness.playbook
    print(f"\n{'='*60}")
    print(f"DONE — {args.iterations} iterations, state: {state_id.combined_hash[:12]}")
    print(f"Playbook: {len(playbook.entries)} entries")
    print(f"Logs: {output_dir.resolve()}")
    for e in playbook.entries[:5]:
        print(f"  [{e.id[:8]}] {e.content[:80]}")
    if len(playbook.entries) > 5:
        print(f"  ... and {len(playbook.entries) - 5} more")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
