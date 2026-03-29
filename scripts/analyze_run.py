#!/usr/bin/env python3
"""Analyze an clawloop experiment run — prints reward curves, playbook evolution,
and per-task metric breakdowns from experiment.jsonl + results.json files.

Usage:
    uv run python scripts/analyze_run.py enterprise_clawloop/runs/car/train_icl_001
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_experiment_log(run_dir: Path) -> list[dict]:
    """Load all iterations from experiment.jsonl."""
    log_path = run_dir / "experiment.jsonl"
    if not log_path.exists():
        print(f"No experiment.jsonl found in {run_dir}", file=sys.stderr)
        return []
    entries = []
    for line in log_path.read_text().strip().split("\n"):
        if line:
            entries.append(json.loads(line))
    return entries


def load_results(run_dir: Path, iteration: int) -> dict | None:
    """Load results.json for a given iteration."""
    path = run_dir / f"iter_{iteration}" / "results.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    if "results" in raw and raw["results"]:
        return raw["results"][0]
    return raw


def print_reward_curve(entries: list[dict]) -> None:
    """Print ASCII reward curve over iterations."""
    print("=" * 60)
    print("REWARD CURVE")
    print("=" * 60)
    for e in entries:
        i = e["iteration"]
        avg = e["avg_reward"]
        mn = e["min_reward"]
        mx = e["max_reward"]
        n = e["n_episodes"]
        bar = "#" * int(avg * 40)
        print(f"  iter {i}: avg={avg:.3f} min={mn:.3f} max={mx:.3f} n={n}  |{bar}")
    print()


def print_playbook_evolution(entries: list[dict]) -> None:
    """Print playbook growth and entries over iterations."""
    print("=" * 60)
    print("PLAYBOOK EVOLUTION")
    print("=" * 60)
    for e in entries:
        i = e["iteration"]
        size = e.get("playbook_size", 0)
        pbe = e.get("playbook_entries", [])
        print(f"\n  iter {i}: {size} entries")
        for entry in pbe:
            h = entry.get("helpful", 0)
            harm = entry.get("harmful", 0)
            tags = entry.get("tags", [])
            content = entry.get("content", "")[:120]
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            score = h - harm
            indicator = "+" if score > 0 else ("-" if score < 0 else "~")
            print(f"    {indicator} h={h} harm={harm}{tag_str}: {content}")
    print()


def print_per_task_metrics(run_dir: Path, entries: list[dict]) -> None:
    """Print per-task CAR-bench metrics for each iteration."""
    print("=" * 60)
    print("PER-TASK METRICS")
    print("=" * 60)
    for e in entries:
        i = e["iteration"]
        results = load_results(run_dir, i)
        if not results:
            print(f"\n  iter {i}: no results.json")
            continue

        detailed = results.get("detailed_results_by_split", {})
        print(f"\n  iter {i}:")
        for split, tasks in detailed.items():
            for task in tasks:
                tid = task["task_id"]
                reward = task["reward"]
                info = task.get("reward_info", {}).get("info", {})
                metrics = {k: v for k, v in info.items() if k.startswith("r_") and v is not None}
                missing = info.get("tool_subset_missing_tools", [])
                errors = info.get("tool_execution_errors", [])
                traj_len = len(task.get("trajectory", []))

                status = "PASS" if reward == 1.0 else "FAIL"
                print(f"    {tid}: {status} (reward={reward:.1f}, turns={traj_len})")
                for mk, mv in metrics.items():
                    flag = "x" if mv >= 1.0 else " "
                    print(f"      [{flag}] {mk}: {mv}")
                if missing:
                    print(f"      missing tools: {missing}")
                if errors:
                    print(f"      errors: {errors}")
    print()


def print_reflector_stats(entries: list[dict]) -> None:
    """Print reflector insights generated per iteration."""
    print("=" * 60)
    print("REFLECTOR ACTIVITY")
    print("=" * 60)
    for e in entries:
        i = e["iteration"]
        fb = e.get("fb_results", {})
        harness_fb = fb.get("harness", {})
        metrics = harness_fb.get("metrics", {})
        insights = metrics.get("insights_generated", 0)
        entries_signaled = metrics.get("entries_signaled", 0)
        print(f"  iter {i}: {insights} insights generated, {entries_signaled} entries updated")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_run.py <run_dir>", file=sys.stderr)
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    entries = load_experiment_log(run_dir)
    if not entries:
        sys.exit(1)

    print(f"\nAnalyzing: {run_dir}")
    print(f"Iterations: {len(entries)}")
    print()

    print_reward_curve(entries)
    print_playbook_evolution(entries)
    print_per_task_metrics(run_dir, entries)
    print_reflector_stats(entries)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if len(entries) >= 2:
        first = entries[0]["avg_reward"]
        last = entries[-1]["avg_reward"]
        delta = last - first
        print(f"  Reward: {first:.3f} -> {last:.3f} (delta={delta:+.3f})")
    final_pb = entries[-1].get("playbook_size", 0)
    print(f"  Final playbook: {final_pb} entries")
    print()


if __name__ == "__main__":
    main()
