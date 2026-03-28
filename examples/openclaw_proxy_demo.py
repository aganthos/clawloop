"""
OpenClaw Proxy Demo — Improve any OpenAI-compatible agent with ClawLoop.

This example shows how ClawLoop's LLM proxy sits between an agent and the
upstream LLM, transparently:
  1. Injecting playbook skills learned from past episodes
  2. Capturing every conversation trace for training
  3. Stripping the injected skills before storing (model learns behavior, not scaffolding)

The agent requires ZERO code changes — just point base_url at the proxy.

Usage:
    # Install runner deps (one time)
    cd scripts/openclaw_runner && npm install && cd ../..

    # Run the demo
    PYTHONPATH=. python examples/openclaw_proxy_demo.py

    # Or with a custom upstream:
    UPSTREAM_URL=http://localhost:8317/v1 UPSTREAM_KEY=my-key MODEL=claude-haiku-4-5-20251001 \\
        PYTHONPATH=. python examples/openclaw_proxy_demo.py

Architecture:
    Agent  ──► ClawLoop Proxy ──► Upstream LLM
                │                      │
                ├─ inject skills       │
                ├─ forward request ────┤
                │◄─ SSE stream ────────┤
                ├─ tee response        │
                ├─ strip skills        │
                └─ ingest Episode      │
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time

import uvicorn
from pydantic import SecretStr

from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline
from clawloop.layers.harness import Harness, Playbook, PlaybookEntry
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig


def build_demo_harness() -> Harness:
    """Create a harness with example playbook skills."""
    entries = {
        "concise-answers": PlaybookEntry(
            id="concise-answers",
            name="concise-answers",
            description="When answering questions",
            content="Keep answers concise. Lead with the answer, then explain.",
            anti_patterns="Don't write long preambles before getting to the point.",
            category="communication",
        ),
        "code-examples": PlaybookEntry(
            id="code-examples",
            name="code-examples",
            description="When explaining code concepts",
            content="Always include a short code example. Show, don't just tell.",
            category="coding",
        ),
    }
    playbook = Playbook()
    playbook._entries = entries
    return Harness(
        system_prompts={"openclaw": "You are a helpful coding assistant."},
        playbook=playbook,
    )


def main():
    # --- Config from env ---
    upstream_url = os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8317/v1")
    upstream_key = os.environ.get("UPSTREAM_KEY", "kuhhandel-bench-key")
    model = os.environ.get("MODEL", "claude-haiku-4-5-20251001")
    task_file = os.environ.get("TASKS", "examples/openclaw_tasks/base.jsonl")

    # --- Build components ---
    harness = build_demo_harness()
    episodes = []
    collector = EpisodeCollector(
        pipeline=RewardPipeline.with_defaults(),
        on_batch=lambda eps: episodes.extend(eps),
        batch_size=1,
    )
    config = ProxyConfig(
        upstream_url=upstream_url,
        upstream_api_key=SecretStr(upstream_key),
        bench_mode=True,
        bench="openclaw",
    )
    proxy = ProxyApp(config, collector=collector, harness=harness)

    # --- Start proxy ---
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(
        proxy.asgi_app, host="127.0.0.1", port=port, log_level="warning",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(2)

    print(f"ClawLoop Proxy running on http://127.0.0.1:{port}/v1")
    print(f"Upstream: {upstream_url}")
    print(f"Model:    {model}")
    print(f"Skills:   {len(harness.playbook.active_entries())} active playbook entries")
    print()

    # --- Load tasks ---
    tasks = []
    with open(task_file) as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))

    print(f"Running {len(tasks)} tasks through pi-mono agent...\n")
    print("=" * 60)

    # --- Run each task ---
    for i, task in enumerate(tasks):
        task["model"] = model
        run_id = f"demo-{task['task_id']}"

        print(f"\n[Task {i+1}/{len(tasks)}] {task['task_id']}")
        print(f"  Instruction: {task['instruction'][:80]}")

        proc = subprocess.run(
            [
                "node", "scripts/openclaw_runner/runner.js",
                "--base-url", f"http://127.0.0.1:{port}/v1",
                "--run-id", run_id,
            ],
            input=json.dumps(task).encode(),
            capture_output=True,
            timeout=30,
        )

        if proc.returncode != 0:
            print(f"  FAILED (exit {proc.returncode})")
            print(f"  {proc.stderr.decode()[:200]}")
            continue

        result = json.loads(proc.stdout.decode())
        output = result.get("output", "")
        preview = output[:150] + ("..." if len(output) > 150 else "")
        print(f"  Response: {preview}")

    # --- Show captured episodes ---
    time.sleep(2)
    print(f"\n{'=' * 60}")
    print(f"\nEpisodes captured: {len(episodes)}")

    for ep in episodes:
        print(f"\n  [{ep.session_id}]")
        for m in ep.messages:
            c = m.content if isinstance(m.content, str) else str(m.content)
            role_tag = f"[{m.role:>9s}]"
            preview = c[:100] + ("..." if len(c) > 100 else "")
            print(f"    {role_tag} {preview}")

        # Verify skills were stripped
        has_sentinel = any(
            "clawloop-skills" in (m.content if isinstance(m.content, str) else "")
            for m in ep.messages
        )
        if has_sentinel:
            print("    WARNING: skills sentinel found in stored episode!")
        else:
            print("    (skills stripped before storage)")

    print(f"\nThese {len(episodes)} episodes are ready for harness learning.")
    print("Run `python examples/train_runner.py examples/configs/openclaw_proxy.json`")
    print("to optimize the playbook based on these traces.")

    server.should_exit = True


if __name__ == "__main__":
    main()
