"""OpenClaw Proxy + Learning Demo — improve a real agent through ClawLoop.

This demo shows the FULL value loop:

  Round 1: Agent runs tasks through the proxy (no playbook yet)
           → Proxy captures traces → Reflector analyses them
           → Learns strategies → Creates playbook entries

  Round 2: Same tasks, but now the proxy injects the learned playbook
           → Agent gets better instructions → Better responses

This is the core ClawLoop loop: observe → learn → inject → improve.
The agent requires ZERO code changes — just point base_url at the proxy.

Requires:
    - Node.js (for the pi-mono agent runner)
    - An OpenAI-compatible LLM API (OpenAI, Anthropic via proxy, vLLM, etc.)
    - Chat Completions support (`POST /v1/chat/completions`). The proxy does not
      implement `/v1/completions`, `/v1/embeddings`, or `/v1/responses`.

    Note: Google Gemini's OpenAI-compatible endpoint has SSE format
    differences that cause empty responses with pi-mono. Use OpenAI
    or an Anthropic-to-OpenAI proxy instead.

Usage:
    cd examples/openclaw_runner && npm install && cd ../..

    # With OpenAI:
    UPSTREAM_URL=https://api.openai.com/v1 UPSTREAM_KEY=$OPENAI_API_KEY \\
        PYTHONPATH=. python examples/openclaw_demo.py

    # With any OpenAI-compatible endpoint:
    UPSTREAM_URL=http://your-api/v1 UPSTREAM_KEY=your-key MODEL=model-name \\
        PYTHONPATH=. python examples/openclaw_demo.py

Bench vs live mode:
    This demo runs the proxy in `bench_mode=True`, which requires
    `X-ClawLoop-Run-Id` for session correlation (the Node runner sets it). For a
    deployed proxy, set `bench_mode=False` and configure `proxy_key` so requests
    are authenticated via `Authorization: Bearer ...`.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
import threading
import time

import uvicorn
from pydantic import SecretStr

from clawloop.collector import EpisodeCollector
from clawloop.core.episode import Episode, EpisodeSummary, StepMeta
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.core.reward import RewardPipeline
from clawloop.core.types import Datum
from clawloop.evolvers.local import LocalEvolver
from clawloop.layers.harness import Harness
from clawloop.llm import LiteLLMClient
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig


def banner(text: str) -> None:
    print(f"\n{'═' * 64}\n  {text}\n{'═' * 64}")


def show_playbook(harness: Harness) -> None:
    entries = harness.playbook.active_entries()
    if not entries:
        print("  (empty — no skills learned yet)")
        return
    for i, e in enumerate(entries, 1):
        score = f"+{e.helpful}/-{e.harmful}"
        print(f"  [{i}] {e.name or e.id}  ({score})")
        if e.description:
            print(f"      When: {e.description}")
        wrapped = textwrap.fill(
            e.content, width=60, initial_indent="      ", subsequent_indent="      "
        )
        print(wrapped)
        print()


def start_proxy(upstream_url, upstream_key, harness, collector, bench):
    config = ProxyConfig(
        upstream_url=upstream_url,
        upstream_api_key=SecretStr(upstream_key),
        bench_mode=True,
        bench=bench,
    )
    proxy = ProxyApp(config, collector=collector, harness=harness)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(
        proxy.asgi_app, host="127.0.0.1", port=port, log_level="warning",
    ))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(2)
    return port, server


def run_task(task, port, model):
    proc = subprocess.run(
        [
            "node", "examples/openclaw_runner/runner.js",
            "--base-url", f"http://127.0.0.1:{port}/v1",
            "--run-id", f"run-{task['task_id']}",
        ],
        input=json.dumps({**task, "model": model}).encode(),
        capture_output=True, timeout=60,
    )
    if proc.returncode != 0:
        return {"task_id": task["task_id"], "status": "error", "output": proc.stderr.decode()[:200]}
    return json.loads(proc.stdout.decode())


def run_round(label, tasks, port, model):
    banner(label)
    for task in tasks:
        print(f"  [{task['task_id']}] {task['instruction'][:60]}...")
        result = run_task(task, port, model)
        output = result.get("output", "")[:150]
        print(f"    → {output}{'...' if len(result.get('output', '')) > 150 else ''}\n")


def main():
    upstream_url = os.environ.get("UPSTREAM_URL", "")
    upstream_key = os.environ.get("UPSTREAM_KEY", "")
    model = os.environ.get("MODEL", "gpt-4o-mini")
    bench = "openclaw"

    if not upstream_url or not upstream_key:
        print("Set UPSTREAM_URL and UPSTREAM_KEY. Example:")
        print("  UPSTREAM_URL=https://api.openai.com/v1 UPSTREAM_KEY=$OPENAI_API_KEY \\")
        print("      PYTHONPATH=. python examples/openclaw_demo.py")
        sys.exit(1)

    tasks = [
        {"task_id": "explain-1", "instruction": "Explain what a Python decorator is."},
        {"task_id": "debug-1", "instruction": "The user says: 'My script crashes with KeyError on response[\"data\"]'. Help them debug."},
        {"task_id": "review-1", "instruction": "Review this code: `for i in range(len(lst)): print(lst[i])`"},
    ]

    # LLM for the Reflector (analyses traces, produces insights)
    reflector_llm = LiteLLMClient(
        model=f"openai/{model}", api_base=upstream_url, api_key=upstream_key,
    )
    reflector = Reflector(client=reflector_llm, config=ReflectorConfig())
    evolver = LocalEvolver(reflector=reflector)
    harness = Harness(
        system_prompts={bench: "You are a helpful coding assistant."},
        evolver=evolver,
    )

    # ── ROUND 1: Baseline (no skills) ────────────────────────────────

    episodes: list[Episode] = []
    collector = EpisodeCollector(
        pipeline=RewardPipeline.with_defaults(),
        on_batch=lambda eps: episodes.extend(eps),
        batch_size=1,
    )
    port, server = start_proxy(upstream_url, upstream_key, harness, collector, bench)
    print(f"\n  Proxy :{port} → {upstream_url} ({model})")
    show_playbook(harness)

    run_round("ROUND 1: Baseline (no playbook)", tasks, port, model)
    server.should_exit = True
    time.sleep(2)
    print(f"  Traces captured: {len(episodes)}")

    # ── LEARNING ─────────────────────────────────────────────────────

    banner("LEARNING: Reflector analyses traces")

    # Show reward signals from the pipeline (ExecutionExtractor + UserFeedback)
    for ep in episodes:
        signals = {k: f"{v.value:+.1f}" for k, v in ep.summary.signals.items()} if ep.summary.signals else {}
        reward = ep.summary.effective_reward()
        print(f"  [{ep.task_id}] reward={reward:+.2f}  signals={signals or '(none — no tool/user feedback)'}")
        # Ensure steps exist for forward_backward
        if not ep.steps:
            ep.steps = [StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)]

    print("\n  Calling LLM to extract reusable strategies...\n")

    if not episodes:
        print("  No episodes captured. Check proxy/upstream connection.")
        sys.exit(1)

    from clawloop.core.evolver import EvolverContext
    harness.set_evolver_context(EvolverContext())
    fb = harness.forward_backward(Datum(episodes=episodes)).result()
    opt = harness.optim_step().result()
    print(f"  Insights: {fb.metrics.get('insights_generated', 0)}")
    print(f"  Updates:  {opt.updates_applied}")

    banner("LEARNED PLAYBOOK")
    show_playbook(harness)

    # ── ROUND 2: With learned skills ─────────────────────────────────

    episodes2: list[Episode] = []
    collector2 = EpisodeCollector(
        pipeline=RewardPipeline.with_defaults(),
        on_batch=lambda eps: episodes2.extend(eps),
        batch_size=1,
    )
    port2, server2 = start_proxy(upstream_url, upstream_key, harness, collector2, bench)
    n_skills = len(harness.playbook.active_entries())
    print(f"\n  Injecting {n_skills} learned skills into every LLM call")

    run_round("ROUND 2: With learned skills", tasks, port2, model)
    server2.should_exit = True
    time.sleep(2)

    # ── VERIFICATION ─────────────────────────────────────────────────

    banner("RESULT")
    print(f"  Round 1 traces: {len(episodes)}")
    print(f"  Round 2 traces: {len(episodes2)}")
    skills_leaked = any(
        "clawloop-skills" in (m.content if isinstance(m.content, str) else "")
        for ep in episodes2 for m in ep.messages
    )
    print(f"  Skills stripped from stored traces: {'yes' if not skills_leaked else 'NO — BUG!'}")

    prompt = harness.system_prompt(bench)
    banner("SYSTEM PROMPT (what the agent sees in round 2)")
    print(textwrap.indent(prompt[:600], "  "))
    if len(prompt) > 600:
        print(f"  ... ({len(prompt)} chars total)")

    print(f"\n{'═' * 64}")
    print("  observe → learn → inject → improve")
    print(f"{'═' * 64}\n")


if __name__ == "__main__":
    main()
