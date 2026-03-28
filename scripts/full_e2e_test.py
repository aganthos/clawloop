"""Full E2E test: pi-mono agent → ClawLoop proxy → real LLM → trace capture.

PRIVATE — not synced to public clawloop repo (scripts/ not in .publicpaths).

Proves the complete flow works end-to-end with a real LLM:
  1. Proxy starts with a Harness (playbook skills)
  2. pi-mono agent (Node runner) sends prompt through proxy
  3. Proxy injects skills, forwards to upstream, streams back
  4. Proxy captures trace, strips skills, ingests into EpisodeCollector
  5. Episode is ready for harness learning

Usage:
    # Default: auto-detects CLIProxyAPI on :8317 (free, preferred)
    PYTHONPATH=. python scripts/full_e2e_test.py

    # Or with explicit upstream:
    UPSTREAM_URL=https://api.openai.com/v1 UPSTREAM_KEY=sk-... MODEL=gpt-4o-mini \
        PYTHONPATH=. python scripts/full_e2e_test.py
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time

import httpx
import uvicorn
from pydantic import SecretStr

from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline
from clawloop.layers.harness import Harness, Playbook, PlaybookEntry
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig


def _pick_upstream() -> tuple[str, str, str]:
    """Auto-detect upstream. CLIProxyAPI first (free), then env vars."""
    if os.environ.get("UPSTREAM_URL"):
        return (
            os.environ["UPSTREAM_URL"],
            os.environ.get("UPSTREAM_KEY", ""),
            os.environ.get("MODEL", "gpt-4o-mini"),
        )
    # CLIProxyAPI (free, always available locally)
    try:
        r = httpx.get(
            "http://127.0.0.1:8317/v1/models",
            headers={"Authorization": "Bearer kuhhandel-bench-key"},
            timeout=2,
        )
        if r.status_code == 200:
            return "http://127.0.0.1:8317/v1", "kuhhandel-bench-key", "claude-haiku-4-5-20251001"
    except Exception:
        pass
    # Gemini (free)
    if os.environ.get("GEMINI_API_KEY"):
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            os.environ["GEMINI_API_KEY"],
            "gemini-2.5-flash-lite",
        )
    print("ERROR: No API. Start CLIProxyAPI or set UPSTREAM_URL/GEMINI_API_KEY")
    sys.exit(1)


def main():
    upstream_url, upstream_key, model = _pick_upstream()

    # --- Harness with playbook skills ---
    entries = {
        "always-polite": PlaybookEntry(
            id="always-polite",
            content="Always respond politely and include a greeting.",
            name="always-polite",
            description="When responding to any user message",
            category="communication",
        ),
    }
    playbook = Playbook()
    playbook._entries = entries
    harness = Harness(
        system_prompts={"openclaw": "You are a helpful assistant."},
        playbook=playbook,
    )

    # --- Collector ---
    ingested = []
    collector = EpisodeCollector(
        pipeline=RewardPipeline.with_defaults(),
        on_batch=lambda eps: ingested.extend(eps),
        batch_size=1,
    )

    # --- Proxy ---
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
    time.sleep(3)

    print(f"Proxy on :{port} → {upstream_url} ({model})")
    print("=" * 60)

    # --- Run pi-mono agent ---
    task = {
        "task_id": "full-e2e",
        "instruction": "Say hello and tell me what skills you have.",
        "model": model,
    }
    proc = subprocess.run(
        [
            "node", "scripts/openclaw_runner/runner.js",
            "--base-url", f"http://127.0.0.1:{port}/v1",
            "--run-id", "full-e2e-001",
        ],
        input=json.dumps(task).encode(),
        capture_output=True,
        timeout=30,
    )

    print(f"\n[RUNNER] exit={proc.returncode}")
    if proc.stderr:
        print(f"[RUNNER] stderr: {proc.stderr.decode()[:300]}")

    result = json.loads(proc.stdout.decode())
    print(f"[RUNNER] status={result['status']}")
    print(f"[RUNNER] output: {result['output'][:300]}")

    ok = result["status"] == "success" and len(result["output"]) > 0
    print(f"\n[CHECK 1] Runner got LLM response: {'PASS' if ok else 'FAIL'}")

    # --- Check collector ---
    time.sleep(2)
    print(f"\n[CHECK 2] Collector ingestion:")
    print(f"  Episodes: {len(ingested)}")

    if ingested:
        ep = ingested[0]
        print(f"  Session:  {ep.session_id}")
        print(f"  Bench:    {ep.bench}")
        print(f"  Messages: {len(ep.messages)}")
        for m in ep.messages:
            c = m.content if isinstance(m.content, str) else str(m.content)
            preview = c[:120] + ("..." if len(c) > 120 else "")
            print(f"    [{m.role}] {preview}")

        skills_in_stored = any(
            "clawloop-skills" in (m.content if isinstance(m.content, str) else "")
            for m in ep.messages
        )
        print(f"\n  Skills stripped: {'PASS' if not skills_in_stored else 'FAIL'}")
    else:
        print("  FAIL — no episodes ingested")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    all_pass = (
        ok
        and len(ingested) >= 1
        and not any(
            "clawloop-skills" in (m.content if isinstance(m.content, str) else "")
            for m in ingested[0].messages
        )
    )
    print(f"FULL E2E: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    server.should_exit = True


if __name__ == "__main__":
    main()
