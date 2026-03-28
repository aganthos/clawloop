"""Full E2E test: pi-mono agent → proxy → CLIProxyAPI → Haiku 4.5 → collector."""
import json
import socket
import subprocess
import threading
import time

import uvicorn
from pydantic import SecretStr

from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline
from clawloop.layers.harness import Harness, Playbook, PlaybookEntry
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig


def main():
    # --- Harness with a playbook skill ---
    entry = PlaybookEntry(
        id="always-polite",
        content="Always respond politely and include a greeting.",
        name="always-polite",
        description="When responding to any user message",
        category="communication",
    )
    playbook = Playbook()
    playbook._entries = {"always-polite": entry}
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
        upstream_url="http://127.0.0.1:8317/v1",
        upstream_api_key=SecretStr("kuhhandel-bench-key"),
        bench_mode=True,
        bench="openclaw",
    )
    proxy = ProxyApp(config, collector=collector, harness=harness)

    # --- Start proxy on ephemeral port ---
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = uvicorn.Server(uvicorn.Config(
        proxy.asgi_app, host="127.0.0.1", port=port, log_level="warning",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(3)

    print(f"Proxy on :{port} → CLIProxyAPI:8317 → Haiku 4.5")
    print("=" * 60)

    # --- Run pi-mono agent ---
    task = {
        "task_id": "full-e2e",
        "instruction": "Say hello and tell me what skills you have.",
        "model": "claude-haiku-4-5-20251001",
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
    stderr = proc.stderr.decode()
    if stderr:
        print(f"[RUNNER] stderr: {stderr[:300]}")

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

        # Check skills were stripped
        skills_in_stored = any(
            "clawloop-skills" in (m.content or "")
            for m in ep.messages
        )
        print(f"\n  Skills sentinel in stored messages: {skills_in_stored}")
        print(f"  (Should be False = skills stripped before training)")
        print(f"  {'PASS' if not skills_in_stored else 'FAIL'}")
    else:
        print("  FAIL — no episodes ingested")

    # --- Summary ---
    print(f"\n{'=' * 60}")
    all_pass = (
        ok
        and len(ingested) >= 1
        and not any("clawloop-skills" in (m.content or "") for m in ingested[0].messages)
    )
    print(f"FULL E2E: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    server.should_exit = True


if __name__ == "__main__":
    main()
