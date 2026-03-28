"""Smoke test: run the ClawLoop proxy against a real LLM API.

PRIVATE — not synced to public clawloop repo (scripts/ not in .publicpaths).

Usage:
    # Via CLIProxyAPI (preferred, free):
    UPSTREAM_URL=http://127.0.0.1:8317/v1 UPSTREAM_KEY=kuhhandel-bench-key MODEL=claude-haiku-4-5-20251001 \
        PYTHONPATH=. python scripts/smoke_test_proxy.py

    # Via Gemini (free):
    source .env && PYTHONPATH=. python scripts/smoke_test_proxy.py

    # Via OpenAI:
    PYTHONPATH=. python scripts/smoke_test_proxy.py
"""
import json
import os
import socket
import sys
import threading
import time

import httpx
import uvicorn
from pydantic import SecretStr

from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig


def _pick_upstream() -> tuple[str, str, str]:
    """Auto-detect which API to use from env vars."""
    # Explicit override takes priority
    if os.environ.get("UPSTREAM_URL"):
        return (
            os.environ["UPSTREAM_URL"],
            os.environ.get("UPSTREAM_KEY", ""),
            os.environ.get("MODEL", "gpt-4o-mini"),
        )
    # CLIProxyAPI (free, preferred for internal testing)
    try:
        import httpx
        r = httpx.get("http://127.0.0.1:8317/v1/models",
                       headers={"Authorization": "Bearer kuhhandel-bench-key"},
                       timeout=2)
        if r.status_code == 200:
            return (
                "http://127.0.0.1:8317/v1",
                "kuhhandel-bench-key",
                "claude-haiku-4-5-20251001",
            )
    except Exception:
        pass
    # Gemini (free)
    if os.environ.get("GEMINI_API_KEY"):
        return (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            os.environ["GEMINI_API_KEY"],
            "gemini-2.5-flash-lite",
        )
    # OpenAI
    if os.environ.get("OPENAI_API_KEY"):
        return (
            "https://api.openai.com/v1",
            os.environ["OPENAI_API_KEY"],
            "gpt-4o-mini",
        )
    print("ERROR: No API available. Start CLIProxyAPI, or set GEMINI_API_KEY or OPENAI_API_KEY")
    sys.exit(1)


def _find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    upstream_url, api_key, model = _pick_upstream()
    ingested = []

    collector = EpisodeCollector(
        pipeline=RewardPipeline.with_defaults(),
        on_batch=lambda eps: ingested.extend(eps),
        batch_size=1,
    )

    config = ProxyConfig(
        upstream_url=upstream_url,
        upstream_api_key=SecretStr(api_key),
        bench_mode=True,
        bench="smoke-test",
    )
    proxy = ProxyApp(config, collector=collector)

    port = _find_free_port()
    server = uvicorn.Server(uvicorn.Config(
        proxy.asgi_app, host="127.0.0.1", port=port, log_level="warning",
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            httpx.get(f"http://127.0.0.1:{port}/", timeout=1)
            break
        except Exception:
            time.sleep(0.1)

    proxy_url = f"http://127.0.0.1:{port}/v1"
    print(f"Proxy:    {proxy_url}")
    print(f"Upstream: {upstream_url}")
    print(f"Model:    {model}")
    print("=" * 60)

    passed = 0
    failed = 0

    # --- Test 1: Non-streaming ---
    print("\n[1] Non-streaming request...")
    try:
        resp = httpx.post(
            f"{proxy_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Say 'hello smoke test' and nothing else."}],
                "max_tokens": 20,
            },
            headers={"X-ClawLoop-Run-Id": "smoke-nonstream"},
            timeout=30.0,
        )
        print(f"    Status:  {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            print(f"    Content: {content!r}")
            print(f"    Usage:   {usage}")
            print(f"    Model:   {data.get('model', '?')}")
            print("    PASS")
            passed += 1
        else:
            print(f"    Body: {resp.text[:300]}")
            print("    FAIL")
            failed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1

    # --- Test 2: Streaming ---
    print("\n[2] Streaming SSE request...")
    chunks_received = 0
    full_content = ""
    try:
        with httpx.stream(
            "POST",
            f"{proxy_url}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "Count 1 to 5, one per line."}],
                "max_tokens": 50,
                "stream": True,
            },
            headers={"X-ClawLoop-Run-Id": "smoke-stream"},
            timeout=30.0,
        ) as resp:
            print(f"    Status: {resp.status_code}")
            for line in resp.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunks_received += 1
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            full_content += delta["content"]
                    except json.JSONDecodeError:
                        pass
                elif line == "data: [DONE]":
                    pass

        print(f"    Chunks:  {chunks_received}")
        print(f"    Content: {full_content!r}")
        if chunks_received > 1 and full_content:
            print("    PASS")
            passed += 1
        else:
            print("    FAIL")
            failed += 1
    except Exception as e:
        print(f"    FAIL: {e}")
        failed += 1

    # --- Test 3: Collector ingestion ---
    print("\n[3] Collector ingestion...")
    time.sleep(2.0)
    print(f"    Episodes: {len(ingested)}")
    for ep in ingested:
        print(f"    - session={ep.session_id} bench={ep.bench} msgs={len(ep.messages)}")
        for m in ep.messages:
            preview = m.content[:80] + ("..." if len(m.content) > 80 else "")
            print(f"      [{m.role}] {preview}")
    if len(ingested) >= 1:
        print("    PASS")
        passed += 1
    else:
        print("    FAIL")
        failed += 1

    # --- Summary ---
    print("\n" + "=" * 60)
    print(f"SMOKE TEST: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)

    server.should_exit = True


if __name__ == "__main__":
    main()
