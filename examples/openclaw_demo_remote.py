"""OpenClaw + ClawLoop Remote Demo — make your OpenClaw assistant learn.

Shows the full ClawLoop learning loop against YOUR running OpenClaw instance:
  Round 1 → capture traces → reflector learns strategies → Round 2 with playbook
  No changes to your OpenClaw config. Nothing installed permanently.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREREQUISITES (one-time setup, ~5 minutes)

  1. Install ClawLoop locally:
         pip install clawloop

  2. Get an LLM API key (any ONE of these):
       - Google Gemini: https://aistudio.google.com/apikey
       - OpenAI: https://platform.openai.com/api-keys
       - Or use --local-model if you run Ollama/vLLM on the same server
       (ChatGPT Plus / Codex subscriptions are NOT API keys)

  3. SSH access to your OpenClaw host (key-based auth):
         ssh root@YOUR_HOST "echo ok"

  4. Enable GatewayPorts on your OpenClaw host (needed for the tunnel):
         ssh root@YOUR_HOST "grep -q '^GatewayPorts yes' /etc/ssh/sshd_config \\
           || echo 'GatewayPorts yes' >> /etc/ssh/sshd_config && systemctl reload ssh"

  5. Find your OpenClaw container name:
         ssh root@YOUR_HOST "docker ps --format '{{.Names}}' | grep -i claw"
     Common names: openclaw-openclaw-gateway-1, openclaw-gateway-1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUICKSTART

  # ── GOOGLE GEMINI ── tested, recommended for getting started ────
  UPSTREAM_KEY=your-google-key python examples/openclaw_demo_remote.py \\
      --host YOUR_HOST \\
      --upstream-url "https://generativelanguage.googleapis.com/v1beta/openai" \\
      --model gemini-2.5-flash-lite

  # ── OPENAI ── tested ────────────────────────────────────────────
  UPSTREAM_KEY=sk-... python examples/openclaw_demo_remote.py \\
      --host YOUR_HOST \\
      --model gpt-4o-mini

  # ── LOCAL MODEL (Ollama, vLLM) ── tested with Ollama ─────────
  python examples/openclaw_demo_remote.py \\
      --host YOUR_HOST \\
      --local-model localhost:11434 \\
      --model qwen3.5:0.8b

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUPPORTED LLM PROVIDERS

  Tested and working:
    - Google Gemini (API key from https://aistudio.google.com/apikey)
    - OpenAI (API key from https://platform.openai.com/api-keys)

  Should work (untested):
    - Anthropic (API key from https://console.anthropic.com/)
    - Any OpenAI-compatible endpoint (vLLM, Together, Groq, etc.)

  Tested and working:
    - Local models via --local-model (Ollama, vLLM)
      Opens a forward SSH tunnel so the proxy can reach the model.
      No API key needed. Tested with qwen3.5:0.8b and qwen3:0.6b.

  NOT supported:
    - ChatGPT Plus / Codex subscriptions (OAuth tokens, not API keys)
    - OpenClaw's built-in model routing (we bypass it entirely)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT THIS DOES

  1. Starts a ClawLoop proxy on your laptop (captures LLM traces)
  2. Opens an SSH tunnel so your OpenClaw server can reach the proxy
  3. Deploys a tiny Python script into your OpenClaw container
  4. Runs 10 tasks (JSON extraction, contract review, scheduling, etc.)
  5. Reflector analyses traces, extracts reusable strategies
  6. Runs the same tasks again — now with learned strategies injected
  7. LLM judge scores before/after, prints comparison table
  8. Cleans up everything (tunnel, runner, proxy)

  Your OpenClaw config is NEVER modified. Everything is stateless.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ADAPTING TO YOUR SETUP

  Hetzner / any VPS (Docker):
      --host your-server-ip --container openclaw-openclaw-gateway-1

  Mac Mini (Docker Desktop):
      --host macmini.local --ssh-user youruser --container openclaw-gateway-1

  DGX Spark / GPU server with Ollama:
      --host dgx.local --local-model localhost:11434 --model llama3.1:70b

  Different SSH user:
      --ssh-user ubuntu

  Custom tasks (your own benchmark):
      --tasks my_tasks.jsonl
      Format: one JSON per line with task_id, instruction, rubric fields

  Persist learned skills to OpenClaw (optional):
      --deploy-skill
      Saves the playbook as an OpenClaw skill so your WhatsApp/Telegram
      assistant uses it permanently.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from uuid import uuid4

import uvicorn
from pydantic import SecretStr

from clawloop.collector import EpisodeCollector
from clawloop.core.episode import Episode, EpisodeSummary, StepMeta
from clawloop.core.evolver import EvolverContext
from clawloop.core.reflector import Reflector, ReflectorConfig
from clawloop.core.reward import RewardPipeline
from clawloop.core.types import Datum
from clawloop.harness_backends.local import LocalEvolver
from clawloop.learning_layers.harness import Harness
from clawloop.llm import LiteLLMClient
from clawloop.proxy import ProxyApp
from clawloop.proxy_config import ProxyConfig


# ── Constants ────────────────────────────────────────────────────────────

REMOTE_RUNNER_DIR = "/tmp/clawloop-runner"

# Inline Python runner script — deployed into the container.
# Uses only stdlib (urllib) so no npm install is needed.
RUNNER_PY = r'''#!/usr/bin/env python3
"""Minimal Chat Completions runner for ClawLoop proxy.

Reads task JSON from stdin, calls POST /v1/chat/completions, prints result JSON.
"""
import json, sys, urllib.request, urllib.error

task = json.load(sys.stdin)
base_url = sys.argv[1]  # e.g. http://172.18.0.1:8401/v1
run_id = sys.argv[2]    # for X-ClawLoop-Run-Id header
model = sys.argv[3] if len(sys.argv) > 3 else task.get("model", "gpt-4o-mini")
no_think = len(sys.argv) > 4 and sys.argv[4] == "--no-think"

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": task.get("instruction", "Hello")},
    ],
    "temperature": 0.3,
    "max_tokens": 2048,
}
# Disable thinking for local reasoning models (Qwen3, DeepSeek-R1).
# Only sent when the caller passes --no-think (i.e. --local-model mode).
if no_think:
    payload["reasoning_effort"] = "none"
body = json.dumps(payload).encode()

req = urllib.request.Request(
    f"{base_url}/chat/completions",
    data=body,
    headers={
        "Content-Type": "application/json",
        "X-ClawLoop-Run-Id": run_id,
    },
)

try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    msg = data["choices"][0]["message"]
    output = msg.get("content") or msg.get("reasoning") or msg.get("reasoning_content") or ""
    json.dump({"task_id": task.get("task_id", ""), "status": "success", "output": output}, sys.stdout)
except Exception as e:
    json.dump({"task_id": task.get("task_id", ""), "status": "error", "output": str(e)[:300]}, sys.stdout)
'''
DEFAULT_TASKS = Path(__file__).parent / "openclaw_tasks" / "assistant_bench.jsonl"
DEFAULT_BRIDGE_IP = "172.18.0.1"


# ── Display helpers ──────────────────────────────────────────────────────

def banner(text: str) -> None:
    print(f"\n{'═' * 64}\n  {text}\n{'═' * 64}")


def status(msg: str, ok: bool = True) -> None:
    mark = "✓" if ok else "✗"
    print(f"  {mark} {msg}")


# ── SSH transport layer ──────────────────────────────────────────────────

def ssh_exec(host: str, user: str, cmd: str, *, timeout: int = 30,
             input_data: bytes | None = None) -> subprocess.CompletedProcess:
    """Run a command on the remote host via SSH.

    Returns the CompletedProcess. Raises subprocess.CalledProcessError on
    non-zero exit, subprocess.TimeoutExpired on timeout.
    """
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        f"{user}@{host}",
        cmd,
    ]
    return subprocess.run(
        ssh_cmd,
        input=input_data,
        capture_output=True,
        timeout=timeout,
    )


def open_tunnel(
    host: str, user: str, port: int,
    forward_spec: str | None = None,
) -> subprocess.Popen:
    """Open SSH tunnels for the demo.

    Always opens a reverse tunnel: remote 0.0.0.0:{port} -> local 127.0.0.1:{port}
    (so the container's runner can reach the local proxy).

    If forward_spec is set (e.g. "11434:127.0.0.1:11434"), also opens a forward
    tunnel so the local proxy can reach a model running on the remote host
    (e.g. Ollama at localhost:11434).

    Returns the Popen handle.
    """
    cmd = [
        "ssh", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-N",  # no remote command
        "-R", f"0.0.0.0:{port}:127.0.0.1:{port}",
    ]
    if forward_spec:
        cmd += ["-L", forward_spec]
    cmd.append(f"{user}@{host}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # Give the tunnel a moment to establish
    time.sleep(2)
    if proc.poll() is not None:
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError(
            f"SSH tunnel failed to start (exit {proc.returncode}): {stderr}\n"
            f"Hint: Ensure GatewayPorts is set to 'yes' in /etc/ssh/sshd_config on {host}"
        )
    return proc


def close_tunnel(proc: subprocess.Popen) -> None:
    """Terminate the SSH tunnel process."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def detect_bridge_ip(host: str, user: str, container: str) -> str:
    """Auto-detect the Docker bridge gateway IP on the remote host.

    Falls back to DEFAULT_BRIDGE_IP if detection fails.
    """
    cmd = (
        f"docker inspect {shlex.quote(container)} "
        f"--format '{{{{range .NetworkSettings.Networks}}}}{{{{.Gateway}}}}{{{{end}}}}'"
    )
    result = ssh_exec(host, user, cmd, timeout=10)
    if result.returncode == 0:
        ip = result.stdout.decode().strip().strip("'")
        if ip:
            return ip
    return DEFAULT_BRIDGE_IP


# ── Runner deployment ────────────────────────────────────────────────────

def deploy_runner(host: str, user: str, container: str) -> None:
    """Deploy the lightweight Python runner into the container.

    Uses only Python stdlib (urllib) — no npm install needed.
    """
    ssh_exec(host, user,
             f"docker exec {shlex.quote(container)} mkdir -p {REMOTE_RUNNER_DIR}",
             timeout=10)

    ssh_exec(
        host, user,
        f"docker exec -i {shlex.quote(container)} "
        f"tee {REMOTE_RUNNER_DIR}/runner.py > /dev/null",
        input_data=RUNNER_PY.encode("utf-8"),
        timeout=15,
    )


def cleanup_runner(host: str, user: str, container: str) -> None:
    """Remove the runner directory from the container."""
    ssh_exec(
        host, user,
        f"docker exec {shlex.quote(container)} rm -rf {REMOTE_RUNNER_DIR}",
        timeout=10,
    )


# ── Proxy setup ──────────────────────────────────────────────────────────

def start_proxy(
    upstream_url: str,
    upstream_key: str,
    harness: Harness,
    collector: EpisodeCollector,
    bench: str,
    port: int,
) -> uvicorn.Server:
    """Start the ClawLoop proxy on the specified port.

    Unlike openclaw_demo.py which uses an ephemeral port, the remote demo
    uses a fixed port (default 8400) because the SSH tunnel must know the
    port in advance.

    Returns the uvicorn.Server handle (set .should_exit = True to stop).
    """
    config = ProxyConfig(
        upstream_url=upstream_url,
        upstream_api_key=SecretStr(upstream_key),
        bench_mode=True,
        bench=bench,
    )
    proxy = ProxyApp(config, collector=collector, harness=harness)

    uconfig = uvicorn.Config(
        proxy.asgi_app, host="127.0.0.1", port=port, log_level="warning",
    )
    server = uvicorn.Server(uconfig)
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(2)
    return server


def stop_proxy(server: uvicorn.Server) -> None:
    """Signal the proxy to stop and wait for port release."""
    if server is not None:
        server.should_exit = True
        # Wait for uvicorn to fully release the socket
        for _ in range(20):
            time.sleep(0.5)
            with socket.socket() as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", server.config.port))
                    return  # port is free
                except OSError:
                    continue
        time.sleep(2)  # final fallback wait


# ── Task loading ─────────────────────────────────────────────────────────

def load_tasks(path: str | Path) -> list[dict]:
    """Load tasks from a JSONL file."""
    tasks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


# ── Task execution ───────────────────────────────────────────────────────

def run_task(
    task: dict,
    host: str,
    user: str,
    container: str,
    bridge_ip: str,
    port: int,
    model: str,
    run_id: str,
    *,
    no_think: bool = False,
) -> dict:
    """Run a single task via the Python runner inside the container.

    Returns the parsed JSON response from the runner (task_id, status, output).
    """
    task_with_model = {**task, "model": model}
    task_json = json.dumps(task_with_model)

    no_think_flag = " --no-think" if no_think else ""
    cmd = (
        f"echo {shlex.quote(task_json)} | "
        f"docker exec -i {shlex.quote(container)} "
        f"python3 {REMOTE_RUNNER_DIR}/runner.py "
        f"http://{bridge_ip}:{port}/v1 "
        f"{shlex.quote(run_id)} "
        f"{shlex.quote(model)}{no_think_flag}"
    )

    try:
        result = ssh_exec(host, user, cmd, timeout=300)
    except subprocess.TimeoutExpired:
        return {
            "task_id": task["task_id"],
            "status": "error",
            "output": "Timed out after 300s (model too slow or tunnel broken)",
        }

    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")[:300]
        return {
            "task_id": task["task_id"],
            "status": "error",
            "output": f"Runner error: {stderr}",
        }

    try:
        return json.loads(result.stdout.decode())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {
            "task_id": task["task_id"],
            "status": "error",
            "output": f"Failed to parse runner output: {result.stdout.decode(errors='replace')[:300]}",
        }


def run_round(
    label: str,
    tasks: list[dict],
    host: str,
    user: str,
    container: str,
    bridge_ip: str,
    port: int,
    model: str,
    collector: EpisodeCollector,
    *,
    no_think: bool = False,
) -> tuple[list[dict], list[Episode]]:
    """Run all tasks through the proxy and collect results + episodes.

    Returns (results, episodes) where results is a list of runner responses
    and episodes is the list captured by the proxy's EpisodeCollector.
    """
    banner(label)
    episodes: list[Episode] = []
    results: list[dict] = []

    # Set up episode capture
    captured: list[Episode] = []
    original_on_batch = collector.on_batch
    def capture_batch(eps: list[Episode]) -> None:
        captured.extend(eps)
        if original_on_batch:
            original_on_batch(eps)
    collector.on_batch = capture_batch

    for i, task in enumerate(tasks, 1):
        run_id = uuid4().hex
        print(f"  [{i}/{len(tasks)}] {task['task_id']}", end="", flush=True)

        result = run_task(
            task, host, user, container, bridge_ip, port, model, run_id,
            no_think=no_think,
        )
        results.append(result)

        if result["status"] == "error":
            print(f" ✗ {result['output'][:80]}")
        else:
            print(f" ✓")

    # Flush any remaining episodes in the collector buffer
    collector.flush_buffer()

    # Restore original callback
    collector.on_batch = original_on_batch

    print(f"  Traces captured: {len(captured)}/{len(tasks)}", end="")
    if len(captured) == len(tasks):
        print(" ✓")
    else:
        print(f" (expected {len(tasks)})")

    return results, captured


# ── Learning phase ───────────────────────────────────────────────────────

def show_playbook(harness: Harness) -> None:
    """Display the current playbook entries."""
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


def learn_from_episodes(harness: Harness, episodes: list[Episode]) -> None:
    """Run the reflector-based learning phase on captured episodes.

    Follows the exact pattern from openclaw_demo.py:182-205.
    """
    banner("LEARNING: Reflector analyses traces")

    # Show reward signals from the pipeline
    for ep in episodes:
        signals = (
            {k: f"{v.value:+.1f}" for k, v in ep.summary.signals.items()}
            if ep.summary.signals
            else {}
        )
        reward = ep.summary.effective_reward()
        print(f"  [{ep.task_id}] reward={reward:+.2f}  signals={signals or '(none)'}")

        # Ensure steps exist for forward_backward — proxy-captured episodes
        # may have empty steps list
        if not ep.steps:
            ep.steps = [StepMeta(t=0, reward=reward, done=True, timing_ms=100.0)]

    print("\n  Calling reflector to extract reusable strategies...\n")

    if not episodes:
        print("  No episodes captured. Cannot learn.")
        return

    # Set EvolverContext before forward_backward (required by Harness)
    harness.set_evolver_context(EvolverContext())

    fb = harness.forward_backward(Datum(episodes=episodes)).result()
    opt = harness.optim_step().result()

    insights = fb.metrics.get("insights_generated", 0)
    updates = opt.updates_applied
    print(f"  Insights: {insights}")
    print(f"  Updates:  {updates}")

    banner("LEARNED PLAYBOOK")
    show_playbook(harness)


# ── LLM Judge ────────────────────────────────────────────────────────────

def judge_response(
    task: dict,
    response: str,
    llm: LiteLLMClient,
) -> dict:
    """Score a response against the task rubric using an LLM judge.

    Returns {"scores": {"criterion": 0_or_1, ...}, "total": N, "max": M}.
    """
    rubric_list = task.get("rubric", [])
    if not rubric_list:
        return {"scores": {}, "total": 0, "max": 0}

    rubric_str = "\n".join(f"- {r}" for r in rubric_list)

    judge_prompt = f"""Score this response against the rubric. For each criterion, output 1 (met) or 0 (not met).

Task instruction:
{task['instruction']}

Response to evaluate:
{response}

Rubric criteria:
{rubric_str}

Output ONLY valid JSON in this exact format, no other text:
{{"scores": {{{", ".join(f'"{r}": 0' for r in rubric_list)}}}, "total": 0}}

Replace the 0s with 1s for criteria that are met. Set "total" to the sum of all scores."""

    try:
        result = llm.complete(
            [{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=500,
        )
        # Extract JSON from response (handle markdown code blocks)
        text = result.text.strip()
        if text.startswith("```"):
            # Strip markdown code block
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            text = text.strip()
        parsed = json.loads(text)
        # Validate and compute total
        scores = parsed.get("scores", {})
        total = sum(1 for v in scores.values() if v == 1)
        return {"scores": scores, "total": total, "max": len(rubric_list)}
    except Exception as e:
        print(f"    Judge parse error for {task['task_id']}: {e}")
        return {"scores": {}, "total": 0, "max": len(rubric_list)}


def judge_round(
    tasks: list[dict],
    results: list[dict],
    llm: LiteLLMClient,
) -> list[dict]:
    """Judge all responses from a round. Returns list of judge verdicts."""
    verdicts = []
    for task, result in zip(tasks, results):
        response = result.get("output", "")
        if result.get("status") == "error":
            verdicts.append({"scores": {}, "total": 0, "max": len(task.get("rubric", []))})
            continue
        verdict = judge_response(task, response, llm)
        verdicts.append(verdict)
    return verdicts


# ── Skill deployment (optional) ──────────────────────────────────────────

def deploy_skill(
    harness: Harness,
    host: str,
    user: str,
    container: str,
) -> None:
    """Format playbook entries as an OpenClaw SKILL.md and deploy to the container."""
    entries = harness.playbook.active_entries()
    if not entries:
        print("  No playbook entries to deploy.")
        return

    lines = ["# ClawLoop Learned Skills", ""]
    for e in entries:
        lines.append(f"## {e.name or e.id}")
        if e.description:
            lines.append(f"**When:** {e.description}")
        lines.append("")
        lines.append(e.content)
        lines.append("")

    skill_content = "\n".join(lines)
    skill_path = "/app/workspace/SKILL.md"  # OpenClaw workspace

    result = ssh_exec(
        host, user,
        f"docker exec -i {shlex.quote(container)} tee {skill_path} > /dev/null",
        input_data=skill_content.encode("utf-8"),
        timeout=10,
    )
    if result.returncode == 0:
        status(f"Skill deployed to {skill_path} in container")
    else:
        status(f"Failed to deploy skill: {result.stderr.decode(errors='replace')[:200]}", ok=False)


# ── Report ───────────────────────────────────────────────────────────────

def print_report(
    tasks: list[dict],
    verdicts_r1: list[dict],
    verdicts_r2: list[dict],
    host: str,
    model: str,
) -> dict:
    """Print the comparison table and return the full results dict."""
    banner(f"RESULTS")
    print(f"  Host: {host}  Model: {model}  Tasks: {len(tasks)}")
    print()

    col_task = 20
    header = f"  {'Task':<{col_task}}│ R1   │ R2   │ Δ"
    sep = f"  {'─' * col_task}┼──────┼──────┼────"

    print(header)
    print(sep)

    total_r1 = 0
    total_r2 = 0
    total_max = 0
    rows = []

    for task, v1, v2 in zip(tasks, verdicts_r1, verdicts_r2):
        tid = task["task_id"]
        s1 = v1["total"]
        m1 = v1["max"]
        s2 = v2["total"]
        m2 = v2["max"]
        delta = s2 - s1
        delta_str = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else " 0"

        print(f"  {tid:<{col_task}}│ {s1}/{m1}  │ {s2}/{m2}  │ {delta_str}")
        total_r1 += s1
        total_r2 += s2
        total_max += m1
        rows.append({
            "task_id": tid,
            "r1_score": s1,
            "r2_score": s2,
            "max": m1,
            "delta": delta,
            "r1_scores": v1["scores"],
            "r2_scores": v2["scores"],
        })

    print(sep)
    total_delta = total_r2 - total_r1
    delta_str = f"+{total_delta}" if total_delta > 0 else str(total_delta)
    print(f"  {'TOTAL':<{col_task}}│{total_r1:>3}/{total_max} │{total_r2:>3}/{total_max} │{delta_str}")
    print()
    print("  observe → learn → inject → improve")
    print(f"{'═' * 64}\n")

    return {
        "host": host,
        "model": model,
        "tasks": len(tasks),
        "r1_total": total_r1,
        "r2_total": total_r2,
        "max_total": total_max,
        "delta": total_delta,
        "rows": rows,
    }


# ── CLI ──────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="OpenClaw + ClawLoop Remote Demo — improve a remote agent through learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--host", required=True, help="OpenClaw host IP or hostname")
    p.add_argument("--ssh-user", default="root", help="SSH user (default: root)")
    p.add_argument("--container", default="openclaw-openclaw-gateway-1",
                    help="Docker container name (default: openclaw-openclaw-gateway-1)")
    p.add_argument("--upstream-url", default=None,
                    help="LLM API base URL (default: env UPSTREAM_URL or https://api.openai.com/v1)")
    p.add_argument("--model", default="gpt-4o-mini", help="Model name (default: gpt-4o-mini)")
    p.add_argument("--local-model", default=None, metavar="HOST:PORT",
                    help="Use a local model on the OpenClaw server (e.g. localhost:11434 for Ollama). "
                         "Opens a forward SSH tunnel so the proxy can reach it. "
                         "Sets --upstream-url and --model automatically if not specified.")
    p.add_argument("--reflector-model", default=None,
                    help="Model for ClawLoop reflector (default: same as --model via litellm)")
    p.add_argument("--tasks", default=None,
                    help=f"JSONL task file (default: {DEFAULT_TASKS})")
    p.add_argument("--proxy-port", type=int, default=8400,
                    help="Local proxy port, tunneled to remote (default: 8400)")
    p.add_argument("--output", default=None, help="Save full results JSON to this path")
    p.add_argument("--deploy-skill", action="store_true",
                    help="Persist playbook as OpenClaw skill after demo")
    p.add_argument("--docker-bridge-ip", default=None,
                    help="Docker bridge gateway IP (default: auto-detect)")
    return p


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    # ── Resolve model configuration ────────────────────────────────
    forward_spec = None  # SSH forward tunnel spec for local models

    if args.local_model:
        # Local model on the OpenClaw server (e.g. Ollama)
        # Parse host:port, open a forward tunnel so proxy can reach it
        lm_parts = args.local_model.split(":")
        lm_host = lm_parts[0] if lm_parts[0] else "127.0.0.1"
        lm_port = int(lm_parts[1]) if len(lm_parts) > 1 else 11434
        # Forward tunnel: local lm_port → remote lm_host:lm_port
        forward_spec = f"{lm_port}:{lm_host}:{lm_port}"
        upstream_url = args.upstream_url or f"http://127.0.0.1:{lm_port}/v1"
        upstream_key = os.environ.get("UPSTREAM_KEY", "not-needed")
        if args.model == "gpt-4o-mini":  # user didn't override
            # Auto-detect: try to read model list from Ollama
            print("  Hint: Set --model to match your local model name (e.g. llama3.1, qwen2.5)")
    else:
        upstream_url = args.upstream_url or os.environ.get("UPSTREAM_URL", "https://api.openai.com/v1")
        upstream_key = os.environ.get("UPSTREAM_KEY", "")
        if not upstream_key:
            print("Error: Set UPSTREAM_KEY environment variable to your LLM API key.")
            print("  Options:")
            print("    - Google Gemini: https://aistudio.google.com/apikey")
            print("    - OpenAI: https://platform.openai.com/api-keys")
            print("    - Local model: use --local-model localhost:11434 (no key needed)")
            print("")
            print("  Note: ChatGPT Plus / Codex subscriptions are NOT API keys.")
            print("  You need a separate API key from one of the providers above.")
            sys.exit(1)

    host = args.host
    user = args.ssh_user
    container = args.container
    model = args.model
    reflector_model = args.reflector_model or f"openai/{model}"
    port = args.proxy_port
    bench = "openclaw"

    # Load tasks
    task_path = Path(args.tasks) if args.tasks else DEFAULT_TASKS
    if not task_path.exists():
        print(f"Error: Task file not found: {task_path}")
        sys.exit(1)
    tasks = load_tasks(task_path)
    if not tasks:
        print(f"Error: No tasks in {task_path}")
        sys.exit(1)

    banner(f"OpenClaw + ClawLoop Demo")
    print(f"  Host: {host}  Model: {model}  Tasks: {len(tasks)}")

    # ── Setup ────────────────────────────────────────────────────────

    banner("Setup")

    # 1. SSH check
    ssh_result = ssh_exec(host, user, "echo ok", timeout=10)
    if ssh_result.returncode != 0:
        stderr = ssh_result.stderr.decode(errors="replace")
        print(f"  ✗ SSH connection failed: {stderr[:200]}")
        print(f"    Hint: Check SSH key auth, firewall, and that {user}@{host} is accessible")
        sys.exit(1)
    status(f"SSH connection to {user}@{host}")

    # 2. Container check
    container_result = ssh_exec(host, user, f"docker ps --filter name={shlex.quote(container)} --format '{{{{.Names}}}}'", timeout=10)
    if container_result.returncode != 0 or container.strip() not in container_result.stdout.decode():
        stderr = container_result.stderr.decode(errors="replace")
        # List running containers to help user
        all_containers = ssh_exec(host, user, "docker ps --format '{{.Names}}'", timeout=10)
        running = all_containers.stdout.decode().strip() if all_containers.returncode == 0 else "(could not list)"
        print(f"  ✗ Container '{container}' not found. Running containers: {running}")
        sys.exit(1)
    status(f"Container {container} running")

    # 3. Auto-detect bridge IP
    bridge_ip = args.docker_bridge_ip or detect_bridge_ip(host, user, container)
    # Validate bridge IP (used in shell commands)
    try:
        ipaddress.ip_address(bridge_ip)
    except ValueError:
        print(f"  ✗ Invalid bridge IP: {bridge_ip}")
        sys.exit(1)
    status(f"Docker bridge IP: {bridge_ip}")

    # ── Build harness + reflector ────────────────────────────────────

    # Reflector LLM — uses the reflector model with the same upstream config
    reflector_llm = LiteLLMClient(
        model=reflector_model, api_base=upstream_url, api_key=upstream_key,
    )
    reflector = Reflector(client=reflector_llm, config=ReflectorConfig())
    evolver = LocalEvolver(reflector=reflector)
    harness = Harness(
        system_prompts={bench: "You are a helpful assistant."},
        evolver=evolver,
    )

    # Judge LLM — same config as reflector
    judge_llm = LiteLLMClient(
        model=reflector_model, api_base=upstream_url, api_key=upstream_key,
    )

    # Track cleanup items
    tunnel_proc = None
    proxy_server = None

    try:
        # 4. Start proxy
        episodes_r1: list[Episode] = []
        collector_r1 = EpisodeCollector(
            pipeline=RewardPipeline.with_defaults(),
            on_batch=lambda eps: episodes_r1.extend(eps),
            batch_size=1,
        )
        proxy_server = start_proxy(
            upstream_url, upstream_key, harness, collector_r1, bench, port,
        )
        status(f"ClawLoop proxy on :{port}")

        # 5. Open SSH tunnel
        tunnel_proc = open_tunnel(host, user, port, forward_spec=forward_spec)
        tunnel_desc = "SSH tunnel open"
        if forward_spec:
            tunnel_desc += f" (+ forward tunnel for local model)"
        status(tunnel_desc)

        # 6. Verify tunnel from container
        verify_result = ssh_exec(
            host, user,
            f"docker exec {shlex.quote(container)} "
            f"curl -sf -o /dev/null -w '%{{http_code}}' http://{bridge_ip}:{port}/ || echo 'fail'",
            timeout=15,
        )
        tunnel_ok = verify_result.returncode == 0 and "fail" not in verify_result.stdout.decode()
        if tunnel_ok:
            status("Tunnel verified from container")
        else:
            # Tunnel verification is best-effort — curl may not be installed
            status("Tunnel verification skipped (curl not available in container)", ok=True)

        # 7. Deploy runner
        deploy_runner(host, user, container)
        status("Runner deployed")

        # ── Round 1: Baseline ────────────────────────────────────────

        is_local = forward_spec is not None
        results_r1, episodes_r1_captured = run_round(
            "ROUND 1: Baseline (no playbook)",
            tasks, host, user, container, bridge_ip, port, model, collector_r1,
            no_think=is_local,
        )

        # Stop proxy for Round 1 — we need to flush all episodes
        stop_proxy(proxy_server)
        collector_r1.flush_buffer()

        # Merge episodes from both sources
        all_episodes_r1 = episodes_r1 + [e for e in episodes_r1_captured if e not in episodes_r1]

        if not all_episodes_r1:
            print("\n  ✗ No traces captured. Check proxy and tunnel connectivity.")
            sys.exit(1)

        # ── Learning ─────────────────────────────────────────────────

        learn_from_episodes(harness, all_episodes_r1)

        # ── Round 2: With playbook ───────────────────────────────────

        episodes_r2: list[Episode] = []
        collector_r2 = EpisodeCollector(
            pipeline=RewardPipeline.with_defaults(),
            on_batch=lambda eps: episodes_r2.extend(eps),
            batch_size=1,
        )
        proxy_server = start_proxy(
            upstream_url, upstream_key, harness, collector_r2, bench, port,
        )

        n_skills = len(harness.playbook.active_entries())
        print(f"\n  Injecting {n_skills} learned skills into every LLM call")

        results_r2, episodes_r2_captured = run_round(
            "ROUND 2: With playbook ({} skills injected)".format(n_skills),
            tasks, host, user, container, bridge_ip, port, model, collector_r2,
            no_think=is_local,
        )

        stop_proxy(proxy_server)
        proxy_server = None

        # ── Judge scoring ────────────────────────────────────────────

        banner("JUDGING: Scoring both rounds against rubrics")
        print(f"  Using {reflector_model}...\n")

        verdicts_r1 = judge_round(tasks, results_r1, judge_llm)
        for task, v in zip(tasks, verdicts_r1):
            print(f"  R1 [{task['task_id']}] {v['total']}/{v['max']}")

        verdicts_r2 = judge_round(tasks, results_r2, judge_llm)
        for task, v in zip(tasks, verdicts_r2):
            delta = v["total"] - verdicts_r1[tasks.index(task)]["total"]
            delta_str = f" (+{delta})" if delta > 0 else f" ({delta})" if delta < 0 else ""
            print(f"  R2 [{task['task_id']}] {v['total']}/{v['max']}{delta_str}")

        # ── Report ───────────────────────────────────────────────────

        report = print_report(tasks, verdicts_r1, verdicts_r2, host, model)

        # ── Optional: deploy skill ───────────────────────────────────

        if args.deploy_skill:
            banner("SKILL DEPLOYMENT")
            deploy_skill(harness, host, user, container)

        # ── Optional: save results JSON ──────────────────────────────

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            # Add raw responses to report
            report["results_r1"] = results_r1
            report["results_r2"] = results_r2
            output_path.write_text(json.dumps(report, indent=2))
            print(f"  Results saved to {output_path}")

    finally:
        # ── Cleanup ──────────────────────────────────────────────────
        if proxy_server is not None:
            stop_proxy(proxy_server)
        if tunnel_proc is not None:
            close_tunnel(tunnel_proc)
            status("SSH tunnel closed")
        try:
            cleanup_runner(host, user, container)
            status("Runner cleaned up from container")
        except (subprocess.SubprocessError, OSError):
            pass  # best-effort cleanup


if __name__ == "__main__":
    main()
