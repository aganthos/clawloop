# OpenClaw Adapter — LLM Proxy + pi-mono Integration

**Date**: 2026-03-28
**Status**: Draft
**Codex review**: 2 rounds, approved with minor tweaks incorporated

## Goal

Let ClawLoop improve agents built with pi-mono (github.com/badlogic/pi-mono)
in two modes:

1. **Bench mode** — run pi-mono agents against task sets inside the training
   loop, capture traces, learn (harness + weights).
2. **Live mode** — ingest traces from pi-mono agents running independently,
   learn from usage data.

Both modes use the same mechanism: an OpenAI-compatible LLM proxy that sits
between the agent and the upstream model. The proxy injects harness skills and
captures traces transparently. The agent requires zero code changes — it just
sets `base_url` to point at ClawLoop.

This is the same pattern MetaClaw uses (arXiv 2603.17187): the proxy IS the
integration layer.

## Architecture

```
pi-mono Agent                ClawLoop Server              Upstream LLM
    │                             │                            │
    ├─ POST /v1/chat/completions ─►│                            │
    │                             ├─ inject skills into sysmsg ─┤
    │                             ├─ POST /v1/chat/completions ─►│
    │                             │◄─ SSE stream ───────────────┤
    │◄─ SSE stream (passthrough) ─┤                            │
    │                             ├─ tee → EpisodeCollector     │
    │                             │                            │
```

## Component 1: LLM Proxy (`clawloop/proxy.py`)

Mounted at `/v1` on the existing `clawloop-server` Starlette app. Not a
separate process.

### Routes

| Route | Purpose |
|-------|---------|
| `POST /v1/run/<run_id>/chat/completions` | Bench mode — adapter assigns run_id |
| `POST /v1/chat/completions` | Live mode — session from header or `user` field |
| `GET /v1/models` | Passthrough to upstream |

### Skill injection

Add a new leading system message (don't mutate the agent's system message):

```
{"role": "system", "content": "<!-- clawloop-skills:v1 -->\n## Active Skills\n\n### backup-before-modify\n..."}
```

- Sentinel `<!-- clawloop-skills:v1 -->` makes injection idempotent (detect
  on retries, don't duplicate).
- New message, not mutation — preserves agent's original system prompt intact.
- Skills retrieved from `agent_state.harness` using the configured bench name.

### Streaming

Byte-for-byte SSE passthrough to the client. The proxy tees chunks into a
buffer that reconstructs the final assistant message (including incremental
tool call arguments). Buffer has a max size cap (default 512 KB per request)
to prevent OOM on long streams. Truncated traces get a `"truncated": true`
marker.

### Trace capture

When a response completes (SSE stream ends or non-streaming response returns):

1. Reconstruct the full request messages + response message.
2. Feed into `EpisodeCollector.ingest_external()` with:
   - `bench`: configured bench name (e.g. `"openclaw"`)
   - `session_id`: run_id (bench) or session header (live)
   - `model`: from upstream response
   - `usage`: from upstream response
3. Multi-turn correlation: all requests sharing a run_id or session_id
   accumulate into the same Episode. Episode is finalized when the process
   exits (bench) or after a configurable idle timeout (live, default 5 min).

### Episode correlation

| Mode | Identifier | Source |
|------|-----------|--------|
| Bench | `run_id` | URL path segment, assigned by adapter |
| Live | `session_id` | `X-ClawLoop-Session-Id` header OR OpenAI `user` field |

Live mode requires an explicit session identifier. No silent fallback to
connection-based grouping — that breaks under concurrency.

### Config

```python
class ProxyConfig(BaseModel):
    upstream_url: str           # e.g. "https://api.openai.com/v1"
    upstream_api_key: SecretStr
    bench: str = "openclaw"
    port: int = 8400            # same as clawloop-server default
    proxy_key: str = ""         # optional shared secret (CLAWLOOP_PROXY_KEY)
    max_tee_bytes: int = 524288 # 512 KB
    live_idle_timeout_s: int = 300
```

### Security

- Bind `127.0.0.1` only by default.
- If `proxy_key` is set, enforce on all `/v1/*` routes via
  `Authorization: Bearer <key>` check.
- Redact `Authorization` headers and API keys from trace logs.
- Skill content visible to the model — same trust boundary as any system
  prompt. No secrets in playbook entries.

## Component 2: Environment Adapter (`clawloop/adapters/openclaw.py`)

`OpenClawAdapter(EnvAdapter)` — runs pi-mono agents against task sets.

### Methods

**`setup(config)`**
- Validates config (upstream URL, runner script path, task file).
- Starts a lightweight Starlette server with the proxy routes mounted.
  In bench mode the adapter owns the server lifecycle (start in setup,
  stop on teardown). In live mode the same proxy routes are mounted on
  the existing clawloop-server instead.

**`run_episode(task, agent_state)`**
- Generates a unique `run_id`.
- Spawns the Node runner as a subprocess:
  `node scripts/openclaw_runner.js --base-url http://127.0.0.1:PORT/v1/run/RUN_ID`
- Writes task JSON to stdin, reads result JSON from stdout.
- Hard timeout (configurable, default 120s). On timeout, kills the entire
  process tree (`os.killpg`).
- Collects the Episode from EpisodeCollector (keyed by run_id).
- Returns Episode.

**`list_tasks(split)`**
- Reads JSONL file (one task per line): `{"task_id": "...", "instruction": "..."}`.
- Split maps to file: `tasks/<split>.jsonl`.

**`get_traces(episode)`**
- Returns the raw proxy log entries for the episode's run_id.

**`run_batch(agent_state, task_ids)`**
- Default sequential (inherited). Can override for parallel later if needed.

### Config

```python
class OpenClawAdapterConfig(BaseModel):
    runner_script: str = "scripts/openclaw_runner.js"
    task_dir: str = "tasks/"
    timeout_s: int = 120
    node_bin: str = "node"
```

## Component 3: Node Runner (`scripts/openclaw_runner.js`)

Thin script, our code, not a pi-mono fork. ~50 lines.

```
stdin  → {"task_id": "abc", "instruction": "Fix the login bug in auth.py"}
stdout → {"task_id": "abc", "status": "success"|"error"|"timeout", "output": "..."}
stderr → debug/error logs (captured by Python adapter)
```

The runner:
1. Reads task JSON from stdin.
2. Creates a pi-mono `Agent` with `base_url` pointing at the proxy URL
   (passed via `--base-url` CLI arg).
3. Sets system prompt from task instruction.
4. Calls `agent.prompt(task.instruction)`.
5. Waits for completion (`agent.waitForIdle()`).
6. Writes result JSON to stdout.
7. Exits (code 0 on success, 1 on error).

Dependencies: `@mariozechner/pi-agent-core`, `@mariozechner/pi-ai`.
Installed via `npm install` in the runner's directory.

## What's NOT in scope

- No pi-mono TypeScript plugin or event listener (richer tool traces can be
  added later as an optional upgrade).
- No Anthropic `/v1/messages` endpoint — pi-mono can use the OpenAI provider.
- No hot LoRA swap in the proxy — the weights layer handles that separately.
- No standalone proxy mode — it's mounted on clawloop-server.
- No connection-based session fallback — require explicit IDs.

## Registration in train.py

Add `"openclaw"` to the `ENV_BUILDERS` dict in `train.py`:

```python
ENV_BUILDERS = {
    "harbor": build_harbor_env,
    "math": build_math_env,
    "entropic": build_entropic_env,
    "openclaw": build_openclaw_env,
}
```

## Testing

- Unit tests for proxy: skill injection, streaming passthrough, trace capture,
  idempotent injection, session correlation.
- Unit tests for adapter: subprocess lifecycle, timeout handling, JSONL
  parsing, Episode collection.
- Integration test: proxy + mock upstream + adapter + runner script
  (end-to-end).
- No real pi-mono dependency in unit tests — mock the Node subprocess.
