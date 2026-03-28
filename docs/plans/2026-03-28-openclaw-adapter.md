# OpenClaw Adapter — LLM Proxy + pi-mono Integration

**Date**: 2026-03-28
**Status**: Final
**Codex review**: 6 rounds total

## Goal

Let ClawLoop improve agents built with pi-mono (github.com/badlogic/pi-mono)
in two modes:

1. **Bench mode** — run pi-mono agents against task sets inside the training
   loop, capture traces, learn (harness + weights).
2. **Live mode** — ingest traces from pi-mono agents running independently,
   learn from usage data.

Both modes use the same mechanism: an OpenAI-compatible LLM proxy that sits
between the agent and the upstream model. The proxy injects harness skills and
captures traces transparently. The agent requires zero code changes — only
configuration: set `base_url` to point at ClawLoop, and in live mode set the
API key to the `proxy_key` value (pi-mono's `Model.headers` or provider
API key config).

This is the same pattern MetaClaw uses (arXiv 2603.17187): the proxy IS the
integration layer.

## Architecture

```
pi-mono Agent                ClawLoop Proxy               Upstream LLM
    │                             │                            │
    ├─ POST /v1/chat/completions ─►│                            │
    │   + X-ClawLoop-Run-Id header │                            │
    │                             ├─ inject skills into sysmsg ─┤
    │                             ├─ POST /v1/chat/completions ─►│
    │                             │◄─ SSE stream ───────────────┤
    │◄─ SSE stream (passthrough) ─┤                            │
    │                             ├─ store raw SSE events       │
    │                             │                            │
```

## Deployment model

- **Bench mode**: `ProxyApp` runs on `127.0.0.1` with an ephemeral port in a
  background thread. Single process. No auth needed (localhost only).
- **Live mode**: `ProxyApp` routes mounted on `clawloop-server` at `/v1`.
  Auth (`proxy_key`) **required** in live mode. TLS termination handled by
  reverse proxy (nginx/caddy) if exposed beyond localhost.

Single-process deployment only (both modes). Multi-worker (gunicorn/uvicorn
workers) is not supported in v1 — `turn_index` counters are in-process
atomics. **Runtime guard**: on startup, `ProxyApp` checks
`os.environ.get("WEB_CONCURRENCY")` and uvicorn worker count; raises
`ConfigError` if >1 worker detected. This prevents silent data corruption.

## Component 1: LLM Proxy (`clawloop/proxy.py`)

A reusable `ProxyApp` class that produces Starlette routes.

### Routes

| Route | Purpose |
|-------|---------|
| `POST /v1/chat/completions` | Proxy endpoint (both modes) |

`GET /v1/models` deferred — not required by pi-mono for our use cases.

### Session/episode correlation

Precedence (first match wins):

1. `X-ClawLoop-Run-Id` header (bench mode — set by Node runner)
2. `X-ClawLoop-Session-Id` header (live mode — set by agent config)
3. **Auto-generated session** (uuid4) if none of the above are present

The OpenAI `user` field is **not** used for correlation — it is too coarse
and unreliable across clients. It is stored as metadata only.

The proxy never rejects for missing session ID. If no explicit ID is
provided, a server-generated session is used and the turn is logged with
`"attributed": false`. Unattributed turns are still trainable but cannot
be correlated across requests — each gets its own single-turn Episode.

This keeps the proxy truly drop-in for any OpenAI-compatible client.

### Turn ordering

Each proxied request gets a monotonic `turn_index` (per session/run_id,
incremented via `threading.Lock` counter — single process only) assigned
at **request arrival time**. A `timestamp_ns` (monotonic clock) is also
stored. EpisodeCollector sorts turns by `(turn_index, timestamp_ns)` for
deterministic sequencing even under concurrent requests.

A "turn" is a single `/v1/chat/completions` request-response pair. Tool
call loops (assistant → tool results → assistant) span multiple turns.
The turn boundary is the HTTP request, not the logical conversation step.

### Skill injection

Prepend a new system message (don't mutate the agent's existing system
message):

```json
{"role": "system", "content": "<!-- clawloop-skills:v1 -->\n## Active Skills\n\n### backup-before-modify\n..."}
```

- Sentinel `<!-- clawloop-skills:v1 -->` makes injection idempotent (detect
  on retries, don't duplicate).
- New message, not mutation — preserves agent's original system prompt intact.
- Skills retrieved from `agent_state.harness` using the configured bench name.
- If no skills are active, no message is injected.

**Training policy**: the injected skills message is **stripped** before
ingestion into EpisodeCollector. The model should learn the agent's behavior,
not the harness scaffolding. The harness layer manages skill content
separately.

### Streaming

**Streaming responses** (`stream: true`): byte-for-byte SSE passthrough to
the client. The proxy stores raw SSE event bytes into a per-request
`bytearray` buffer. No inline reconstruction — that happens in post-
processing.

**Non-streaming responses** (`stream: false` or omitted): the proxy reads
the upstream response in chunks (same `httpx` streaming iter), forwards each
chunk to the client, and tees into the `bytearray` buffer up to the cap.
No SSE parsing needed — the complete JSON is reconstructed from the buffer
in post-processing. This avoids loading the entire response into memory.

Buffer cap (both modes): 512 KB per request (`max_tee_bytes`). If exceeded,
remaining data is still streamed through to the client but not buffered.
The turn is marked `"truncated": true`.

### Post-processing

Post-processing is **fire-and-forget from the request handler's perspective**.
The request handler enqueues work into a bounded `asyncio.Queue(maxsize=64)`.
A fixed pool of `max_post_process_tasks` (default 8) worker tasks drain the
queue. If the queue is full, the request handler **drops** the post-process
job: the turn is marked `"post_process_dropped": true` (non-trainable), a
structured warning is logged with session_id/turn_index, and a counter
`proxy_post_process_drops_total` is incremented. Client latency is never
coupled to post-processing.

**Shutdown**: on ASGI lifespan shutdown, workers drain remaining queue items
with a grace period (default 10s). After grace, remaining items are dropped
and logged. Semaphore/task cleanup uses `try/finally` to prevent leaks.

**Error handling**: worker exceptions are caught, logged as structured
errors (session_id, turn_index, traceback), and the turn is marked
non-trainable. Workers never crash — they catch and continue.

If a worker encounters a **parse failure** (malformed SSE, invalid JSON) or
**ingest failure** (EpisodeCollector rejects the turn), the turn is marked
`"post_process_failed": true` and treated as non-trainable. **Raw file
persistence** on failure: bench mode only. The `redaction_hook` is applied
to both request messages and the response buffer **before** any disk write.
In live mode, raw buffers are never persisted to disk. In both modes, turn
metadata (session_id, turn_index, error type, truncated traceback) is logged
as a structured error.

**Raw file storage** (bench mode only): `runs/<bench>/<run_id>/` directory.
Files named `<turn_index>_<timestamp>.raw`. Max size equals `max_tee_bytes`
(512 KB default). Cleaned up by adapter `teardown()`.

Steps:
1. Apply `redaction_hook` (if configured) to request messages and response
   buffer before any further processing or persistence.
2. Parse response buffer into a complete assistant message:
   - Streaming: parse raw SSE bytes → reconstruct text + tool calls + usage.
   - Non-streaming: parse JSON response directly.
3. Strip the injected skills system message from the request messages.
4. Feed into `EpisodeCollector.ingest_external()` with bench, session_id,
   model, usage, turn_index.

**Usage in streaming**: proxy adds `stream_options: {"include_usage": true}`
to upstream requests only if `upstream_supports_stream_usage` config is true
(default true, disable for non-OpenAI upstreams). If usage is missing from
the response, the turn is ingested without token counts — not a blocker.

### Cancellation and disconnect

- **Client disconnects** mid-stream → cancel upstream request immediately
  (close `httpx` response) to avoid burning tokens. Partially buffered trace
  marked `"partial": true`.
- **Upstream fails** (connection error, timeout, 5xx) → return error to
  client transparently. No proxy-level retry. Partial traces discarded.

### Upstream timeouts

Connect timeout 10s, read timeout 120s (configurable via `ProxyConfig`).
The read timeout applies **per chunk** (httpx `Timeout(read=...)` behavior),
not to the entire response. This means long-running SSE streams that
produce chunks within the timeout window are not killed — only streams that
go silent for >120s are terminated. This is the correct behavior for live
mode where agent sessions can run for minutes.
No proxy-level retry — client handles retries.

### Do-not-train signal

Clients can send `X-ClawLoop-No-Train: 1` header on any request. The turn
is captured for observability but excluded from training. Use case: debug
requests, sensitive data, manual testing.

### Episode finalization

| Mode | Trigger |
|------|---------|
| Bench | Adapter calls `collector.finalize_episode(run_id)` after subprocess exits |
| Live | Idle timeout (configurable, default 5 min since last turn) |

Bench finalization is explicit and deterministic — no timers, no races.

### Trainability rules

Non-trainable turn states: `truncated`, `partial`, `post_process_failed`,
`post_process_dropped`, `no_train_header`. Enforced centrally in
`EpisodeCollector`.

**Trainable-prefix policy**: an Episode with non-trainable turns is not
entirely discarded. Instead, the Episode is split at the first non-trainable
turn. The prefix (all turns before the failure) is trainable if it contains
at least one complete request-response pair (user messages + complete
assistant response). The suffix is stored for observability only. This
avoids throwing away good early turns when a late timeout or disconnect
occurs.

A "turn" for trainability purposes is a single `/v1/chat/completions`
request-response pair — not a logical conversation step. Tool-call loops
(assistant with tool_calls → tool results → assistant final) span multiple
turns. The split point is always between complete request-response pairs.

### Config

```python
class ProxyConfig(BaseModel):
    upstream_url: str           # e.g. "https://api.openai.com/v1"
    upstream_api_key: SecretStr
    bench: str = "openclaw"
    proxy_key: str = ""         # required in live mode, optional in bench
    max_tee_bytes: int = 524288 # 512 KB
    live_idle_timeout_s: int = 300
    upstream_connect_timeout_s: float = 10.0
    upstream_read_timeout_s: float = 120.0
    upstream_supports_stream_usage: bool = True
    max_post_process_tasks: int = 8
```

No `port` field — bench mode uses ephemeral port (OS-assigned), live mode
inherits from clawloop-server config.

### Security

- Bench mode: bind `127.0.0.1` only. `X-ClawLoop-Run-Id` required on all
  requests (rejects without it) to prevent accidental cross-process
  contamination on shared hosts.
- Live mode: `proxy_key` **required** (enforced at startup). Auth via
  `Authorization: Bearer <key>` on all `/v1/*` routes.
- **Auth flow (single-tenant)**: client sends `Authorization: Bearer <proxy_key>`
  to the proxy. The proxy **strips** this header and uses its own
  `upstream_api_key` (from `ProxyConfig`) when forwarding to the upstream LLM.
  Client credentials are never forwarded upstream. Multi-tenant (per-client
  upstream keys) is not supported in v1.
- **Header forwarding policy**: the proxy forwards only a safe allowlist of
  headers to upstream (`Content-Type`, `Accept`, `User-Agent`). All others
  are dropped. Specifically stripped: `Authorization`, `Proxy-Authorization`,
  `X-ClawLoop-*` (internal), hop-by-hop headers (`Connection`, `Keep-Alive`,
  `Transfer-Encoding`, `TE`, `Trailer`, `Upgrade`).
- Redact `Authorization` headers from stored traces. `proxy_key` is never
  logged or stored in any trace/raw file.
- `upstream_url` validated at config time: must be https (or http localhost
  for dev). No per-request override. Redirects disabled on upstream
  httpx client. `HTTP_PROXY`/`HTTPS_PROXY` env vars ignored by the upstream
  httpx client (explicit `trust_env=False`).
- Truncated/partial traces excluded from training.
- PII in messages/tool args: `redaction_hook` (optional callable) applied
  to request/response bodies before any persistence. If not configured,
  data is stored as-is (same trust boundary as existing clawloop-server
  `/ingest`).

## Component 2: Environment Adapter (`clawloop/adapters/openclaw.py`)

`OpenClawAdapter(EnvAdapter)` — runs pi-mono agents against task sets.

### Methods

**`setup(config)`**
- Validates config (upstream URL, runner script path, task file).
- Creates a `ProxyApp` instance and starts it on an ephemeral port in a
  background thread. Stores the assigned port for runner base_url
  construction.

**`run_episode(task, agent_state)`**
- Generates a unique `run_id` (uuid4 hex).
- Spawns the Node runner as a subprocess:
  `node scripts/openclaw_runner.js --base-url http://127.0.0.1:PORT/v1 --run-id RUN_ID`
- Writes task JSON to stdin, reads result JSON from stdout.
- Hard timeout (configurable, default 120s). On timeout, kills the entire
  process group (`os.killpg` — Unix only, documented).
- Calls `collector.finalize_episode(run_id)` to seal the Episode.
- Returns Episode.

**`list_tasks(split)`**
- Reads JSONL file (one task per line): `{"task_id": "...", "instruction": "..."}`.
- Split maps to file: `tasks/<split>.jsonl`.

**`get_traces(episode)`**
- Returns the raw SSE event log for the episode's run_id.

**`run_batch(agent_state, task_ids)`**
- Default sequential (inherited). Can override for parallel later if needed.

**`teardown()`**
- Stops the background proxy server.

### Config

```python
class OpenClawAdapterConfig(BaseModel):
    runner_script: str = "scripts/openclaw_runner.js"
    task_dir: str = "tasks/"
    timeout_s: int = 120
    node_bin: str = "node"
```

## Component 3: Node Runner (`scripts/openclaw_runner.js`)

Thin script, our code, not a pi-mono fork. ~60 lines.

```
stdin  → {"task_id": "abc", "instruction": "Fix the login bug in auth.py"}
stdout → {"task_id": "abc", "status": "success"|"error"|"timeout", "output": "..."}
stderr → debug/error logs (captured by Python adapter)
```

The runner:
1. Reads task JSON from stdin.
2. Parses `--base-url` and `--run-id` from CLI args.
3. Creates a pi-mono `Agent` with the OpenAI provider pointed at the proxy
   base_url. Adds `X-ClawLoop-Run-Id` as a default header on every request.
4. Sets system prompt from task instruction.
5. Calls `agent.prompt(task.instruction)`.
6. Waits for completion (`agent.waitForIdle()`).
7. Writes result JSON to stdout.
8. Exits (code 0 on success, 1 on error).

The `X-ClawLoop-Run-Id` header is set via pi-mono's `headers` option on the
model config — no pi-mono source changes needed:
```js
const model = getModel("openai", "gpt-4o");
model.baseUrl = baseUrl;
model.headers = { "X-ClawLoop-Run-Id": runId };
```

Dependencies: `@mariozechner/pi-agent-core`, `@mariozechner/pi-ai`.
Installed via `npm install` in the runner's directory.

## What's NOT in scope

- No pi-mono TypeScript plugin or event listener (richer tool traces can be
  added later as an optional upgrade).
- No Anthropic `/v1/messages` endpoint — pi-mono uses the OpenAI provider.
- No hot LoRA swap in the proxy — the weights layer handles that separately.
- No standalone proxy CLI entrypoint — not needed yet.
- No `GET /v1/models` — deferred, not needed for pi-mono integration.
- No proxy-level retry — client handles retries.
- No multi-worker deployment — single process only in v1.
- No Windows support for subprocess kill (`os.killpg` is Unix-specific).

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
  idempotent injection, session correlation (all 4 precedence levels),
  truncation exclusion, client disconnect cancellation, partial stream
  handling, turn ordering, do-not-train header, auto-generated sessions.
- Unit tests for adapter: subprocess lifecycle, timeout handling, JSONL
  parsing, Episode finalization.
- Unit tests for SSE post-processor: reconstruct assistant message from raw
  events, handle partial tool call deltas, handle missing usage,
  post-processor crash → raw file persistence.
- Unit tests for trainable-prefix splitting.
- Integration test: proxy + mock upstream + adapter + runner script
  (end-to-end).
- No real pi-mono dependency in unit tests — mock the Node subprocess.
