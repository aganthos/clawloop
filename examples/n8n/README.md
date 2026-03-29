# ClawLoop + n8n Integration

Make your n8n AI agent workflows learn from experience. ClawLoop sits
alongside n8n as a learning server — your workflow sends tickets to an LLM,
posts the conversation to ClawLoop, and ClawLoop learns strategies that
improve the system prompt over time.

```
curl / browser
     │
     ▼
┌─────────────────────────────────┐
│  n8n (Docker, :5678)            │
│  Webhook → LLM call → ingest   │
└────────┬──────────────┬─────────┘
         │              │
   GET /state    POST /ingest
         ▼              ▼
┌─────────────────────────────────┐
│  clawloop-server (:8400)        │
│  Dashboard: :8400/dashboard/    │
└────────────────┬────────────────┘
                 │ Learning (LLM)
                 ▼
┌─────────────────────────────────┐
│  LLM API (via litellm)          │
│  OpenAI, Gemini, Ollama, etc.   │
└─────────────────────────────────┘
```

## Prerequisites

- Docker (for n8n)
- Python 3.11+ with ClawLoop installed (`pip install -e ".[server]"`)
- An LLM API key (`GEMINI_API_KEY`, `OPENAI_API_KEY`, or use Ollama for
  a free local setup). The clawloop-server uses litellm for the reflector.
  The n8n workflow calls the LLM directly via OpenAI-compatible HTTP —
  configure `LLM_API_URL` and `LLM_API_KEY` in n8n (see below).

## Quick Start

```bash
cd /path/to/clawloop

# 1. Create a seed prompt file
echo "You are a helpful customer support agent." > seed_prompt.txt

# 2. Start n8n in Docker
docker run -d --name n8n -p 5678:5678 \
  -e N8N_SECURE_COOKIE=false \
  --add-host=host.docker.internal:host-gateway \
  n8nio/n8n:1.76.1

# 3. Start clawloop-server
#    Set CLAWLOOP_MODEL + the matching provider key (GEMINI_API_KEY, OPENAI_API_KEY, etc.)
#    Or pass --api-key explicitly.
CLAWLOOP_MODEL=gemini/gemini-2.0-flash-lite \
  python -m clawloop.server \
    --seed-prompt seed_prompt.txt \
    --port 8400

# 4. Import the workflow into n8n (one-time, see below)

# 5. Open the dashboard in your browser
#    http://localhost:8400/dashboard/
```

**With Ollama (free, local):**
```bash
# Start Ollama first: ollama serve && ollama pull llama3.2
CLAWLOOP_MODEL=ollama/llama3.2 python -m clawloop.server \
  --seed-prompt seed_prompt.txt \
  --port 8400
```

## n8n Workflow Setup (one-time)

1. Open **http://localhost:5678**
2. First time: create a local owner account (stays in Docker volume, no cloud signup)
3. Click **Add workflow** (or **+**)
4. Click the **...** menu (top right) → **Import from file**
5. Select `examples/n8n/customer-support.json`
6. Click **Save** (Ctrl+S)
7. Toggle **Active** (top right) to enable the webhook

The workflow uses n8n environment variables for LLM configuration:
- `LLM_API_URL` — defaults to `https://api.openai.com/v1/chat/completions`
- `LLM_API_KEY` — your API key
- `LLM_MODEL` — defaults to `gpt-4o-mini`

Set these in n8n: **Settings → Environment Variables**, or pass them to
`docker run` with `-e`.

No n8n credentials needed — the workflow uses plain HTTP Request nodes.

## Send Test Messages

```bash
# Single ticket
curl -X POST http://localhost:5678/webhook/support \
  -H "Content-Type: application/json" \
  -d '{"message": "I want a refund for order 5678. The product arrived damaged."}'

# Response includes episode_id — use it for feedback:
curl -X POST http://localhost:8400/feedback \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "PASTE_EPISODE_ID", "score": -1.0}'
```

## Run the Full Demo

Sends 5 support tickets, gives negative feedback, waits for learning,
then replays to show improved responses:

```bash
pip install httpx  # one-time
python examples/n8n/demo.py
```

## Dashboard

Open **http://localhost:8400/dashboard/** to see:

- **Episode feed** — each ticket with query and response. Click to see full conversation.
- **Reward trend** — chart showing reward per episode
- **Playbook entries** — learned strategies with helpful/harmful counts
- **Insights log** — what the Reflector learned and from which episodes
- **Before/after prompt** — seed prompt vs current (with playbook)
- **Feedback buttons** — thumbs up/down per episode

## Stopping

```bash
# Stop clawloop-server: Ctrl+C

# Stop n8n
docker stop n8n && docker rm n8n

# Or keep n8n for next time (workflow persists in Docker volume)
docker stop n8n
docker start n8n  # next time
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Send episode (messages + metadata) |
| `/feedback` | POST | Submit reward feedback (episode_id + score [-1,1]) |
| `/state` | GET | Current prompt, playbook, version, status |
| `/metrics` | GET | Aggregated metrics + reward trend |
| `/episodes` | GET | All episodes with full conversations |
| `/events` | GET | SSE stream (live dashboard updates) |
| `/reset` | POST | Clear all state, reload seed prompt |
| `/dashboard/` | GET | Live dashboard UI |
