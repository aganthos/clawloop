# n8n + lfx Demo

## Prerequisites

- Docker (for n8n)
- Python 3.11+ with lfx installed (`pip install -e ".[n8n]"`)
- CLIProxyAPI running on `localhost:8317`

## Quick Start

```bash
cd /Users/robertmueller/Desktop/aganthos

# 1. Start n8n in Docker
docker run -d --name n8n -p 5678:5678 \
  -e N8N_SECURE_COOKIE=false \
  --add-host=host.docker.internal:host-gateway \
  n8nio/n8n:1.76.1

# 2. Start lfx-server locally
python -m lfx.server \
  --seed-prompt enterprise_clawloop/config/seed_prompt.txt \
  --api-base http://127.0.0.1:8317/v1 \
  --api-key your-api-key-1 \
  --model openai/claude-haiku-4-5-20251001 \
  --port 8400

# 3. Import workflow into n8n (one-time setup, see below)

# 4. Open dashboard
open http://localhost:8400/dashboard/
```

## n8n Workflow Setup (one-time)

1. Open **http://localhost:5678**
2. First time: create a local owner account (no cloud signup — stays in Docker volume)
3. Click **Add workflow** (or **+**)
4. Click the **...** menu (top right) → **Import from file**
5. Select `enterprise_clawloop/examples/n8n_cliproxy/customer-support.json`
6. Click **Save** (Ctrl+S)
7. Toggle **Active** (top right) to enable the webhook

No n8n credentials needed — the workflow calls the LLM directly via HTTP Request nodes.

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

## Run the Full Demo Script

Sends 5 tickets, gives negative feedback, waits for learning, replays with positive feedback — shows the reward trend going from 0.10 to 0.90:

```bash
python enterprise_clawloop/examples/n8n_cliproxy/demo.py
```

Or use the automated 2-round demo (no n8n needed, calls lfx-server directly):

```bash
python << 'EOF'
import urllib.request, time, json

lfx = "http://localhost:8400"
n8n = "http://localhost:5678/webhook/support"

def post(url, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())

def get(url):
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())

tickets = [
    "I want a refund for order 5678. Product arrived damaged.",
    "Your app crashes on login every time.",
    "Cancel my subscription please.",
    "I was charged twice on my card.",
    "Where is my order 1234?",
]

post(f"{lfx}/reset", {})
print("Reset done.\n")

# Round 1
print("=== ROUND 1 ===")
ids1 = []
for t in tickets:
    r = post(n8n, {"message": t})
    ids1.append(r["episode_id"])
    print(f"  {t[:50]}")
    time.sleep(1)

for eid in ids1:
    post(f"{lfx}/feedback", {"episode_id": eid, "score": -0.8})
print("  Negative feedback submitted.\n")

print("  Waiting for learning...")
for _ in range(40):
    m = get(f"{lfx}/metrics")
    if m["learning_status"] == "idle" and m["playbook_version"] > 0:
        print(f"  Learned! Version {m['playbook_version']}\n")
        break
    time.sleep(2)

# Round 2
print("=== ROUND 2 ===")
ids2 = []
for t in tickets:
    r = post(n8n, {"message": t})
    ids2.append(r["episode_id"])
    print(f"  {t[:50]}")
    time.sleep(1)

for eid in ids2:
    post(f"{lfx}/feedback", {"episode_id": eid, "score": 0.8})

m = get(f"{lfx}/metrics")
trend = m["reward_trend"]
print(f"\n  Round 1 avg: {sum(trend[:5])/5:.2f}")
print(f"  Round 2 avg: {sum(trend[5:10])/5:.2f}")
print(f"  Improvement: {sum(trend[5:10])/5 - sum(trend[:5])/5:+.2f}")
EOF
```

## Dashboard

Open **http://localhost:8400/dashboard/** to see:

- **Episode feed** — each ticket with query text and response preview. Click any episode to see the full conversation, all signals, and give feedback.
- **Reward trend** — chart showing normalized reward per episode (0 = bad, 1 = good)
- **Playbook entries** — learned strategies with helpful/harmful counts
- **Insights log** — what the Reflector learned and which episodes it came from
- **Before/after prompt** — seed prompt vs current (with playbook appended)
- **Feedback buttons** — thumbs up/down per episode, or click into detail view

## Architecture

```
                    curl / browser
                         │
                         ▼
    ┌───────────────────────────────────┐
    │  n8n (Docker, localhost:5678)     │
    │  Webhook → LLM call → lfx ingest │
    └──────────┬────────────────────────┘
               │                    │
     GET /state│          POST /ingest
               ▼                    ▼
    ┌───────────────────────────────────┐
    │  lfx-server (local, :8400)       │
    │  Dashboard: :8400/dashboard/     │
    │  Episodes:  :8400/episodes       │
    └───────────────────────────────────┘
               │
               │ LLM calls (Reflector)
               ▼
    ┌───────────────────────────────────┐
    │  CLIProxyAPI (local, :8317)      │
    └───────────────────────────────────┘
```

## Stopping

```bash
# Stop lfx-server: Ctrl+C in the terminal where it's running

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
