# n8n + lfx Demo

## Prerequisites

- Docker + Docker Compose
- CLIProxyAPI running on `localhost:8317` (or any OpenAI-compatible endpoint)

## Quick Start

```bash
cd /Users/robertmueller/Desktop/aganthos

# 1. Create .env in the project root (next to docker-compose.yml)
cp .env.example .env
# Edit .env if your CLIProxyAPI is on a different port than 8317

# 2. Make sure CLIProxyAPI is running
curl -s http://localhost:8317/v1/models | head -1  # should return JSON

# 3. Start both services
docker compose up --build

# 4. Open n8n and import the workflow
open http://localhost:5678

# 5. Open the lfx dashboard
open http://localhost:8400/dashboard/
```

## Import the Workflow

1. Open n8n at **http://localhost:5678** (first time: create an owner account — local only, no cloud needed)
2. Click **Add workflow**
3. Click the **...** menu (top right) → **Import from file**
4. Select `n8n-workflows/customer-support.json`
5. Click **Save**, then toggle **Active** (top right)

## Workflow Explained

```
Webhook (POST /webhook/support)
  → GET lfx-server:8400/state          ← fetches learned prompt
  → POST CLIProxyAPI/chat/completions   ← calls LLM with learned prompt
  → POST lfx-server:8400/ingest         ← sends trace for learning
  → Respond to Webhook                  ← returns LLM response
```

No n8n credentials needed — the workflow calls the LLM directly via HTTP Request nodes using your CLIProxyAPI.

## Test It

```bash
# Send a support ticket through n8n
curl -X POST http://localhost:5678/webhook/support \
  -H "Content-Type: application/json" \
  -d '{"message": "I want a refund for order 5678. The product arrived damaged."}'

# Open the lfx dashboard to watch learning
open http://localhost:8400/dashboard/

# Submit feedback (use episode_id from the curl response)
curl -X POST http://localhost:8400/feedback \
  -H "Content-Type: application/json" \
  -d '{"episode_id": "EPISODE_ID_HERE", "score": -1.0}'
```

## Demo Script

After sending 5+ tickets and submitting feedback, check the dashboard — you should see:
- Episodes in the feed
- Reward trend chart
- After learning completes: new playbook entries showing what the agent learned
- Subsequent tickets get better responses (the prompt now includes learned strategies)

## Architecture

```
                    curl / browser
                         │
                         ▼
    ┌───────────────────────────────────┐
    │  n8n (localhost:5678)             │
    │  Webhook → LLM call → lfx ingest │
    └──────────┬────────────────────────┘
               │                    │
     GET /state│          POST /ingest
               ▼                    ▼
    ┌───────────────────────────────────┐
    │  lfx-server (localhost:8400)      │
    │  Dashboard: localhost:8400/dashboard│
    └───────────────────────────────────┘
               │
               │ LLM calls (Reflector)
               ▼
    ┌───────────────────────────────────┐
    │  CLIProxyAPI (localhost:8317)     │
    └───────────────────────────────────┘
```
