# Agent Prompt: WP3 — Cloud Pro v0

## Your Task

Build the Aganthos Cloud Pro API — a hosted version of lfx with persistence, auth,
versioning, and approval workflows. This is a **separate repo** (`aganthos-cloud`) that
imports `lfx` as a dependency.

## Architecture

The core improvement engine (Harness, Reflector, Curator, GEPA, Reward Pipeline) stays
**the same code** from the lfx package. Cloud wraps it in:

```
Client SDK (lfx.wrap with cloud_url) → API Gateway (TLS + rate limit)
  → Auth middleware (API key → workspace) → API endpoints (Starlette)
    → Postgres (metadata, harness versions, gate history)
    → S3 (full episode data, playbook snapshots)
    → Redis queue → Improvement Worker (runs Reflector + Curator + GEPA)
```

## Step 1: Init Repo

```bash
mkdir aganthos-cloud && cd aganthos-cloud
git init
# pyproject.toml with lfx==0.1.0 as dependency
# Starlette, asyncpg, boto3, redis as deps
```

Directory structure:
```
aganthos_cloud/
  api/
    app.py          — Starlette app factory
    middleware.py   — API key auth, rate limiting, request logging
    routes/
      episodes.py   — POST/GET /v1/episodes, POST feedback
      harness.py    — GET/POST /v1/harness (state, history, rollback, pending, approve, reject)
  db/
    schema.sql      — Postgres DDL
    models.py       — async DB access layer (asyncpg)
  storage/
    s3.py           — Episode data storage
  workers/
    improve.py      — Improvement worker (pulls from Redis, runs lfx learning)
  config.py         — Settings from env vars
tests/
  test_api.py
  test_auth.py
  test_persistence.py
```

## Step 2: Postgres Schema

```sql
CREATE TABLE workspaces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name TEXT NOT NULL,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE api_keys (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workspace_id UUID REFERENCES workspaces(id),
  key_prefix VARCHAR(8) NOT NULL,       -- first 8 chars for fast lookup
  key_hash TEXT NOT NULL,                -- HMAC-SHA256 of full key
  name TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  revoked_at TIMESTAMPTZ
);
CREATE INDEX idx_api_keys_prefix ON api_keys(key_prefix) WHERE revoked_at IS NULL;

CREATE TABLE episodes (
  id UUID PRIMARY KEY,
  workspace_id UUID REFERENCES workspaces(id),
  task_id TEXT,
  bench TEXT,
  reward FLOAT,
  signals JSONB,
  metadata JSONB,
  s3_key TEXT,                           -- full episode data in S3
  created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX idx_episodes_workspace ON episodes(workspace_id, created_at DESC);

CREATE TABLE harness_versions (
  id SERIAL PRIMARY KEY,
  workspace_id UUID REFERENCES workspaces(id),
  version INT NOT NULL,
  system_prompts JSONB,
  playbook JSONB,
  pareto_fronts JSONB,
  status VARCHAR(20) DEFAULT 'pending',  -- pending | active | superseded
  created_at TIMESTAMPTZ DEFAULT now(),
  promoted_at TIMESTAMPTZ,
  UNIQUE(workspace_id, version)
);

CREATE TABLE audit_log (
  id BIGSERIAL PRIMARY KEY,
  workspace_id UUID REFERENCES workspaces(id),
  key_id UUID REFERENCES api_keys(id),
  action TEXT NOT NULL,
  resource TEXT NOT NULL,
  detail JSONB,
  created_at TIMESTAMPTZ DEFAULT now()
);
```

## Step 3: API Key Auth Middleware

```python
async def auth_middleware(request, call_next):
    key = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not key:
        return JSONResponse({"error": "missing api key"}, 401)
    prefix = key[:8]
    row = await db.fetchrow(
        "SELECT id, workspace_id, key_hash FROM api_keys WHERE key_prefix=$1 AND revoked_at IS NULL",
        prefix
    )
    if not row or not hmac.compare_digest(hmac_sha256(key), row["key_hash"]):
        return JSONResponse({"error": "invalid api key"}, 401)
    request.state.workspace_id = row["workspace_id"]
    request.state.key_id = row["id"]
    return await call_next(request)
```

## Step 4: MVP Endpoints (10 routes)

```
POST   /v1/episodes                  — ingest episode (store metadata in Postgres, full data in S3)
POST   /v1/episodes/{id}/feedback    — add reward signal
GET    /v1/episodes                  — list (paginated, filtered by bench/task_id/date)
GET    /v1/episodes/{id}             — detail (fetch from S3)

GET    /v1/harness                   — current active version (prompt + playbook)
GET    /v1/harness/history           — version list with status
POST   /v1/harness/rollback          — set a previous version as active
GET    /v1/harness/pending           — pending version awaiting approval
POST   /v1/harness/pending/approve   — promote pending → active
POST   /v1/harness/pending/reject    — discard pending
```

All endpoints scoped to `request.state.workspace_id`. All writes logged to audit_log.

## Step 5: Improvement Worker

```python
# workers/improve.py
# Pulls episode batches from Redis queue
# Runs lfx learning loop (Harness.forward_backward → optim_step)
# Saves new harness state as pending version in Postgres
# Does NOT auto-promote — waits for approve/reject

while True:
    batch = redis.blpop("improve:{workspace_id}")
    episodes = fetch_episodes_from_s3(batch)
    harness = load_current_harness(workspace_id)
    # Run lfx learning
    datum = Datum(episodes=episodes)
    harness.forward_backward(datum)
    harness.optim_step()
    # Save as pending
    save_pending_version(workspace_id, harness)
```

## Step 6: cloud_url in lfx community

One change to the lfx package (in a PR against the lfx repo):

```python
# In wrapper.py and/or __init__.py:
def wrap(client, *, collector=None, cloud_url=None, cloud_api_key=None, **kw):
    if cloud_url:
        # POST episodes to cloud API instead of local collector
        collector = CloudCollector(cloud_url, cloud_api_key)
    ...
```

`CloudCollector` implements the same `ingest()` interface but POSTs to `/v1/episodes`.

## Testing

```bash
# Use testcontainers for Postgres + localstack for S3
pytest tests/ -x -v
```

## Commit Style

- Format: `fix:`, `feat:`, or `chore:` + one line description
- NO Co-Authored-By, NO multi-line bodies
