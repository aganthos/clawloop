# Aganthos Cloud — Offering, Pricing & Technical Requirements

**Status:** Draft v3 for internal discussion (cloud expert review)
**Date:** 2026-03-17
**Reviewed by:** Codex (3 rounds), Claude

---

## What Aganthos Cloud Is

Aganthos Cloud is the hosted managed backend for the lfx learning-from-experience loop. Every ingestion channel — n8n community node, `lfx.wrap()`, litellm callback, Langfuse adapter, raw HTTP — sends traces to it. It runs the improvement loop and serves back better harnesses, routing decisions, and model updates.

It is NOT a separate product from lfx. It is lfx, hosted and managed, with production-grade capabilities that the local lfx-server does not have.

---

## Two Products, One Platform

Aganthos Cloud serves two distinct products with different cost structures, buyers, and sales motions:

### Product A: Managed Harness Improvement (self-serve + sales-led)

Automated improvement of system prompts, playbooks, and tool configs from production traces. Mostly automated — the improvement engine runs continuously, the customer sees results in the dashboard.

- **Cost to us:** ~$0.011/episode (Reflector + judge + storage)
- **Value to customer:** faster iteration, fewer failures, no manual prompt engineering
- **Sales motion:** self-serve via n8n node / SDK / litellm → free tier → paid tiers
- **Pricing:** platform fee + per-episode usage

### Product B: Managed Model Training (sales-led, high-touch)

LoRA/GRPO fine-tuning of the customer's model from their workflow traces. This is NOT a commodity GPU job — it requires ML engineering: reward function design, hyperparameter tuning, evaluation suite development, failed experiment iteration, regression testing, deployment validation.

- **Cost to us:** CHF 25,000-55,000 per training engagement (ML engineer time + GPU + eval)
- **Value to customer:** step-change in agent capability that harness tuning alone cannot achieve
- **Sales motion:** enterprise direct, pilot engagement, ongoing retainer
- **Pricing:** setup fee + monthly retainer + per-job compute

---

## Competitive Landscape & Pricing Benchmarks

| Competitor | Model | What customer gets | Pricing | ML engineering included? |
|---|---|---|---|---|
| **RunRL** (YC X25) | Self-serve compute | GPU cluster + RL platform. Customer brings reward fn. Full fine-tuning on 8× H100. | **$80/node-hour** (~$640/day for 14B) | No |
| **Nebius Token Factory** | Self-serve fine-tuning | LoRA or full FT on 30+ models. Token-based. | **$0.40-2.80/1M tokens** processed | No |
| **Nebius White-Glove RFT** | Invitation-only engagement | Researchers build custom reward models, architect RFT pipeline, end-to-end behavioral tuning on dedicated infra | **Not public** (enterprise 6-figure) | Yes — their researchers |
| **Adaptive ML** | Enterprise FDE engagement | Forward Deployed Engineers embed with customer. Custom training recipes, eval, deployment. Outcome-backed money-back guarantee. | **Not public** (custom quotes, likely $200-500k+/yr; customers incl. AT&T, Manulife, SK Telecom) | Yes — FDEs |

**Market has three price bands:**
1. **Raw compute** ($80/node-hour or $0.40/1M tokens) — no ML expertise, customer does everything
2. **White-glove RL** (Nebius RFT) — researcher-led, invitation-only, enterprise pricing
3. **Full FDE engagement** (Adaptive ML) — embed engineers with customer, $200k+/year

**Aganthos positioning:** We sit across bands 2 and 3 for enterprise, and uniquely own the self-serve harness improvement space (no competitor does automated prompt/playbook improvement from traces at $149-499/mo). Our advantage over Adaptive ML and Nebius: we're not just training a model — we run the full improvement loop (harness + training + eval + regression gate + ongoing retraining) from real workflow traces.

---

## Open Source / Community Edition Tie-In

### One repo, two offerings

```
lfx repo (public, source-available BSL or non-commercial license)
├── Community Edition (free, non-commercial)
│   ├── lfx.wrap(), EpisodeCollector, AsyncLearner
│   ├── Harness layer (Reflector, Pareto, Playbook)
│   ├── Reward pipeline + default extractors
│   ├── Local lfx-server (in-memory, single-user)
│   ├── n8n workflow templates
│   └── Dashboard (local)
│
└── Used by → aganthos-cloud repo (private)
    ├── Imports lfx as dependency
    ├── Adds: auth, multi-tenancy, persistence, billing
    ├── Adds: queue-based workers, managed infra
    ├── Adds: training project management, eval suites
    └── Adds: cloud API wrapper
```

**Community Edition is enough to:** run locally, prove the loop works, experiment, evaluate, build on top of.

**Community Edition is NOT enough for:** production persistence, managed judge/reflector (must bring own LLM key), regression gate at scale, multi-tenant, SLA, model training.

**The n8n community node** (`n8n-nodes-lfx`, separate small public repo or inside lfx) works with both:
- Point at local lfx-server → community edition
- Point at `api.aganthos.com` → Aganthos Cloud

### Why source-available, not full OSS

If we release under MIT/Apache and want to restrict later, all prior versions stay permissive forever. Competitors can fork. Better to start with the license we actually want:
- **BSL (Business Source License):** source-available, non-production use free, converts to open after X years
- **Or custom non-commercial license:** inspect, experiment, no commercial use without commercial license

This gives us: public repo for credibility/recruiting/ecosystem, developer experimentation, trust — without giving away commercial rights.

---

## n8n Customer Journey (end-to-end)

### Stage 1: Discovery (free)

n8n user finds the lfx node in n8n's node panel or sees a shared workflow template ("Customer Support That Learns"). They install it. Two nodes:
- **lfx Get State** — fetches current system prompt (improving over time)
- **lfx Ingest** — sends conversation back after each LLM call

They point at Aganthos Cloud free tier (just enter API key) or local lfx-server.

**What they see:** After ~20 conversations, system prompt starts improving. Playbook entries appear. Dashboard shows reward trend going up.

### Stage 2: Self-serve Pro ($149/mo)

They hit 500 episodes, or want managed judge (no own LLM key), or want regression gate.

**Monthly bill:** $149 platform + episodes under 5k limit = **$149/mo**

### Stage 3: Team ($499/mo)

5+ workflows in n8n. Want router optimization, shared playbooks, team approval.

**Monthly bill:** $499 + 10k overage episodes × $0.02 = **$699/mo**

### Stage 4: Enterprise (managed improvement)

Harness tuning has plateaued. They want model training. No longer self-serve — Aganthos ML engineer assesses their workflows, designs reward functions, runs training, evaluates, deploys.

**Cost:** Pilot CHF 15k → ongoing CHF 10k/mo (harness + monthly retraining)

**Value:** If n8n support workflow handles 10k tickets/mo and auto-resolution goes 35% → 55%, that's 2,000 fewer human tickets × $5 = **$10,000/month saved**. They pay CHF 10k for CHF 10k+ in savings — and it compounds.

### The funnel

```
n8n node (free)  →  Free tier ($0, 500 eps)
                         │
                    Pro ($149/mo)  ← needs judge, gate, more volume
                         │
                    Team ($499/mo)  ← multiple workflows, routing
                         │
                    Enterprise (CHF 10-50k/mo)  ← wants model training
```

### Comparison for an n8n customer

| Option | What they get | Monthly cost | Effort |
|---|---|---|---|
| Do nothing | Manual prompt engineering | $0 | High (constant) |
| lfx community (local) | Basic harness improvement, BYO LLM key | ~$5/mo (LLM API) | Medium |
| Aganthos Cloud Free | Same, managed, 500 eps | $0 | Low |
| Aganthos Cloud Pro | Managed harness + judge + gate | $149-300/mo | Very low |
| RunRL (DIY training) | GPU compute, you design everything | ~$5k-15k/engagement | Very high |
| Aganthos Enterprise | Full managed improvement | CHF 10-25k/mo | None (we do it) |
| Adaptive ML | FDE team embedded | $200-500k/yr | Low but expensive |

---

## Value Examples (why the pricing is justified)

### Example 1: Customer Support Automation (SaaS company)

| Metric | Before | After Aganthos |
|--------|--------|---------------|
| Tickets/month | 80,000 | 80,000 |
| Agent auto-resolution rate | 35% | 55% |
| Human handle time (remaining) | 12 min | 10 min |
| Fully-loaded human agent cost | $35/hr | $35/hr |

**Value math:**
- 16,000 more tickets auto-resolved × 12 min × $35/hr = **$112,000/mo saved**
- 36,000 remaining tickets × 2 min saved × $35/hr = **$42,000/mo saved**
- **Total: $154,000/month in savings**

**Aganthos charges:** $10,000/month ongoing = **6.5% of value created**

### Example 2: Legal/Compliance Contract Review (Fintech)

| Metric | Before | After Aganthos |
|--------|--------|---------------|
| Contracts/month | 900 | 900 |
| Human review rate | 100% | 60% (40% auto-cleared) |
| Review time (human) | 1.5 hr | 1.1 hr |
| Outside counsel cost | $220/hr | $220/hr |

**Value math:**
- 360 contracts auto-cleared × 1.5 hr × $220 = **$118,800/mo**
- 540 remaining × 0.4 hr saved × $220 = **$47,520/mo**
- **Total: $166,320/month in savings**

**Aganthos charges:** $25,000/month = **15% of value** (regulated domain, higher risk)

### Example 3: Code Review Triage (Large Engineering Org)

| Metric | Before | After Aganthos |
|--------|--------|---------------|
| PRs/month | 3,500 | 3,500 |
| Human review time | 45 min/PR | 32 min/PR |
| Critical defects caught earlier | 0% | 4% |
| Engineer cost | $120/hr | $120/hr |

**Value math:**
- 3,500 PRs × 13 min saved × $120/hr = **$91,000/mo**
- 140 PRs × 3 hr rework avoided × $120/hr = **$50,400/mo**
- **Total: $141,400/month in savings**

**Aganthos charges:** $10,000/month = **7% of value**

### Example 4: Financial Ops / Accounts Payable

| Metric | Before | After Aganthos |
|--------|--------|---------------|
| Invoices/month | 40,000 | 40,000 |
| Exception rate | 30% | 15% |
| Human review time | 8 min | 6 min |
| AP analyst cost | $40/hr | $40/hr |

**Value math:**
- 6,000 fewer exceptions × 8 min × $40/hr = **$32,000/mo**
- 6,000 remaining × 2 min saved × $40/hr = **$8,000/mo**
- **Total: $40,000/month in savings**

**Aganthos charges:** $5,000/month = **12.5% of value**

### Pricing justification summary

| Scenario | Monthly value created | Aganthos monthly fee | % of value |
|----------|---------------------|---------------------|------------|
| Customer support | $154,000 | $10,000 | 6.5% |
| Legal review | $166,000 | $25,000 | 15% |
| Code review | $141,000 | $10,000 | 7% |
| Financial ops | $40,000 | $5,000 | 12.5% |

**Rule of thumb:** Charge 5-15% of measurable value created. Higher for regulated/high-risk domains.

---

## Pricing Model

### Self-Serve: Platform Fee + Usage

For Product A (harness improvement) via n8n, SDK, litellm:

| Component | Free | Pro | Team |
|-----------|------|-----|------|
| **Platform fee** | $0/mo | $149/mo | $499/mo |
| **Episodes included** | 500 | 5,000 | 25,000 |
| **Per episode overage** | — | $0.03/ep | $0.02/ep |
| **Managed Reflector** | Rate-limited | Yes | Yes |
| **Managed judge** | No (BYOK) | Yes | Yes |
| **Regression gate** | No | Yes | Yes |
| **Workflows** | 1 | 5 | Unlimited |
| **Router optimization** | No | No | Yes |
| **Dashboard** | Basic | Full | Full + multi-workflow |
| **Support** | Community | Email | Priority |

**What a customer actually pays:**

| Customer type | Episodes/mo | Platform | Overage | **Monthly bill** |
|---------------|-------------|----------|---------|-----------------|
| Hobby/eval | 300 | $0 | $0 | **$0** |
| Small (1 workflow) | 2,000 | $149 | $0 (under limit) | **$149** |
| Medium (5 workflows) | 15,000 | $499 | 10k × $0.02 = $200 | **$699** |
| Growing (10 workflows) | 40,000 | $499 | 15k × $0.02 = $300 | **$799** |

### Mid-Market: Scale Tier ($1,499/mo)

Bridges the gap between Team ($499) and Enterprise (CHF 5k+). For companies that need more support and governance but aren't ready for managed training.

| Component | Scale |
|-----------|-------|
| **Platform fee** | $1,499/mo |
| **Episodes included** | 100,000 |
| **Per episode overage** | $0.015/ep |
| **Managed Reflector + Judge** | Yes |
| **Regression gate** | Yes + human approval workflow |
| **Router optimization** | Yes |
| **Workflows** | Unlimited |
| **Shared playbooks** | Yes |
| **Dashboard** | Full + multi-workflow + A/B |
| **Support** | Dedicated CSM |
| **SLA** | 99.9% uptime |

This de-risks the conversion for mid-market teams that can't jump to CHF 5k+.

### Enterprise: Setup + Retainer + Compute

For Product B (managed model training) and large-scale Product A:

| Component | Price | Notes |
|-----------|-------|-------|
| **Onboarding / pilot** | CHF 10,000-25,000 | Workflow assessment, reward design, first improvement cycle |
| **Monthly platform** | CHF 999-2,499/mo | Depends on workflows, scale, SLA |
| **Per episode** | CHF 0.015/ep | Volume discount |
| **Harness improvement** | Included in platform | Managed Reflector, judge, gate |
| **Model training (initial)** | CHF 25,000-55,000 | Per engagement (reward design + training + eval) |
| **Model retraining** | CHF 5,000-25,000/mo | Ongoing retainer, depends on complexity |
| **Additional training jobs** | CHF 2,000-10,000/job | On-demand, beyond retainer scope |
| **VPC/on-prem** | CHF 5,000+/mo | Infrastructure pass-through + support |

### When is retraining CHF 5k vs 25k vs 50k/month?

| Monthly retainer | Profile |
|-----------------|---------|
| **CHF 5,000** | Low-risk domain (support, ops). Mostly harness tuning with light monitoring. Single workflow. Small LoRA refresh quarterly. |
| **CHF 10,000** | 2-3 workflows. Biweekly harness review + quarterly LoRA retraining. Moderate compliance needs. |
| **CHF 25,000** | Regulated / high-stakes (legal, healthcare, finance). Monthly model updates. Red-team evals, human QA loop. Multiple agents + pipelines. |
| **CHF 50,000** | Business-critical systems at scale. Dedicated ML engineer(s) + analyst. Continuous training / A/B experimentation. Heavy evaluation, audits, data governance. |

### Cost structure (internal)

| Component | Our cost | Price to customer | Margin |
|-----------|----------|------------------|--------|
| Episode storage | ~$0.001/ep | Included | — |
| Reflector (per batch of 7 eps) | ~$0.03 | ~$0.004/ep amortized | Baked in |
| Judge (per episode, 20% trigger) | ~$0.006/ep effective | Baked into Pro+ | ~80% at Pro |
| Router optimization | ~$0 | Team feature | Pure margin |
| ML engineer time (CH) | CHF 80-120/hr | CHF 150-250/hr billed | 40-50% |
| GPU compute (7B LoRA) | CHF 10-50/run | CHF 2,000-5,000/engagement | High (bundled with engineering) |
| GPU compute (70B LoRA) | CHF 100-500/run | CHF 5,000-15,000/engagement | High (bundled with engineering) |
| Infrastructure per tenant | ~CHF 8/mo | Covered by platform fee | — |

**Key insight:** The margin on model training is NOT on GPU compute — it is on ML engineering expertise. GPU is <10% of the engagement cost. The value is in reward design, evaluation, and judgment.

---

## Compliance & Privacy (Minimum Viable Stance)

### What we do today

- **Encryption:** All data encrypted in transit (TLS 1.2+) and at rest (AES-256 via cloud provider)
- **Tenant isolation:** Workspace-level isolation on all queries; no cross-tenant data access
- **Data residency:** EU-hosted (eu-west-1 or eu-central-2). Customer data never leaves the region.
- **Retention:** Default 90 days, configurable per workspace. Customer can purge episodes on demand via `DELETE /v1/episodes/{id}`
- **PII handling:** Episodes may contain PII from customer conversations. We store what the customer sends. Customer is responsible for redacting PII before ingestion if required.
- **Access control:** API keys per workspace. Dashboard access via authenticated sessions.
- **Audit logging:** All API calls logged with workspace_id, timestamp, endpoint, response code. Available to customer on request.

### What we don't yet do (roadmap)

- SOC2 Type II (target: 6-12 months post-launch)
- HIPAA BAA (target: enterprise demand-driven)
- SSO / SAML / OIDC (target: 3-6 months)
- Customer-managed encryption keys (BYOK)
- Formal incident response SLA (target: with SOC2)
- Automated PII redaction at ingestion

### What we will NOT do

- Store or process data outside the EU without explicit customer consent
- Share customer traces across workspaces or tenants
- Use customer data to train models for other customers
- Retain data beyond the configured retention period

---

## Aganthos Cloud API (v1)

### Design principles

1. **Same API for all channels.** n8n node, SDK, litellm callback, and raw HTTP all use the same endpoints.
2. **Two-speed improvement.** Harness improvement runs automatically (seconds). Model training is explicit (days/weeks).
3. **Everything versioned.** Playbooks, prompts, adapters — all versioned with rollback.
4. **Metered.** Every billable operation is tracked and attributable to a workspace.

### Authentication

All requests require `Authorization: Bearer <api-key>` header.
API keys are scoped to a workspace. Dashboard access uses session tokens (JWT).

### MVP API (ship in 6-8 weeks)

The minimum surface to convert design partners:

### Core API: Trace Ingestion & Feedback

These endpoints power all channels (n8n, SDK, litellm, etc.):

```
POST   /v1/episodes                 Ingest an episode (messages + metadata)
POST   /v1/episodes/{id}/feedback   Submit reward signal (score [-1,1])
GET    /v1/episodes                 List episodes (paginated, filterable)
GET    /v1/episodes/{id}            Get episode detail (messages, signals, reward)
DELETE /v1/episodes/{id}            Delete episode (GDPR, cleanup)
```

**POST /v1/episodes** request:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]}
  ],
  "metadata": {
    "conversation_id": "...",
    "model": "gpt-4o",
    "usage": {"prompt_tokens": 150, "completion_tokens": 80}
  }
}
```

**POST /v1/episodes** response:
```json
{
  "id": "ep_abc123",
  "workspace_id": "ws_xyz",
  "reward_signals": {"execution": {"value": 0.6, "confidence": 0.8}},
  "harness_version": 14,
  "improvement_status": "scheduled"
}
```

### Harness State & Improvement

These endpoints serve the current (improving) harness back to the agent:

```
GET    /v1/harness                  Current system prompt + playbook + version
GET    /v1/harness/history          Harness version history with diffs
GET    /v1/harness/versions/{v}     Specific version snapshot
POST   /v1/harness/rollback         Rollback to a previous version
GET    /v1/harness/pending          Pending changes awaiting approval (Team+)
POST   /v1/harness/pending/approve  Approve pending changes
POST   /v1/harness/pending/reject   Reject pending changes
```

**GET /v1/harness** response:
```json
{
  "system_prompt": "You are a customer support agent...\n\n## Learned Strategies\n- Always ask for order number...",
  "playbook_version": 14,
  "playbook_entries": [
    {
      "id": "pb_001",
      "content": "Always ask for order number before looking up status",
      "tags": ["process", "order-tracking"],
      "helpful": 12,
      "harmful": 1,
      "source_episode_ids": ["ep_abc", "ep_def"]
    }
  ],
  "updated_at": "2026-03-17T14:30:00Z",
  "improvement_status": "idle"
}
```

This is the endpoint n8n calls before every LLM request to get the latest prompt.

### Regression Gate

```
GET    /v1/gate                     Current gate status (pass/fail/pending)
GET    /v1/gate/history             Gate evaluation history
POST   /v1/gate/evaluate            Trigger manual gate evaluation
```

**GET /v1/gate** response:
```json
{
  "status": "pending_review",
  "candidate_version": 15,
  "current_version": 14,
  "reward_comparison": {
    "current_mean": 0.72,
    "candidate_mean": 0.78,
    "p_value": 0.03,
    "sample_size": 47
  },
  "recommendation": "approve",
  "requires_human_approval": true
}
```

### Full API (add in months 3-6)

These endpoints ship after MVP is validated with paying customers.

### Model Training (Enterprise)

Training is organized into **projects** (ongoing engagement) containing **runs** (individual training jobs):

```
POST   /v1/training/projects                Create training project
GET    /v1/training/projects                List projects
GET    /v1/training/projects/{id}           Project detail + run history

POST   /v1/training/projects/{id}/runs      Submit training run
GET    /v1/training/runs/{id}               Run status + metrics
GET    /v1/training/runs/{id}/metrics       Training metrics (loss, reward curves)
POST   /v1/training/runs/{id}/cancel        Cancel running job

GET    /v1/training/adapters                List trained adapters
GET    /v1/training/adapters/{id}           Adapter detail + eval results
POST   /v1/training/adapters/{id}/promote   Promote adapter to production
POST   /v1/training/adapters/{id}/rollback  Rollback to previous adapter
```

**POST /v1/training/projects** request:
```json
{
  "name": "support-agent-v2",
  "base_model": "meta-llama/Llama-3-8B",
  "description": "LoRA fine-tuning for Tier-1 support resolution",
  "reward_config": {
    "extractors": ["execution", "user_feedback", "judge"],
    "custom_reward_fn": null
  },
  "grpo_config": {
    "n_samples_per_prompt": 4,
    "learning_rate": 1e-5,
    "kl_coeff": 0.05
  }
}
```

**GET /v1/training/runs/{id}** response:
```json
{
  "id": "run_abc123",
  "project_id": "proj_xyz",
  "status": "completed",
  "started_at": "2026-03-15T10:00:00Z",
  "completed_at": "2026-03-15T14:30:00Z",
  "episodes_used": 2847,
  "model_size": "7B",
  "gpu_hours": 4.5,
  "metrics": {
    "final_reward_mean": 0.82,
    "baseline_reward_mean": 0.65,
    "reward_delta": 0.17,
    "kl_divergence": 0.03
  },
  "adapter_id": "adpt_def456",
  "gate_result": "passed"
}
```

### Evaluation Suites (Enterprise)

```
POST   /v1/evals                    Create evaluation suite
GET    /v1/evals                    List suites
POST   /v1/evals/{id}/run           Run evaluation (against current or candidate)
GET    /v1/evals/{id}/results       Evaluation results + comparisons
```

### Router (Team+)

```
GET    /v1/router                   Current routing config + model mappings
GET    /v1/router/decisions         Recent routing decisions + outcomes
GET    /v1/router/savings           Cost savings report
```

### Workspace Management

```
GET    /v1/workspace                Current workspace info
PATCH  /v1/workspace                Update workspace settings
GET    /v1/workspace/usage          Usage report (episodes, judge calls, training)
GET    /v1/workspace/keys           List API keys
POST   /v1/workspace/keys           Create API key
DELETE /v1/workspace/keys/{id}      Revoke API key
```

### Metrics & Events

```
GET    /v1/metrics                  Aggregated metrics + reward trend
GET    /v1/events                   SSE stream for live dashboard updates
POST   /v1/webhooks                 Register webhook for events (Team+)
GET    /v1/webhooks                 List registered webhooks
```

### Event types (SSE + webhooks)

| Event | Data | Trigger |
|-------|------|---------|
| `episode.ingested` | episode_id, reward_signals | Every episode |
| `feedback.received` | episode_id, score | Every feedback |
| `harness.updated` | version, diff summary | After improvement cycle |
| `harness.pending` | version, changes | When gate requires approval |
| `gate.passed` | version, metrics | Auto-approved change |
| `gate.failed` | version, reason | Blocked regression |
| `training.started` | run_id, project_id | Training job begins |
| `training.completed` | run_id, metrics, adapter_id | Training job finishes |
| `adapter.promoted` | adapter_id, model | Adapter promoted to prod |

---

## Technical Requirements

### Infrastructure Components

```
┌──────────────────────────────────────────────────────┐
│                   API Gateway                         │
│  (auth, rate limiting, routing, TLS)                 │
│  Options: AWS ALB + API Gateway, Cloudflare, Kong    │
└─────────────────────┬────────────────────────────────┘
                      │
         ┌────────────┼────────────────┐
         ▼            ▼                ▼
┌────────────┐ ┌────────────┐ ┌──────────────┐
│ API Workers │ │ Dashboard  │ │ SSE/WebSocket│
│ (Starlette) │ │ (Static)   │ │ (Events)     │
│ 2-4 workers │ │ CDN-served │ │ 1-2 workers  │
└──────┬─────┘ └────────────┘ └──────────────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ PostgreSQL │ │ Object     │ │ Message    │
│ (metadata, │ │ Store (S3) │ │ Queue      │
│  playbooks,│ │ (episodes, │ │ (Redis /   │
│  state,    │ │  traces,   │ │  SQS)      │
│  training) │ │  adapters) │ │            │
└────────────┘ └────────────┘ └──────┬─────┘
                                     │
                        ┌────────────┼────────────┐
                        ▼            ▼            ▼
                 ┌────────────┐ ┌─────────┐ ┌─────────┐
                 │ Improvement│ │ Judge   │ │Training │
                 │ Workers    │ │ Workers │ │ Workers │
                 │ (Reflector,│ │ (LLM-as │ │ (GPU,   │
                 │  Pareto,   │ │ -judge) │ │  SkyRL) │
                 │  Playbook) │ │         │ │         │
                 │ CPU only   │ │ CPU+API │ │ GPU     │
                 └────────────┘ └─────────┘ └─────────┘
```

### Component Details

#### 1. API Workers (CPU)
- **What:** Starlette/FastAPI workers (adaptation of existing `lfx/server.py`)
- **Changes from current code:**
  - API key auth middleware (per-workspace)
  - Workspace/tenant isolation on all queries
  - DB persistence (replace in-memory state)
  - Queue-based job dispatch (replace in-process AsyncLearner)
  - Full REST API surface (see API section above)
  - Usage metering (per-episode, per-judge, per-training)
- **Scaling:** Horizontal, stateless. 2-4 workers per region to start.
- **Compute:** Small instances (2 vCPU, 4GB RAM each)

#### 2. PostgreSQL
- **Stores:** Organizations, workspaces, API keys, users, playbook state (entries + versions + Pareto fronts), router state, training projects + runs + adapters, evaluation suites + results, gate history, usage/billing records, webhook registrations
- **Size:** Small to start. ~1KB per playbook entry, ~200 bytes per episode metadata row (full episode data in S3).
- **Scaling:** Single instance to start. Read replicas when needed.

#### 3. Object Store (S3 / GCS / R2)
- **Stores:** Full episode data (messages, tool calls, signals), LoRA adapter checkpoints, exported training datasets, evaluation artifacts
- **Size:** ~5-50KB per episode. At 50k episodes/mo: ~1-2.5GB/mo. Cheap.
- **Retention:** Configurable per workspace. Default 90 days. Enterprise: unlimited.

#### 4. Message Queue (Redis / SQS)
- **Queues:**
  - `improvement` — episode batches for Reflector + Pareto + Playbook
  - `judge` — episodes needing judge evaluation
  - `training` — training job orchestration (Enterprise)
  - `events` — fan-out for SSE + webhooks
- **Scaling:** Redis to start. SQS if multi-region.

#### 5. Improvement Workers (CPU)
- **Logic:** Pull batch → Reflector → Pareto → Playbook Curator → gate_for_deploy → persist → notify
- **LLM dependency:** Anthropic/OpenAI API for Reflector + Paradigm Breakthrough
- **Compute:** 2 vCPU, 4GB RAM. 1-2 workers to start.

#### 6. Judge Workers (CPU + LLM API)
- **What:** LLM-as-judge on episodes where `needs_judge()` is True (~20% of episodes)
- **LLM dependency:** Anthropic/OpenAI API (we pay, baked into Pro+ subscription)
- **Compute:** 1 worker to start.

#### 7. Training Workers (GPU) — Enterprise only
- **What:** SkyRL GRPO training jobs (already integrated via `lfx/backends/skyrl.py`)
- **GPU:** 1x A100 40GB for 7B, 4x A100 for 70B
- **Provisioning:** On-demand (Modal, RunPod, cloud provider). Not always-on.
- **Note:** The GPU cost is <10% of the training engagement cost. The real cost is ML engineering time.

#### 8. Dashboard + SSE
- CDN-hosted static dashboard with auth
- SSE via Redis pub/sub → dedicated event workers
- Webhook delivery for Team+ (event → queue → HTTP POST to customer URL)

### External Dependencies

| Dependency | Purpose | Required? |
|-----------|---------|-----------|
| Anthropic API | Reflector, Judge, Paradigm Breakthrough | Yes (core) |
| OpenAI API | Alternative Reflector/Judge provider | Optional |
| Stripe | Billing (self-serve tiers) | Yes (Pro+) |
| SkyRL | Weights training backend | Enterprise only |
| Auth0 / Clerk | Auth + SSO | Yes (MVP: Clerk, Enterprise: + SAML) |
| Sentry | Error tracking | Yes |
| PostHog / Segment | Usage analytics | Nice to have |

### MVP vs. Full Build

#### MVP (6-8 weeks) — enough to convert design partners

- Single region (eu-west-1, Switzerland-adjacent)
- Core API: episodes, feedback, harness, gate, workspace, metrics
- API key auth (Clerk for dashboard login)
- Postgres + S3 + Redis
- 1-2 improvement workers + 1 judge worker
- Managed Reflector + Judge (Anthropic API)
- Regression gate (automatic)
- Dashboard (existing, adapted for persistence + auth)
- Stripe for self-serve billing
- Usage metering (episodes, judge calls)
- **No training API** (Enterprise handled manually for now)
- **No VPC/on-prem**
- **No SSO**

**Cloud cost baseline: ~$150-200/mo**

#### Full Build (3-6 months)

- Training project/run/adapter API
- Evaluation suite management
- Multi-region
- Org/team management, RBAC
- Webhook delivery
- VPC deployment (Terraform templates)
- SSO (SAML/OIDC)
- Audit log
- SLA monitoring
- Advanced dashboard (A/B, approval queue, savings report)

---

## What Changes in the Codebase

| Component | Current | Cloud adaptation |
|-----------|---------|-----------------|
| `lfx/server.py` | In-memory, single-user, 11 endpoints | → Multi-tenant, persisted, full REST API |
| `EpisodeCollector` | In-process, on_batch callback | → Option to POST to cloud API |
| `AsyncLearner` | Background thread, in-process | → Queue-based workers (separate process) |
| `Harness` state | In-memory dict | → Postgres (versioned, with rollback) |
| `Router` state | In-memory | → Postgres |
| `gate_for_deploy` | Called manually | → Automatic before every promotion |
| `lfx.wrap()` / `LfxCallback` | Sends to local collector | → Optional cloud_url parameter |
| Dashboard | Local static files | → CDN, auth, persistent data |
| Training (Enterprise) | Manual / SkyRL CLI | → API-driven project/run/adapter lifecycle |

**Key architectural decision:** The improvement engine (Reflector, Pareto, Playbook, Curator) stays the same code. Cloud wraps it in: auth → queue → worker → persist → version → gate → promote. The core learning logic does not change.

---

## Revenue Model by Channel

### Self-serve (n8n, SDK, litellm)

```
Install free connector → episodes flow to Cloud → free tier (500 eps)
                                                      │
                                             hits limit / wants judge
                                                      │
                                                      ▼
                                              Pro ($149/mo) + $0.03/ep overage
                                                      │
                                             multiple workflows / routing
                                                      │
                                                      ▼
                                              Team ($499/mo) + $0.02/ep overage
```

### Enterprise (direct sales)

```
Outreach / "Agent Rescue" pilot → workflow assessment (CHF 10-25k)
                                         │
                                  harness improvement (Platform CHF 999+/mo + usage)
                                         │
                                  model training engagement (CHF 25-55k initial)
                                         │
                                  ongoing retraining retainer (CHF 5-50k/mo)
```

---

## Open Questions for Cloud Expert

1. **Hosting:** AWS eu-west-1 for MVP? Switzerland (eu-central-2) if data residency matters?
2. **GPU provisioning:** Modal (serverless, pay-per-second) vs. RunPod vs. cloud provider on-demand?
3. **Auth:** Clerk for MVP? Timeline to add SAML/OIDC for enterprise?
4. **Deployment:** Kubernetes (EKS), ECS, or simpler (Railway/Fly.io) for MVP?
5. **SSE scaling:** Redis pub/sub → SSE workers, or switch to managed WebSocket (Ably/Pusher)?
6. **Per-tenant cost tracking:** How to attribute LLM API costs per workspace accurately?
7. **Data residency:** Do we need EU-only data processing guarantees from day one?
8. **Compliance:** SOC2 timeline? What's the minimum viable compliance stance for enterprise pilots?

---

## Summary

Aganthos Cloud is two products on one platform:

1. **Managed harness improvement** — self-serve, automated, $149-1,499/mo + usage. The n8n node, SDK, and litellm callback are free on-ramps.

2. **Managed model training ("white-glove RL")** — enterprise, high-touch, CHF 5,000-50,000/month ongoing. The real cost is ML engineering expertise, not GPU compute. This is our highest-value offering and the equivalent of Nebius's white-glove RFT program and Adaptive ML's FDE engagement — but focused on the full learning-from-experience loop, not just model training.

Both use the same API, the same trace format, the same improvement engine. The difference is depth of improvement and level of service.

**Pricing tiers:** Free → Pro ($149) → Team ($499) → Scale ($1,499) → Enterprise (CHF 5k-50k/mo)

**Open source strategy:** Source-available lfx repo (BSL license) for community trust and experimentation. Aganthos Cloud (private repo, imports lfx) for production and revenue. Same core improvement engine, different deployment model.

**n8n GTM:** Free community node → proves loop works → converts to self-serve paid → enterprise upsell for model training. n8n users are the ideal ICP: workflow builders who deploy agents but don't want to build an improvement function.

**Competitive positioning:** We're not raw compute (RunRL) and not a generic FDE shop (Adaptive ML). We're the learning-from-experience lab that operates the full improvement loop — harness, routing, and model training — from production traces. No competitor offers the self-serve harness improvement tier.

Every connector is free. Revenue is on the improvement layer behind them.
