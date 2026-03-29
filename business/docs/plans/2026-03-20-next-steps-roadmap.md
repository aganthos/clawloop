# Next Steps Roadmap

**Date**: 2026-03-20 (rev4)
**Goal**: Launch community edition, build cloud pro, compete in AgentBeats
**Codex review**: SHIP IT (3 rounds)
**Architecture diagram**: `lfx/static/lfx-architecture.html`

---

## Community vs Cloud: The Three-Layer Split

All three learning layers ship in both editions. Community is fully functional locally.
Cloud adds managed infra, persistence, team features, and scale-only algorithms.

### Harness (Prompt + Playbook Memory)

| Capability | Community | Cloud Pro |
|------------|-----------|-----------|
| Reflector (LLM trace → insights) | BYO API key | Managed |
| Playbook (structured memory) | Full, in-memory | Persisted, versioned, rollback |
| Curator (conflict/redundancy) | Heuristic (cosine thresholds) | + LLM classify + auto-resolve |
| GEPA (Pareto prompt evolution) | Full: archive + mutation + crossover | Same + managed compute |
| Consolidation ("dreaming") | Manual trigger | Scheduled background |
| Attribution | Tag match + embedding | + LLM attribution |
| Approval workflow | Auto-apply | Pending → approve/reject |

### Router (Model Selection)

| Capability | Community | Cloud Pro |
|------------|-----------|-----------|
| Tier definitions | Manual config | + auto-discovered |
| Score tracking | In-memory | Persisted, historical |
| Cost optimization | Metrics dict | Dashboard: "saved $X/month" |
| Cross-agent routing | N/A | Shared policy across workspace |

### Weights (Training)

| Capability | Community | Cloud Pro |
|------------|-----------|-----------|
| Training algos | GRPO + custom (self-hosted GPU via SkyRL/Tinker) | Managed GPU workers |
| gate_for_deploy | Manual CLI | Automatic before promotion |
| Checkpoints | Local filesystem | S3-backed, versioned |

### Cloud-only (no local equivalent)

Persistence, multi-tenancy, API key auth, hidden eval sets, A/B testing,
cross-tenant meta-learning, audit trail, billing.

### Upsell Funnel

```
Free:       pip install lfx → lfx.wrap(client) → learns locally
Pro $149:   + Persisted, versioned, approval workflow, team access
Team $499:  + Router cost dashboard, cross-agent learning
Scale $1.5k: + A/B testing, hidden eval gates, meta-learning seeds
Enterprise: + Managed training, VPC, SSO, SLA
```

---

## WP1: Community Release

**Agent prompt**: `docs/plans/2026-03-20-pr3a-community-release-agent-prompt.md`
**Branch**: `chore/community-release-v0.1`

8 tasks, ~1 day: License (BSL 1.1), README.md, pyproject.toml polish, `__init__.py` exports,
`lfx-server` script, `quick_start()`, `learning_loop()` default, packaging smoke test.

No examples dir, no CI/CD, no docs site. Ship and iterate.

---

## WP2: AgentBeats Competition

**Agent prompt**: `docs/plans/2026-03-21-wp2-agentbeats-agent-prompt.md`
**Branch**: merge `feat/entropic-crmarena-bench`, then training runs on main

Tasks: merge CRM Arena branch → init submodule → smoke training (10 iter) → analyze →
full training (50-100 iter) → export frozen prompt+playbook → baseline comparison →
register on agentbeats.dev.

Reproducibility: every run captures config.json, experiment.jsonl, playbook.json,
best_prompt.txt, git_hash.txt, manifest.json (SHA256). Gate: >5% improvement OR
non-worse at 95% CI.

Sprint targets: Sprint 1 Business Process (CRM), Sprint 2 τ²-bench, Sprint 4 General Purpose.

---

## WP3: Cloud Pro v0

**Agent prompt**: `docs/plans/2026-03-21-wp3-cloud-pro-agent-prompt.md`
**Repo**: `aganthos-cloud` (private), pins `lfx==0.1.0`

Thin MVP: 10 endpoints (episodes CRUD + harness versioning + approval workflow) + API key auth.
Deferred: billing (manual invoicing), dashboard (reuse community), cross-tenant learning.

Security: key ID + HMAC hash, TLS, rate limiting 100 req/s, audit log, encryption at rest,
strict workspace isolation.

Cloud-only algorithms (post-MVP): Meta-Reflector v0.3, A/B Gate v0.3, Router dashboard v0.2,
Cross-agent transfer v0.2, Managed PRM v0.4.

One change in lfx community: `cloud_url` + `cloud_api_key` params on `wrap()` / `LfxCallback`.

---

## GTM (Go-to-Market)

### Positioning

**lfx is the only agent learning system with all three layers** (prompt memory, model routing,
weight training) **unified under one API with atomic rollback and regression gating.**

MetaClaw has prompt+weights but no routing, no conflict detection, no pruning, no rollback.
Honcho has memory but no learning. GUM has propositions but no agent improvement loop.

### Launch Channels

| Channel | Action | Timing |
|---------|--------|--------|
| **GitHub** | Public repo with README, architecture diagram, BSL license | WP1 |
| **AgentBeats** | Competition submission — "agent that learns from experience" | WP2 |
| **HackerNews** | "Show HN: lfx — your AI agent learns from its mistakes" | After WP1 |
| **Twitter/X** | Architecture diagram + "3 layers of learning" thread | After WP1 |
| **n8n community** | "Make your n8n AI workflows learn" — workflow template + tutorial | After WP1 |
| **arXiv** | Technical paper on the three-layer protocol + competitive benchmarks | After Sprint 4 |
| **YC/investors** | AgentBeats results + cloud traction as proof points | After WP3 |

### Key Messages

1. **For developers**: "pip install lfx, wrap your client, your agent gets better over time"
2. **For teams**: "Never lose a playbook. Roll back bad changes. See what your agent learned."
3. **For enterprises**: "We fine-tuned a 7B to match GPT-4o on your domain. 90% cost reduction."

### Content

| Asset | Purpose | Where |
|-------|---------|-------|
| Architecture diagram | Visual explainer for README, tweets, decks | `lfx/static/lfx-architecture.html` |
| Competition results | Credibility — "lfx-tuned agent beat baseline by X%" | AgentBeats leaderboard |
| Benchmark comparison | lfx vs MetaClaw vs Honcho feature matrix | README + blog post |
| n8n workflow template | Zero-code entry point for non-developers | `n8n-workflows/` |
| Quick start video | 2-min "wrap → learn → improve" demo | YouTube/Twitter |

---

## Competitive Intel (2026-03-21)

MetaClaw at v0.3.2, no changes. All five gaps open: no skill pruning, no conflict detection,
no per-skill scoring, PRM black box, SkillEvolver buggy (fixes Mar 17+20). Skill evolution
fires only at <40% performance. All RL via proprietary Tinker Cloud (vendor lock-in).

lfx structural advantages (all shipped in PR1+PR2):
1. Active conflict/redundancy resolution (Curator)
2. Per-entry attribution with relevance threshold
3. Atomic rollback + regression gating
4. Generation flush + temporal decay (MetaClaw + GUM approaches combined)
5. Three independent layers vs MetaClaw's two conflated mechanisms
6. Multi-objective reward composition vs single PRM
7. Self-hosted training (SkyRL/Harbor) — no vendor lock-in

---

## Open Questions

1. License: BSL 1.1, 4-year change date → Apache 2.0?
2. Hosting: AWS eu-west-1? Hetzner? Railway for MVP?
3. CRM arena: which LLM for purple agent? which for reflector?
4. Online learning during competition — allowed?
5. Custom training algo timeline?
6. First cloud customer: n8n user? Internal dogfood?
