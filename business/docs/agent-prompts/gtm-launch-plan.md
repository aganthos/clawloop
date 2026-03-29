# Agent Prompt: GTM Launch Plan

Create a go-to-market launch plan for ClawLoop's open-source release, including pricing strategy.

Working directory: /Users/robertmueller/Desktop/aganthos
Output: write the plan to docs/plans/2026-03-29-gtm-launch-plan.md (this is gitignored, local only)

## Context

- Product: ClawLoop — unified learning API for AI agents (pip install clawloop)
- License: BSL 1.1 (→ Apache 2.0 after 4 years). Revenue threshold: $10M for production use.
- Company: Aganthos GmbH, Swiss startup, Berkeley SkyDeck Batch 22
- Repo: github.com/aganthos/clawloop (currently PRIVATE, ready to flip public)
- Version: v0.0.1

Key differentiators:
- Only agent learning system with 3 unified layers (Harness, Router, Weights) under one API
- Atomic rollback + regression gating across all layers
- Competitors: MetaClaw (no conflict detection, no pruning, no rollback), Honcho (memory only), GUM (no learning loop)
- Proof points: 7B model outperforming o4-mini by 30% at 90% cost reduction (German university hospital)
- Two ICML 2026 submissions

## Files to read for context
- README.md (the public-facing README)
- pitch/pitch-scripts.md (1-min, 3-min, 8-min pitches)
- pitch/main_pitch.html or pitch/main_pitch_standalone.html (slides — search for pricing, tiers, editions)
- docs/plans/2026-03-20-next-steps-roadmap.md (GTM section + pricing tiers)
- ressources/overview-investor.md
- CHANGELOG.md
- pyproject.toml (current version, description)

## What to produce

Write a comprehensive, actionable launch plan as a single markdown document.

### 1. Pre-launch checklist
What must be polished before flipping to public:
- README quality (read it, suggest specific improvements)
- Examples quality (read examples/, are they easy to run?)
- GitHub repo presentation (topics, description, social preview image dimensions)
- Missing files? (CODE_OF_CONDUCT.md? SECURITY.md?)
- Is `pip install clawloop` ready? Check pyproject.toml build config.
- Demo GIF needed? What should it show?

### 2. Pricing strategy
Read the pitch slides and next-steps roadmap for existing pricing thinking. Then:
- Define tiers: Free / Pro / Team / Scale / Enterprise
- What's in each tier? Map to community vs cloud features
- Price points (compare to competitors: MetaClaw/Tinker ~$10/LoRA, Langfuse Cloud $8/100K traces, LangSmith pricing)
- What justifies Pro at $149/mo? (measurable improvement + dashboard)
- Usage-based vs seat-based vs flat pricing?
- Free tier limits (iterations/month? episodes/month? agents?)
- Enterprise pricing model (custom, based on what?)
- The BSL revenue threshold ($10M) — how does this interact with pricing tiers?

### 3. Launch timing
- Best day/time: research suggests Tuesday/Wednesday 8:00 AM PT for HN + cross-platform
- Myriade analysis of 157K Show HN posts found weekends outperform for Show HN specifically
- Recommend specific dates for the next 2 weeks
- Time zones: Switzerland-based, targeting US tech audience

### 4. Channel strategy with messaging
For EACH channel, write actual DRAFT copy (not just "write a tweet"):
- HackerNews Show HN: exact title (<80 chars) + first comment text
- Reddit r/MachineLearning [P] tag: title + body
- Reddit r/LocalLLaMA: title + body
- Twitter/X: full thread (5 tweets with specific content)
- LinkedIn: post text
- n8n community: post
- dev.to: article outline with section headers

### 5. BSL license positioning
BSL will get pushback. Write:
- Exact README language for the license section
- Prepared responses for "this isn't open source"
- How Sentry (FSL), CockroachDB ($10M threshold), MariaDB (BSL originator) handle this
- Key message: "source-available, free for orgs under $10M, Apache 2.0 after 4 years"

### 6. Launch day sequence
Hour-by-hour plan for launch day (in both PT and CET).

### 7. Post-launch week 1-2
- Monitoring tools and what to track
- How to respond to feedback
- Follow-up content cadence

### 8. Success metrics
- Week 1 targets (stars, downloads, HN points)
- Month 1 targets
- What defines success for v0.0.1?

## Rules
- Do NOT modify any code files
- Write ONLY to docs/plans/2026-03-29-gtm-launch-plan.md
- Include SPECIFIC, actionable draft copy — not "write something about X"
- Commit: `chore: add GTM launch plan with pricing strategy`
