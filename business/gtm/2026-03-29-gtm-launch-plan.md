# ClawLoop GTM Launch Plan

**Date**: 2026-03-29 (rev2 — co-planned with Codex, 3 rounds)
**Version**: v0.0.1
**Repo**: github.com/aganthos/clawloop (currently private)
**Entity**: Aganthos Inc. (Delaware C-Corp)
**License**: BSL 1.1 → Apache 2.0 on 2030-04-01

---

## 0. Co-Planning Summary (Claude + Codex, 3 rounds)

This plan was developed through 3 rounds of adversarial co-planning between
Claude and OpenAI Codex. Key debates and where we landed:

### Agreements
- **BSL over Apache 2.0** — both agree. Launching BSL from day one is honest;
  switching later (HashiCorp, Redis) is what causes community anger.
- **Per-inference + basis points pricing** — not flat-fee, not seat-based.
  Revenue must scale with value delivered, not just cost incurred.
- **Two trajectories** — self-serve (base + per-inference) and enterprise
  (platform fee + basis points on LLM spend). Flat fees leave all upside on table.
- **Reverse trial for Pro** — 30 days of Pro, then downgrade to Community.
  Users feel the loss of persistence/dashboard.
- **Publish pricing page at launch** — internal inconsistency kills trust.
- **6-month license review commitment** — publicly state you'll re-evaluate
  BSL terms based on adoption data. Defuses HN criticism.

### Disagreements (decisions needed)

| Decision | Option A (Claude) | Option B (Codex) | Recommendation |
|----------|-------------------|-------------------|----------------|
| **BSL threshold** | $10M (CockroachDB precedent, wider funnel) | $5M (earlier monetization, cleaner) | **$10M** — adoption > threshold revenue at v0.0.1. Can lower later; raising is politically hard. |
| **Pricing model** | $149/mo flat | $99–$249 range + credit-based | **Per-inference + basis points** — flat fees don't scale with value. Self-serve: $49–$99 base + $0.001–$0.002/inference. Enterprise: platform fee + 5–15% of LLM spend. See Section 2. |
| **Launch strategy** | Multi-channel same day | Benchmark-led + HN amplifier only | **Benchmark-led IF results ready within 3 weeks; otherwise hospital case study + HN** (see Section 3) |
| **Channel spread** | All 7 channels day 1 | Focus: benchmark → HN → enterprise outbound | **3-channel focus day 1** (HN + Twitter + LinkedIn), stagger others over week 1 |
| **Proof points** | 1 case study enough for v0.0.1 | Need 2–3 before launch | **Launch with 1, promise benchmark follow-up in 30 days** |

### Alternative Options (park for later)

These ideas surfaced during co-planning but are too risky/complex for v0.0.1:

1. **Outcome-based Pro** ("pay $0 until 5% improvement"): Bold story but
   measurement disputes kill it at scale. **Use as limited design-partner
   program for 3 hand-picked teams only.**
2. **"Pay what your agent saves" (10% of savings)**: Anchors value to cost
   reduction rather than quality improvement — wrong signal for enterprise.
3. **Star-gated free Pro** ("star + badge = 1yr free cloud"): Feels spammy
   to enterprise buyers. Could work for a dev-tool, not for infra.
4. **Progressive BSL brackets** ($0–$5M free / $5–$25M Pro / $25M+ Enterprise):
   Fair concept but adds legal complexity. Park until v0.2.
5. **Anti-launch (quiet flip)**: Viable only if you already have 2–3 design
   partners locked. For pre-seed, visible traction matters.
6. **Apache 2.0 instead of BSL**: Eliminates all license friction, but
   the cloud moat isn't strong enough yet at v0.0.1 to survive without
   code protection. Re-evaluate at 6-month review.

---

## 1. Pre-Launch Checklist

### README.md — Issues to Fix

The current README is functional but not launch-ready. Specific improvements:

| Issue | Current State | Fix |
|-------|--------------|-----|
| **Install command** | `pip install -e .` (dev mode) | Change to `pip install clawloop` — that's what users will actually run |
| **No badge strip** | Missing | Add: PyPI version, Python 3.11+, License BSL-1.1, GitHub stars, test status |
| **No one-liner hook** | Jumps straight to "Install" | Add a tagline under the logo: *"Your AI agent learns from its mistakes. Three learning layers — prompt, routing, weights — one API."* |
| **No `clawloop.wrap()` quickstart** | Quickstart shows `train_runner.py` | Lead with the 5-line wrap example — this is the "aha moment" for developers |
| **Missing "Why ClawLoop?"** section | Architecture section is too internal | Add a benefits-first section before architecture |
| **No comparison table** | Competitors mentioned nowhere | Add: ClawLoop vs MetaClaw vs Honcho vs building-from-scratch |
| **Old package reference** | SkyRL install: `pip install -e clawloop/skyrl[fsdp]` | Verify this path works from a clean clone |
| **No link to CHANGELOG** | Missing | Add link in footer |
| **No "Star this repo" CTA** | Missing | Add at bottom — every star compounds discoverability |
| **Brand inconsistency** | README says "Tinker/SkyRL compatible" at top | Clarify relationship: SkyRL is the training backend, Tinker is the cookbook format |

**Suggested README structure (top to bottom):**
1. Logo + tagline + badges
2. 5-line `wrap()` quickstart (the hook)
3. "Why ClawLoop?" (3 bullets: improve, reduce cost, rollback safety)
4. Full quickstart (harness, weight, openclaw)
5. Feature comparison table (vs MetaClaw, Honcho, GUM, scratch)
6. Environments table (existing — good)
7. LLM providers (existing — good)
8. Architecture (existing — trim)
9. Adding new environments (existing — good)
10. License + Contributing + Star CTA

### Examples Quality

**Verdict: Good.** The examples/ directory is well-structured:
- Unified runner (`train_runner.py`) with JSON configs is clean
- 8 config files covering all 4 environments × 2 modes
- `openclaw_proxy_demo.py` is fully annotated
- Recipes directory has standalone training scripts
- Clear README with test coverage matrix

**Issues to fix:**
- ~~`examples/README.md` still references `lfx`~~ — DONE, rename completed
- Add a `# Prerequisites` section to each config explaining what's needed (API key? Docker? GPU?)
- `openclaw_runner/` has `node_modules/` checked in (showing in git status) — add to `.gitignore`

### GitHub Repo Presentation

Before flipping public:

- [ ] **Repository description**: "Learning from Experience — unified learning API for AI agents. Prompt optimization, model routing, weight training. One API."
- [ ] **Topics**: `ai-agents`, `reinforcement-learning`, `llm`, `prompt-optimization`, `model-routing`, `lora`, `grpo`, `agent-learning`, `python`
- [ ] **Social preview image**: 1280×640px. Use the ClawLoop logo on a dark gradient with the tagline. Generate via Figma or Canva.
- [ ] **Website field**: `https://aganthos.com` (or docs URL when ready)
- [ ] **Disable wiki** (unused, confusing)
- [ ] **Enable Discussions** (channels: General, Ideas, Q&A, Show and Tell)
- [ ] **Pin an issue**: "Roadmap" with checkboxes for upcoming features

### Missing Files

| File | Status | Action |
|------|--------|--------|
| `LICENSE` | Exists, BSL 1.1, correct | No action |
| `CONTRIBUTING.md` | Exists, adequate | Add DCO/CLA note if desired |
| `CHANGELOG.md` | Exists, v0.0.1 | No action |
| `CODE_OF_CONDUCT.md` | **Missing** | Add — use Contributor Covenant v2.1 (industry standard) |
| `SECURITY.md` | **Missing** | Add — include responsible disclosure email (security@aganthos.com) |
| `.github/ISSUE_TEMPLATE/` | **Missing** | Add bug report + feature request templates |
| `.github/PULL_REQUEST_TEMPLATE.md` | **Missing** | Add basic PR template |
| `.github/workflows/` | **Missing** | Add CI: pytest on push/PR (Python 3.11 + 3.12) |
| `.github/FUNDING.yml` | **Missing** | Optional — add if you want GitHub Sponsors |

### pyproject.toml Build Config

**Verdict: Ready for PyPI.**
- Build system: hatchling (good)
- `name = "clawloop"`, `version = "0.0.1"` — correct
- Dependencies: `litellm>=1.0`, `pydantic>=2.0` — minimal, good
- Optional extras: `dev`, `car`, `otel`, `server` — well-structured
- Entry points: `clawloop` CLI + `clawloop-server` — both defined
- sdist includes: clawloop/, tests/, examples/, LICENSE, README.md, CONTRIBUTING.md, CHANGELOG.md
- Wheel packages: `["clawloop"]` only — correct

**Entity**: `authors = [{name = "Aganthos Inc."}]` — correct. Aganthos Inc.
(Delaware C-Corp) is the parent and licensor. Aganthos GmbH (Swiss) is a 100% subsidiary.

**Before publishing**: Run `pip install build && python -m build && pip install dist/clawloop-0.0.1-py3-none-any.whl` in a clean venv to smoke-test the package.

### Demo GIF

**Yes, needed.** A terminal GIF is the single highest-ROI asset for GitHub README and HN.

**What to show (30 seconds):**
1. `pip install clawloop` (2 sec)
2. Paste 5-line `wrap()` code (3 sec)
3. Run a math harness learning session (10 sec) — show the reflector improving
4. Show playbook entries being added (5 sec)
5. Show iteration scores improving (5 sec)
6. Final: "Agent improved from 40% → 72% in 5 iterations" (5 sec)

**Tool**: Use [asciinema](https://asciinema.org/) + [agg](https://github.com/asciinema/agg) to render to GIF. Or [vhs](https://github.com/charmbracelet/vhs) for scripted recordings.

---


## 2. Pricing Strategy

### Why Every Prior Attempt Was Wrong

Five pricing iterations all failed the "Cursor test":

| Attempt | Model | Failure |
|---------|-------|---------|
| Roadmap (Mar 20) | Flat $149–$1.5K/mo | Yields ~$0 at platform scale |
| Investor overview | Flat $49–$199/mo | Same |
| Pitch slides | Enterprise fixed + usage | "Usage" undefined |
| GTM v1 | Per-inference $0.001/call | Gameable: extract playbook, bypass meter |
| GTM v2 | Basis points on LLM spend | Misaligned: we earn less when routing saves money |

**Root cause:** All models meter cost (episodes, traces) or throughput
(inferences, LLM spend). Neither meters VALUE (improvement delta).

**Market research (Codex, web search):** Nobody in AI/ML infra has shipped
outcome-based pricing. Langfuse ($8/100K traces), LangSmith ($2.50/1K traces),
Braintrust ($249/mo + usage), W&B ($60/mo + hours), Tinker ($0.40/1M tokens) —
all bill usage. Outcome pricing is aspirational everywhere.

**But:** Pricing an OPTIMIZER like a LOGGER is a category error.

### The Architecture: Phased Pricing Evolution

```
Phase 1 (launch, 0-6mo)       Phase 2 (6-12mo)            Phase 3 (12mo+)
─────────────────────         ─────────────────           ─────────────────
Usage + savings share         + improvement events        Outcome-aligned
(market-legible,              (opt-in for trusted         (value = primary
get adoption)                 partners)                   meter)
```

**Pricing page headline:**
> "Pay for what flows through ClawLoop. When routing saves you money, we share the savings."

### Phase 1: Launch Pricing (v0.0.1)

Two meters. Simple, market-legible, hard to game.

#### Meter 1 — Per-Inference (usage floor)

Every LLM call through `wrap()` or the proxy.

| Variant | Rate | What it covers |
|---------|------|----------------|
| **BYOK** (user's keys) | **$0.0005/inference** | Playbook retrieval + injection, routing, persistence |
| **Managed** (Aganthos LLM) | **$0.001/inference** | + managed reflector + judge |

#### Meter 2 — Routing Savings Share (value capture)

Router tracks savings from routing to cheaper models. **20% of verified savings.**

This is the ONLY outcome-based meter at launch — routing savings are objective,
automated, indisputable. No eval set, no measurement contract, no disputes.

**Example:** Router shifts 40% of traffic from GPT-4o ($15/1M tok) to Haiku
($0.25/1M tok). On 1M tokens/mo: saves ~$5,900. ClawLoop takes 20% = $1,180.
Customer keeps $4,720. Self-funding.

#### Community — Free Forever (local only)

All three layers, fully functional, no limits, no meters.
- Harness, Router, Weights, GEPA, rollback, regression gating
- `clawloop.wrap()`, `clawloop-server`, OTel export

**BSL**: Free for orgs under $10M that aren't building a competing service.

#### Pro Cloud — Per-Inference + Savings Share

30-day reverse trial. No credit card.

| Component | BYOK | Managed |
|-----------|------|---------|
| Per-inference | $0.0005/call | $0.001/call |
| Routing savings | 20% | 20% |
| Dashboard + persistence | Included | Included |
| Managed reflector/judge | No (BYO keys) | Yes |
| Workspace / agents | 1 / 5 | 1 / 10 |

**What customers actually pay:**

| Customer type | Inferences/mo | Inference fee | Savings share (20%) | **Total** |
|---------------|---------------|---------------|--------------------|-----------|
| Hobby | 5K | $2.50 | $0 | **$2.50** |
| Solo dev | 50K | $25 | $50 | **$75** |
| Small team | 500K | $250 | $500 | **$750** |
| Growing team | 2M | $2,000 | $3,000 | **$5,000** |
| Scale (50 agents) | 20M | $10,000 | $15,000 | **$25,000** |

**Margins:**
- BYOK inference: cost ~$0.0001/call → **~80% margin**
- Managed inference: cost ~$0.0003/call → **~70% margin**
- Routing savings share: cost $0 → **100% margin**

**Volume tiers:**

| Volume | BYOK | Managed |
|--------|------|---------|
| 0–5M/mo | $0.0005 | $0.001 |
| 5M–50M/mo | $0.0003 | $0.0007 |
| 50M–500M/mo | $0.00015 | $0.0004 |
| 500M+/mo | Custom → Enterprise | Custom |

#### Enterprise — Platform + Engagements

**Platform component:**

| Component | Price |
|-----------|-------|
| Platform fee | $10K–$50K/mo (SLA, SSO, VPC, audit, support) |
| Per-inference | Volume-discounted |
| Routing savings | 15% (enterprise discount) |

**Training engagements (Modes 3+4) — the real enterprise revenue:**

| Component | Price |
|-----------|-------|
| Engagement fee | **$100K/month × 3–5 months** ($300K–$500K total) |
| Scope | Reward design, data pipeline, training, eval, QA, deployment gating |
| Deliverable | Trained model + eval harness + deployment playbook |
| Retraining retainer | **$25K–$100K/mo** (ongoing) |
| Model royalty (optional) | **1–3% of inference on trained model**, or buyout (12–18mo) |

Training is NOT commodity compute. It's senior ML engineering. GPU is <10%
of cost. This is Adaptive ML territory ($200K-$500K/yr for embedded
engineers), not Tinker Cloud ($0.40/1M tokens).

**Environment synthesis (Mode 5):**

| Component | Price |
|-----------|-------|
| Env construction | $50K–$150K per environment |
| Env licensing to labs | Revenue share (future, post-seed) |

### Phase 2: Improvement Events (6–12 months, opt-in)

Add a third meter for trusted customers: **Gated Improvement Events (GIE)**.

When ClawLoop produces a playbook entry or routing change that passes the
customer's regression gate and deploys to production = billable event.

**Value-indexed pricing** (not flat fee):
- Improvement Fee = ΔMetric × Value-Per-Point × Share (20–30%)
- Value-Per-Point agreed during onboarding
- Only for customers who opt in with defined eval metrics
- Routing savings already live from Phase 1; extend to accuracy/CSAT/conversion

**Why opt-in:** The measurement contract is the hardest part. Phase 2 pilots
with partners who trust us. Phase 3 makes it standard.

### Phase 3: Outcome-Aligned Enterprise (12+ months)

Usage fee shrinks to platform-access minimum. Improvement events become
primary revenue. Training engagements + env synthesis + lab licensing as
additional streams. This is the long-term vision.

### Full Product Surface: 5 Learning Modes × Pricing

```
                        DATA SOURCE
                   Traces          Environment
               ┌──────────────┬──────────────────┐
  Prompt       │ 1. Harness   │ 2. Harness from  │
  (Harness)    │    traces    │    environment    │
               ├──────────────┼──────────────────┤
  Weights      │ 3. Weights   │ 4. Weights from  │
  (Training)   │    traces    │    environment    │
               ├──────────────┼──────────────────┤
  Environment  │ 5. Build env │ (n/a)            │
  (Synthesis)  │    from traces│                  │
               └──────────────┴──────────────────┘
```

**Mode 1 — Harness from traces** (core loop, no GPU):
Reflector analyzes production traces → playbook entries → improved prompt.
*Pricing:* Per-inference + savings share. Bread and butter.

**Mode 2 — Harness from environment** (pre-production):
Run agent in sandbox (Harbor, CRM Arena). Accelerate learning before deploy.
*Pricing:* Per-environment-run fee. Higher per-unit, concentrated value.

**Mode 3 — Weights from traces** ($300K–$500K engagement):
Distill knowledge into model weights from production traces. 7B → beats o4-mini.
*Pricing:* Enterprise engagement. ML engineering, not commodity compute.

**Mode 4 — Weights from environment** ($300K–$500K engagement):
Full weight training in sandbox. Highest ceiling. Hospital result used this mode.
*Pricing:* Same as Mode 3, often combined (traces first, then env refinement).

**Mode 5 — Env synthesis from traces** ($50K–$150K):
Build simulated envs from production data. Turns Mode 3 → Mode 4 (unlimited training).
*Pricing:* Enterprise add-on. Future: sell anonymized envs to labs (separate biz).

**Mode-to-tier mapping:**

| Mode | Community | Pro Cloud | Enterprise |
|------|-----------|-----------|------------|
| 1. Harness from traces | Free, local | Per-inference + savings | Platform + savings |
| 2. Harness from environment | Free, local | Per-env-run | Included in platform |
| 3. Weights from traces | Free (BYO GPU) | Not available | $100K/mo × 3–5mo |
| 4. Weights from environment | Free (BYO GPU) | Not available | $100K/mo × 3–5mo |
| 5. Env synthesis | Not available | Not available | $50K–$150K |

**Customer journey (each step = higher ACV):**

```
Mode 1 ($75/mo) → Mode 2 ($500/mo) → Mode 3+4 ($300K–$500K) → Mode 5 ($150K)
   self-serve        pre-prod            enterprise                add-on
```

### Market Positioning

| Competitor | Model | Rate | ClawLoop difference |
|-----------|-------|------|-------------------|
| Langfuse | Per-trace | $8/100K traces | We meter improvement, not observation |
| LangSmith | Per-trace + seat | $2.50/1K + $39/seat | Per-inference + savings share aligns to value |
| Braintrust | Spans + scores | $249/mo + usage | We add learning, not just evaluation |
| Tinker | Per-token training | $0.40/1M tokens (8B) | Training is $100K/mo engagement, not commodity |
| Honeyhive | Free + enterprise | Custom | We scale from $2.50/mo to $500K/mo |
| W&B | Base + hours | $60/mo + $1/hr | Different category (experiment tracking) |

### Revenue Concentration by Mode (Projected)

| Mode | Year 1 | Year 2 | Year 3 |
|------|--------|--------|--------|
| 1. Harness inference + savings | 75% | 35% | 15% |
| 2. Harness from environment | 5% | 10% | 5% |
| 3+4. Training engagements | 20% | 35% | 45% |
| 5. Env synthesis + lab licensing | 0% | 20% | 35% |

Year 1 = self-serve per-inference revenue dominates. Year 2+ = enterprise
training engagements + env synthesis take over as the growth engine.

### BSL Revenue Threshold

**Recommended: $10M** (DISAGREEMENT: Codex recommended $5M)

| Threshold | Pros | Cons |
|-----------|------|------|
| **$5M** (Codex) | Earlier monetization, cleaner enforcement | Catches Series A startups ($3-8M) who are your best evangelists |
| **$10M** (recommended) | CockroachDB precedent, ~99% of companies free, wider funnel | Well-funded startups deploy free longer |
| **$25M** | Maximum community goodwill | Delays monetization significantly |
| **Progressive** ($5M/$25M brackets) | Fairest | Legal complexity, compliance disputes |

**Decision: $10M with 6-month public review.** Adoption matters more than threshold
revenue at v0.0.1. You can lower the threshold later based on data; raising it
after launch is politically impossible. Publicly commit to reviewing the threshold
at 6 months based on adoption and conversion data — this defuses "will you lower
it?" anxiety.

**How threshold maps to tiers:**

| Organization Revenue | Community | Pro | Enterprise |
|---------------------|-----------|-----|------------|
| Under $10M | Free, no restrictions | Optional cloud (per-inference) | N/A |
| Over $10M | Requires commercial license | Any paid plan satisfies license | Enterprise contract |
| Competing offering | Not permitted under BSL | Not permitted under BSL | Custom agreement required |

**Key message**: "Under $10M revenue? Use everything for free, forever. Over $10M? Any paid plan includes a commercial license."

Per-inference pricing makes the commercial license transition natural: as companies
grow past $10M, they're already paying per-inference and the commercial license
is bundled. No cliff, no awkward sales conversation.

---

## 3. Launch Strategy & Timing

### Strategy: Benchmark-Led Launch (co-planned with Codex)

**Core insight from co-planning**: Don't announce ClawLoop as a product — announce
it as a RESULT. The product announcement lives inside the benchmark story.

This is how Cursor (speed benchmark), Bun (performance benchmark), and many
successful dev tools launched. A result is 10x more shareable than a product pitch.

**Two paths depending on AgentBeats timing:**

**Path A — Benchmark ready within 3 weeks (preferred):**
1. Run AgentBeats CRM Arena benchmark → get result
2. Write benchmark blog post: "We benchmarked 5 agent learning systems. Here's what happened."
3. Launch Show HN with the benchmark as the headline
4. ClawLoop is the tool that produced the result — the product pitch is inside the post
5. Enterprise outbound uses the benchmark as the email opener

**Path B — Benchmark not ready (fallback, still strong):**
1. Launch with hospital case study as the headline proof point
2. Promise benchmark results in 30 days as follow-up content
3. Post Show HN + 3 focused channels
4. Enterprise outbound uses hospital case study

**DISAGREEMENT**: Claude originally proposed 7 channels on day 1. Codex argued
this spreads a 3-person team thin — every channel gets 70% effort. Final decision:
**3 focused channels on day 1** (Show HN + Twitter + LinkedIn), stagger Reddit,
n8n, dev.to over week 1.

### Channel Priority (day 1 vs staggered)

| Priority | Channel | Day |
|----------|---------|-----|
| 1 | **Show HN** (primary — the one bet that matters) | Day 1 |
| 2 | **Twitter/X thread** (amplifier) | Day 1 |
| 3 | **LinkedIn** (enterprise signal) | Day 1 |
| 4 | **Enterprise outbound** (50 personalized emails) | Day 1–3 |
| 5 | r/MachineLearning | Day 3 |
| 6 | r/LocalLLaMA | Day 3 |
| 7 | n8n community | Day 5 |
| 8 | dev.to article | Day 7 |

### Optimal Launch Windows

**Research synthesis:**
- General HN submissions: Tuesday–Wednesday, 8:00–9:00 AM PT performs best
- Show HN specifically: Myriade's analysis of 157K posts found weekends (Saturday AM) can outperform due to less competition
- Cross-platform amplification (Twitter, Reddit, LinkedIn) works best on weekday mornings
- Avoid: Mondays (inbox overload), Fridays (weekend dropoff), major tech announcements

**Switzerland timing consideration**: 8:00 AM PT = 5:00 PM CET. This works well — launch in the late Swiss afternoon, monitor through the evening as US engagement peaks.

### Recommended Dates

**Option A — Benchmark ready (preferred):**
- **Wednesday, April 16, 2026** — 8:00 AM PT / 5:00 PM CET
- Gives ~2.5 weeks to run AgentBeats benchmark and write results post.

**Option B — Launch with hospital case study:**
- **Wednesday, April 2, 2026** — 8:00 AM PT / 5:00 PM CET
- Full work week ahead for people to try it. Avoid April 1 (April Fools).

**Option C — Weekend HN-optimized:**
- **Saturday, April 5, 2026** — 9:00 AM PT / 6:00 PM CET
- Less Show HN competition, longer front-page dwell. Lower Twitter/LinkedIn reach.

**Decision**: If benchmark results look strong (top-2 on CRM Arena), wait for
Option A. If uncertain or delayed, go Option B. Do not wait more than 3 weeks.

---

## 4. Channel Strategy with Draft Copy

> **NOTE**: The copy below is written for **Path B** (hospital case study launch).
> If **Path A** (benchmark-led) is chosen, all headlines and hooks need rewriting
> to lead with the benchmark result, not the product. Path A copy is TBD pending
> AgentBeats results.

### 4.1 Hacker News — Show HN

**Title** (74 chars — HN limit is 80):
```
Show HN: ClawLoop - AI agents that learn from mistakes. 7B beats o4-mini
```

**First comment** (post immediately after submission):

```
Hey HN — I'm one of the founders of Aganthos. We built ClawLoop because we kept
hitting the same wall: AI agents are static. Deploy them, and day 1000 is identical
to day 1.

ClawLoop is a learning layer that sits between your agent framework and the models
that power it. Three layers of learning, one API:

1. Harness: a reflector analyzes failed episodes and writes playbook entries that
   improve the system prompt. No GPU needed.
2. Router: tracks which model performs best per task type and routes to the cheapest
   model that maintains quality.
3. Weights: LoRA/GRPO training via SkyRL when you want to fine-tune small models.

All three layers share the same protocol: forward_backward → optim_step, with atomic
rollback and regression gating. If a change makes things worse, it gets rolled back.

Quickstart is 5 lines:

    import clawloop
    client = clawloop.wrap(your_llm_client)
    # use client normally — ClawLoop captures traces and learns

We tested this at a German university hospital: RL-post-trained a 7B model that
outperforms o4-mini by 30% on clinical benchmarks, at 90% lower cost, with full
data sovereignty. Two ICML 2026 submissions based on the work.

License is BSL 1.1 — free for any org under $10M revenue that isn't building a
competing agent-learning service. Converts to Apache 2.0 in 2030. We chose this
because we saw what happened to Redis and Elastic: we'd rather be upfront from day one.

Built with: Python, litellm, pydantic. No heavy dependencies. Works with any
LLM provider (Anthropic, OpenAI, Gemini, local models via ollama).

Happy to answer questions about the architecture, the RL approach, or why we think
agent learning is an infrastructure problem, not an application problem.
```

### 4.2 Reddit r/MachineLearning

**Title:**
```
[P] ClawLoop: Unified learning layer for AI agents — prompt optimization, model routing, and weight training under one API
```

**Body:**

```
We're releasing ClawLoop, a unified learning API for AI agents that combines three
learning layers:

**Harness** — An LLM reflector analyzes failed agent episodes and generates structured
playbook entries that get injected into the system prompt. Think of it as automated
prompt engineering from production traces. Includes GEPA (Pareto prompt evolution) for
multi-objective optimization.

**Router** — Tracks per-task model performance and routes to the cheapest model that
maintains quality. Start with GPT-4o for accuracy, automatically shift qualifying
tasks to Haiku or local models.

**Weights** — LoRA/GRPO training via SkyRL backend. When playbook improvements plateau,
distill knowledge into smaller fine-tuned models.

All three layers follow the same protocol (forward_backward → optim_step) with atomic
rollback and regression gating via gate_for_deploy().

**Key result:** We RL-post-trained a 7B model for a German university hospital (FHIR
clinical tasks). 30% improvement over o4-mini, 90% cost reduction. Two ICML 2026
submissions based on this work.

**What makes this different from MetaClaw/Tinker?**
- Conflict detection and resolution (MetaClaw has none)
- Per-entry attribution with relevance scoring
- Atomic rollback across all three layers
- Self-hosted training — no vendor lock-in to proprietary cloud
- Multi-objective reward composition vs single PRM

GitHub: github.com/aganthos/clawloop
License: BSL 1.1 (free <$10M revenue + non-competing, Apache 2.0 in 2030)
Install: pip install clawloop

Happy to discuss the RL architecture, reward composition system, or the GEPA
evolutionary approach.
```

### 4.3 Reddit r/LocalLLaMA

**Title:**
```
ClawLoop: open-source tool that makes your local LLMs learn from their mistakes — turned a 7B into something that beats o4-mini by 30%
```

**Body:**

```
We just released ClawLoop, a learning layer for AI agents. Figured r/LocalLLaMA
would be interested in the weight training side specifically.

**The pitch for local model users:**

You have a 7B/8B model running on your hardware. It's decent but makes the same
mistakes repeatedly. ClawLoop wraps your model and captures traces, then:

1. **Prompt layer (no GPU):** Analyzes failures, generates playbook entries, improves
   the system prompt automatically. You'll see improvement in minutes.
2. **Weight layer (GPU):** When prompt optimization plateaus, ClawLoop can LoRA
   fine-tune via GRPO using SkyRL. Train on your own hardware.

**Real result:** We did this for a hospital — RL-post-trained a 7B (Qwen2.5-7B base)
on clinical FHIR tasks. It now outperforms o4-mini by 30%. Full data sovereignty,
90% cost reduction vs API models.

The harness learning mode (prompt optimization) works great even without a GPU. The
weight training mode needs a GPU but runs locally — no cloud, no API calls, your data
stays yours.

Works with any litellm-compatible model: ollama, vllm, local OpenAI-compatible
endpoints.

pip install clawloop
GitHub: github.com/aganthos/clawloop (BSL 1.1 — free <$10M + non-competing, Apache 2.0 in 2030)
```

### 4.4 Twitter/X Thread

**Tweet 1 (hook):**
```
We just released ClawLoop — a learning layer that makes AI agents improve from
their own mistakes.

3 learning layers. 1 API. Atomic rollback.

A 7B model trained with ClawLoop outperforms o4-mini by 30% at 90% lower cost.

Here's how it works 🧵
```

**Tweet 2 (problem):**
```
The problem: AI agents are static.

Deploy one today, and day 1000 is identical to day 1. Same mistakes. Same costs.
Same limitations.

$24.8B in unrealized value from inefficient LLM inference (Linux Foundation).

Agents need to learn from experience. That's what ClawLoop does.
```

**Tweet 3 (solution):**
```
ClawLoop has 3 learning layers:

Layer 1 — Harness: Analyzes failures → writes playbook entries → improves system
prompt. No GPU needed.

Layer 2 — Router: Tracks per-task model performance → routes to cheapest model
that maintains quality.

Layer 3 — Weights: LoRA/GRPO fine-tuning via SkyRL when prompts plateau.
```

**Tweet 4 (differentiator):**
```
What makes this different:

- Atomic rollback: bad change? Auto-reverted.
- Regression gating: changes only deploy if they improve metrics
- All 3 layers share one protocol
- Self-hosted training: your data, your GPU, no vendor lock-in
- Works with any LLM: Anthropic, OpenAI, Gemini, ollama
```

**Tweet 5 (CTA):**
```
Get started in 5 lines:

pip install clawloop

import clawloop
client = clawloop.wrap(your_llm_client)
# use client normally — it learns

GitHub: github.com/aganthos/clawloop
License: BSL 1.1 (free <$10M + non-competing)

Star it if you think agents should learn ⭐
```

### 4.5 LinkedIn

```
We just released ClawLoop — a unified learning API for AI agents.

Every enterprise CTO I talk to says the same thing: "We deployed AI agents, and
they're... okay. They work for the easy stuff, but they hit a wall on anything
domain-specific."

The core problem: today's agents don't learn. Day 1000 is identical to day 1.

ClawLoop is the learning layer that changes this. Three learning mechanisms — prompt
optimization, model routing, and weight training — all under one API with atomic
rollback and regression gating.

Real-world proof: we RL-post-trained a 7-billion-parameter model for a German
university hospital. It outperforms OpenAI's o4-mini by 30% on clinical benchmarks,
at 90% lower inference cost, with full data sovereignty.

This isn't just a healthcare result — it's a universal optimization engine for any
domain-specific AI agent.

ClawLoop is source-available under BSL 1.1 (free for organizations under $10M
revenue that aren't building a competing service, converts to Apache 2.0 in 2030).
We chose transparency over ambiguity.

Two ICML 2026 submissions. Swiss-engineered, US-incorporated. Berkeley SkyDeck Batch 22.

Try it: pip install clawloop
GitHub: github.com/aganthos/clawloop

#AIAgents #MachineLearning #ReinforcementLearning #OpenSource #LLM
```

### 4.6 n8n Community

**Title:** "Make your n8n AI agent workflows learn from experience — ClawLoop integration"

**Body:**

```
Hey n8n community!

We built ClawLoop, a learning layer for AI agents, and we have first-class n8n
integration.

**The problem:** Your n8n AI agent workflows make the same mistakes repeatedly.
You manually tweak prompts, but there's no systematic way for the agent to improve
from its production experience.

**The solution:** ClawLoop sits between your n8n workflow and your LLM. It captures
agent traces, analyzes failures, and automatically improves the system prompt with
structured "playbook" entries.

**How it works with n8n:**
1. Point your n8n AI agent node at ClawLoop's proxy endpoint instead of OpenAI directly
2. ClawLoop transparently proxies to your LLM and captures traces
3. After each batch of interactions, the reflector analyzes what went wrong
4. Playbook entries get injected into future prompts
5. Your agent gets measurably better over time

No code changes to your workflows. Just change the API endpoint.

We have an n8n workflow template that sets this up in minutes.

GitHub: github.com/aganthos/clawloop
License: BSL 1.1 (free for orgs under $10M revenue that aren't building competing services)

Happy to help anyone set this up — drop questions below.
```

### 4.7 dev.to Article

**Title:** "Building AI Agents That Learn: How We Made a 7B Model Outperform o4-mini"

**Outline:**

```
## Introduction
- The static agent problem: deploy and forget
- Why prompt engineering doesn't scale

## The Three Layers of Agent Learning
- Harness: prompt optimization from production traces
  - How the reflector works (with code example)
  - Playbook entries: structured memory that compounds
  - GEPA: evolutionary prompt optimization
- Router: intelligent model selection
  - Cost optimization without quality loss
  - Per-task performance tracking
- Weights: when prompts aren't enough
  - LoRA/GRPO via SkyRL
  - The harness→weights pipeline

## Architecture Deep Dive
- The Layer Protocol: forward_backward → optim_step
- Atomic rollback: why it matters
- Regression gating: gate_for_deploy()
- Content-addressed StateID for reproducibility

## Case Study: German University Hospital
- The challenge: FHIR clinical queries
- Training setup and results
- 30% improvement over o4-mini
- 90% cost reduction
- Data sovereignty implications

## Getting Started (Tutorial)
- pip install clawloop
- 5-line wrap() integration
- Running your first harness learning session
- Interpreting playbook entries
- When to upgrade to weight training

## Comparison: ClawLoop vs MetaClaw vs Honcho vs GUM
- Feature matrix table
- Architectural differences
- When to use what

## What's Next
- Cloud Pro (managed persistence, dashboards, team features)
- Competition results (AgentBeats)
- Community roadmap

## Conclusion
- Agents should learn. This is infrastructure, not magic.
- Link to GitHub, invite contributions
```

---

## 5. BSL License Positioning

### README License Section

Add this to the README:

```markdown
## License

ClawLoop is source-available under the [Business Source License 1.1](LICENSE).

**What this means:**
- **Free for organizations under $10M annual revenue** — use in production,
  no restrictions, no payment required.
- **Organizations over $10M**: production use requires a [commercial license](mailto:info@aganthos.com).
  Any Pro or Enterprise subscription includes a commercial license.
- **Not for competing services**: building an agent-learning-from-experience
  product? You need a [commercial agreement](mailto:info@aganthos.com) regardless
  of revenue size.
- **Apache 2.0 on April 1, 2030** — the code automatically converts to a fully
  permissive open-source license after 4 years.
- **Non-production use is always free** — development, testing, evaluation,
  academic research, personal projects.

We chose BSL because we believe in transparency. You can read every line of code,
audit the learning algorithms, and verify there's no lock-in. We ask that large
companies and competitors contribute back financially or commercially.

For licensing questions: info@aganthos.com
```

### Prepared Responses for "This Isn't Open Source"

**Response 1 — Short (for HN/Twitter):**
```
You're right that BSL isn't OSI-approved open source. We call it "source-available."
The code is fully readable and auditable. It's free for the vast majority of users
(under $10M revenue and not building a competing agent-learning service), and it
converts to Apache 2.0 in 2030. We chose this because we've seen what happens when
infrastructure companies go pure open source — they either die or pull a license
switch later. We'd rather be upfront from day one.
```

**Response 2 — Detailed (for blog/Reddit):**
```
Fair pushback. Let me be precise:

BSL 1.1 is source-available, not open source by the OSI definition. We're not
claiming otherwise.

Here's why we chose it:

1. We're a 3-person Swiss startup. Pure Apache 2.0 means AWS ships "Amazon ClawLoop"
   in 6 months and we're dead. This has happened to Redis, Elastic, MongoDB, and
   many others.

2. The $10M revenue threshold means ~99% of companies use it for free with zero
   restrictions (as long as they're not building a competing agent-learning service).
   Startups, indie hackers, students, researchers, small businesses — all free.

3. Every line of code is readable and auditable. You can fork, modify, self-host,
   and build on it. The only restriction is production use at scale by large companies.

4. It converts to Apache 2.0 on April 1, 2030. This isn't a "maybe" — it's
   contractual. After 4 years, it's fully permissive open source, forever.

Companies who pioneered this model:
- MariaDB (created BSL)
- CockroachDB (BSL with similar revenue threshold)
- Sentry (FSL — Fair Source License, similar concept)
- HashiCorp (BSL after Terraform incident)

We think this is the honest approach: be upfront about the business model instead
of going Apache 2.0 now and switching later when VCs demand revenue.
```

**Response 3 — For the "just use Apache 2.0" crowd:**
```
We considered it. The track record of VC-backed Apache 2.0 infrastructure companies
is: grow → realize cloud providers capture all value → relicense → community outrage.

Redis, Elastic, HashiCorp, Grafana (AGPL), MongoDB — all went through this.
We'd rather set expectations correctly from the start.

If you're under $10M revenue and not building a competing service, there is
literally no difference for you. Over $10M or building something similar? We think
a commercial license is reasonable for infrastructure that measurably improves
your AI agents.
```

### Precedent Positioning

| Company | License | Threshold/Model | Status |
|---------|---------|-----------------|--------|
| MariaDB | BSL 1.1 (creator) | Time-based conversion | Successful, widely adopted |
| CockroachDB | BSL 1.1 | Revenue + cluster size | Enterprise standard |
| Sentry | FSL (Fair Source) | Revenue threshold | Well-received by community |
| HashiCorp | BSL 1.1 | Post-Terraform fork | Controversial (due to switch) |
| Grafana | AGPL | Copyleft (not BSL) | Successful dual-license |

**Key differentiator for ClawLoop**: We're launching with BSL from day one. We never promised open source and then pulled it back. This matters — the community anger around HashiCorp and Redis was about the bait-and-switch, not the license itself.

### 6-Month Review Commitment

Add this to the README license section and use it in HN/Reddit responses:

```
We commit to publicly reviewing our license terms 6 months after launch,
informed by actual adoption data. If the threshold needs adjusting — up or
down — we'll do it transparently. Follow the discussion in GitHub Discussions.
```

This is a powerful de-escalation tool. It signals good faith and gives critics
a reason to wait rather than fight.

### Corporate Entity Note

Licensor in LICENSE is "Aganthos Inc." (Delaware C-Corp, parent entity).
Aganthos GmbH (Swiss) is a 100% subsidiary. All licensing and commercial
agreements should reference Aganthos Inc. as the licensor. Update pyproject.toml
`authors` field if it still says "Aganthos Inc." — this is correct.

---

## 6. Launch Day Sequence

**Target: Wednesday, April 2 or April 16, 2026** (depending on benchmark readiness)

### Pre-Launch (Day Before)

| Time (CET) | Time (PT) | Action |
|-------------|-----------|--------|
| 10:00 | 01:00 | Final README review, all links tested |
| 11:00 | 02:00 | `pip install clawloop` verified in clean venv |
| 12:00 | 03:00 | Demo GIF recorded and embedded in README |
| 14:00 | 05:00 | Finalize Show HN post, Twitter thread, LinkedIn post, enterprise email |
| 15:00 | 06:00 | Social preview image uploaded to GitHub |
| 16:00 | 07:00 | Pre-schedule LinkedIn post for tomorrow 6:00 PM CET |
| 17:00 | 08:00 | Pricing page live on aganthos.com (or GitHub wiki) |
| 18:00 | 09:00 | **Code freeze** — no more changes until after launch |
| 20:00 | 11:00 | Sleep. |

### Launch Day — 3 Focused Channels

| Time (CET) | Time (PT) | Action | Notes |
|-------------|-----------|--------|-------|
| 16:30 | 07:30 | Flip repo to public on GitHub | Verify secrets removed, .env not committed |
| 16:45 | 07:45 | Publish to PyPI | Verify `pip install clawloop` works immediately |
| 17:00 | 08:00 | **Show HN** | Submit + immediately post first comment |
| 17:10 | 08:10 | **Twitter/X thread** | 5 tweets, 1-minute gaps |
| 17:30 | 08:30 | **LinkedIn post** | Pre-scheduled or manual |
| 17:30–19:00 | 08:30–10:00 | **HN engagement** — all attention here | Respond to every comment within 15 min |
| 19:00 | 10:00 | First metrics snapshot | HN rank, stars, pip installs |
| 19:30 | 10:30 | **Enterprise outbound batch 1** | Send 25 personalized emails to CTOs via SkyDeck network |
| 20:00 | 11:00 | Second HN engagement sweep | |
| 21:00 | 12:00 | Midday metrics check | |
| 23:00 | 14:00 | US afternoon engagement sweep | West Coast lunch = high activity |
| 01:00+1 | 16:00 | End-of-US-business sweep + enterprise batch 2 (25 more emails) | |
| 02:00+1 | 17:00 | Day 1 metrics snapshot | Log all numbers (see Section 8) |

### Days 2–7: Staggered Channels

| Day | Channel | Notes |
|-----|---------|-------|
| Day 2 | Twitter "24h since launch" thread | Transparency post with real numbers |
| Day 3 | r/MachineLearning + r/LocalLLaMA | Different framing per subreddit |
| Day 5 | n8n community | Workflow template + tutorial |
| Day 7 | dev.to article | Full tutorial |

### Key Launch Day Rules

1. **Respond to every HN comment** — especially critical ones. Thoughtful, technical responses earn respect.
2. **Don't be defensive about BSL** — acknowledge the tradeoff, explain the reasoning, move on. Use prepared responses (Section 5). Publicly mention the 6-month review commitment.
3. **Have code examples ready** — if someone asks "how does X work?", paste a code snippet, not marketing copy.
4. **Both founders online** — one monitors HN, one monitors Twitter/LinkedIn + enterprise emails.
5. **Don't astroturf** — no fake accounts, no coordinated upvoting. HN detects this and will penalize.

### Contingency Plan (if launch goes badly)

You get at least 2 shots within 90 days if you create new proof artifacts.

| Scenario | Response |
|----------|----------|
| HN post dies (< 10 points) | Reframe 2–3 weeks later as benchmark result or case study post. Different artifact, not a relaunch. |
| Critical bug found day 1 | Ship fix within 24h. Post transparent issue + "fixed" update. Early credibility beats silence. |
| Competitor drops major release same week | Don't fight their news. Publish differentiation post the following week. |
| BSL discussion dominates all threads | Post empathetic explanation + 6-month review commitment. Redirect to tech after one response. Don't engage license trolls. |
| Enterprise outbound gets 0 responses | Double down on SkyDeck network intros. Ask batch-mates for warm intros to their enterprise contacts. |

---

## 7. Post-Launch: Week 1–2

### Monitoring Setup

| Tool | What to Track | Check Frequency |
|------|--------------|-----------------|
| GitHub Insights | Stars, forks, clones, traffic sources | Daily |
| PyPI Stats (pypistats.org) | Daily downloads | Daily |
| HN (hnrankings.info) | Post rank over time, point count | Hourly on day 1, daily after |
| Twitter Analytics | Thread impressions, profile visits, followers | Daily |
| Reddit | Post upvotes, comment count | Daily |
| Google Alerts | "ClawLoop" mentions | Daily digest |
| GitHub Issues | Bug reports, feature requests | Real-time (GitHub notifications) |
| `pip install` errors | PyPI download failures, dependency issues | Check within 2 hours of launch |

### Response Strategy

| Feedback Type | Response Time | Template |
|--------------|---------------|----------|
| Bug report | < 4 hours | Acknowledge, label, provide workaround if possible |
| Feature request | < 24 hours | Thank, label, explain if planned or not |
| "This isn't open source" | < 1 hour (launch day) | Use prepared response (Section 5) |
| Positive mention | < 2 hours | Thank, RT/share, engage with follow-up |
| Technical question | < 4 hours | Code example in response, link to docs |
| PR submission | < 24 hours | Review, provide feedback, merge or explain why not |
| Comparison to competitor | < 2 hours | Factual response, no trash-talking, acknowledge strengths |

### Follow-Up Content Cadence

| Day | Content | Channel |
|-----|---------|---------|
| Day 2 | "24 hours since launch — here's what we learned" thread | Twitter |
| Day 3 | Respond to all remaining HN/Reddit comments | HN, Reddit |
| Day 5 | "Week 1 numbers" transparency post | Twitter, LinkedIn |
| Day 7 | Publish dev.to article + GitHub release notes (v0.0.1 tag) | dev.to, GitHub |
| Day 8 | Address top 3 community-requested features | GitHub Issues, Twitter |
| Day 10 | Short demo video (2 min): "wrap → learn → improve" | YouTube, Twitter, LinkedIn |
| Day 12 | Second dev.to article: "ClawLoop vs MetaClaw — an honest comparison" | dev.to |
| Day 14 | Two-week retrospective + what's next | Twitter, LinkedIn, GitHub Discussions |

### Community Building

- **Enable GitHub Discussions** on launch day — pin a "Welcome" thread
- **Create a Discord** (or defer until 100+ stars to avoid ghost-town effect)
- **Label "good first issue"** on 3–5 easy issues before launch — this attracts contributors
- **Write a "Contributing to ClawLoop" blog post** for Week 2 — lower the barrier

### Guerrilla Tactics (learned from Clawby launch playbook)

These are scrappier, higher-effort-per-lead tactics. Not all will apply to dev
tools, but the ones marked with * are worth doing:

| Tactic | Applicability | Effort |
|--------|---------------|--------|
| **Monitor competitor pain points*** | High — find MetaClaw issues, Tinker bugs on Twitter/Reddit/GitHub, offer ClawLoop as alternative | 30 min/day |
| **SEO content*** | High — write "ClawLoop vs MetaClaw", "how to optimize AI agent prompts", "LoRA fine-tuning for agents" comparison/guide pages | 1 article/week |
| **Engage competitor communities*** | Medium — find MetaClaw/Tinker GitHub stars, follow on Twitter, engage with their content | 15 min/day |
| **DM conference speakers** | Medium — find people who gave talks on agent learning at NeurIPS/ICML, DM with ClawLoop | One-time |
| **Referral program** | Low at v0.0.1 — no product loop yet. Defer to Cloud Pro launch | — |
| **Group chats** | Low for dev tools — not relevant like it is for consumer products | — |
| **Cross-post on free sites** | Medium — dev.to, Hashnode, Medium (syndicate the dev.to article) | 1 hour |

**The highest-ROI guerrilla tactic**: Respond to people struggling with agent
learning on Twitter, Reddit, and Stack Overflow. Not "use ClawLoop!" — instead,
answer their technical question AND mention ClawLoop where relevant. This is how
Supabase grew against Firebase.

---

## 8. Success Metrics

### Week 1 Targets

| Metric | Target | Stretch | Source |
|--------|--------|---------|--------|
| GitHub stars | 200 | 500 | Comparable projects: MetaClaw has ~1.2K total |
| PyPI downloads | 500 | 1,500 | First-week installs indicate real interest |
| HN points | 100 | 250 | Top 10 on Show HN for 4+ hours |
| HN comments | 30 | 80 | Engagement depth matters more than points |
| GitHub issues opened | 5 | 15 | Signal that people are actually using it |
| Twitter thread impressions | 20K | 100K | With RT from 2-3 ML accounts |
| Reddit total upvotes | 100 | 300 | Across both subreddits |
| Forks | 10 | 30 | Indicator of deep interest |
| First external PR | 1 | 3 | Community contribution |

### Month 1 Targets

| Metric | Target | Stretch | Notes |
|--------|--------|---------|-------|
| GitHub stars | 500 | 1,500 | Sustained growth after launch spike |
| PyPI downloads (cumulative) | 2,000 | 5,000 | Weekly download trend matters more than total |
| GitHub contributors | 3 | 10 | Beyond the founding team |
| GitHub issues | 20 | 50 | Mix of bugs and features |
| Discord/community members | 50 | 200 | Only if Discord is created |
| Newsletter/waitlist signups (cloud) | 100 | 500 | For Cloud Pro launch |
| First blog post by external developer | 1 | 3 | Organic content = validation |
| Conference/meetup talk invitations | 1 | 3 | Speaking slots from launch visibility |
| Enterprise inbound inquiries | 2 | 5 | "Can we use this at [company]?" |

### What Defines Success for v0.0.1

**Minimum viable success:**
- 200+ stars and sustained weekly downloads (not just launch spike)
- At least 1 external contributor submits a PR
- At least 2 enterprise inquiries from companies exploring agent learning
- No critical bugs reported in the first week (proves packaging/testing quality)
- BSL license doesn't dominate the conversation (learning + results are the story)

**Strong success:**
- 500+ stars, trending on GitHub for Python/ML
- Show HN hits front page and stays 4+ hours
- Someone independently benchmarks ClawLoop and posts results
- Cloud Pro waitlist hits 100+ signups
- One or more ML influencer shares/endorses

**Exceptional success:**
- 1,000+ stars in month 1
- Featured in ML newsletter (The Batch, TLDR AI, etc.)
- Partnership inquiry from agent framework (LangChain, CrewAI, AutoGen)
- Direct competitor response (validates market)
- Inbound VC interest from launch visibility alone

### What to Watch For (Red Flags)

- Download count high but no issues/stars → people install, hit friction, leave silently. Check for install/import errors.
- Stars high but downloads low → people star for later but don't try it. README isn't converting to installs. Improve quickstart.
- All feedback is about BSL → license is drowning the product story. Respond once, then redirect every conversation to the tech.
- No enterprise inbound by week 3 → messaging isn't reaching decision-makers. Double down on LinkedIn, reach out to SkyDeck network directly.

---

## Appendix A: Pre-Launch Task List

Ordered by priority. Complete before flipping to public.

**Critical (blocks launch):**
1. **Fix README** (Section 1 improvements — especially `pip install clawloop` and wrap() quickstart)
2. **Publish pricing page** — reconcile 3 conflicting sources into one canonical page
3. **Record demo GIF** (asciinema or vhs, 30 seconds)
4. **Verify `pip install clawloop`** works from PyPI in clean venv
5. **Run AgentBeats benchmark** (if pursuing Path A launch) — decides launch date

**Important (launch quality):**
6. **Add CODE_OF_CONDUCT.md** (Contributor Covenant v2.1)
7. **Add SECURITY.md** (responsible disclosure to security@aganthos.com)
8. **Add .github/ISSUE_TEMPLATE/** (bug report + feature request)
9. **Create social preview image** (1280×640px)
10. **Add CI workflow** (.github/workflows/test.yml — pytest on push)
11. **Fix .gitignore** — add `examples/openclaw_runner/node_modules/`

**Growth (launch amplification):**
12. **Write Show HN post + first comment** (Section 4 drafts as starting point)
13. **Write Twitter thread + LinkedIn post**
14. **Draft 50 enterprise outbound emails** (CTO-targeted, benchmark or case study as opener)
15. **Brief SkyDeck cohort** — ask for launch-day RTs/shares and warm enterprise intros
16. **Label 3–5 issues as "good first issue"**
17. **Enable GitHub Discussions** — pin a Welcome thread
18. **Identify 3 design-partner candidates** for outcome-based pilot program

## Appendix B: Full Disagreement Log (Claude vs Codex)

Preserved for decision-making reference. These are genuine differences in
strategic judgment, not right/wrong.

| # | Topic | Claude Position | Codex Position | Resolution |
|---|-------|----------------|----------------|------------|
| 1 | BSL threshold | $10M (wider funnel, CockroachDB precedent) | $5M (earlier monetization) | **$10M** — adoption priority at v0.0.1 |
| 2 | Number of tiers | 4 (Free/Pro/Team/Enterprise) | 3 (Free/Pro/Enterprise) | **3** — Team deferred |
| 3 | Pro pricing model | $149/mo flat subscription | Credit-based + reverse trial | **$99 BYOK / $199 Managed** + reverse trial + credit alternative. Original $149 had ~42% margin at high volumes. BYOK split fixes this (see Section 2 for updated math at canonical 5K eps). |
| 4 | Launch channel spread | 7 channels day 1 | 3 channels max | **3 day 1**, stagger rest over week 1 |
| 5 | Launch framing | Product announcement | Benchmark result | **Benchmark if ready**, else case study |
| 6 | Proof points needed | 1 (hospital) is enough | Need 2–3 | **1 + promise of benchmark** |
| 7 | Pricing page at launch | Not required | Required | **Required** — Codex was right |
| 8 | License review commitment | Not proposed | 6-month public review | **Include** — Codex was right |
| 9 | Outcome-based pricing | Not proposed | Limited to 3 design partners | **Include** as design-partner program |
| 10 | Anti-launch (quiet flip) | Not considered | Viable if 2-3 partners locked | **Rejected** — need visible traction for pre-seed |
| 11 | Apache 2.0 vs BSL | BSL strongly | BSL but re-evaluate | **BSL with 6-month review** |
| 12 | Enterprise outbound | Day 1 multi-channel | Day 1 focused outbound | **Day 1 enterprise emails** — Codex was right |

### Ideas Parked for v0.2+

- Progressive BSL brackets ($5M/$25M)
- "Pay what your agent saves" (10% of savings) model
- Star-gated free Pro (badge + star = 1yr free cloud)
- Credit-only pricing (no subscription)
- Apache 2.0 conversion (pending 6-month review data)
