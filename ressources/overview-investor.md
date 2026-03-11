# lfx — Learning from Experience

**A universal continual improvement layer for AI agents.**

lfx makes any LLM-powered agent get better over time. One line of code to integrate. Three independent learning mechanisms. No vendor lock-in.

---

## What It Does

Every AI agent generates traces — conversations, tool calls, successes, failures. Today those traces are thrown away. lfx captures them and uses them to improve the agent automatically.

```
Agent = Model + Harness
Harness = System Prompt + Tools + Memory
```

The Model generates responses. The Harness is everything around it. lfx improves the harness continuously from real usage data — no manual prompt engineering required.

---

## Three Learning Layers

| Layer | What it improves | Speed | Cost |
|-------|-----------------|-------|------|
| **Harness** | System prompt, strategy playbook, tool configs | Seconds | Free (prompt changes only) |
| **Router** | Which model handles each query | Minutes | Minimal |
| **Weights** | Model weights via LoRA fine-tuning | Hours | GPU cost |

Most users start with Harness only (zero cost, immediate results) and add layers as needed.

---

## Usage

```python
import lfx

# One line to add learning to any LLM client
wrapped = lfx.wrap(your_client, collector=lfx.EpisodeCollector(
    pipeline=lfx.RewardPipeline.with_defaults(),
    batch_size=16,
))

# Use exactly as before — lfx observes and learns in the background
response = wrapped.complete(messages)
```

No infrastructure changes. No API proxy. No vendor lock-in. Works with any LLM provider via litellm.

---

## Integration Ecosystem

lfx plugs into the existing AI stack:

- **litellm** — wrap any model provider (OpenAI, Anthropic, Gemini, local)
- **Langfuse** — export episodes and reward signals for dashboards and observability
- **n8n** — webhook integration for workflow automation agents
- **SkyRL / OpenClaw** — export to GRPO trainers for weight fine-tuning
- **Any LLM client** — `lfx.wrap()` works with anything that has `.complete()`

---

## Competitive Positioning

### vs. MetaClaw (API proxy + LoRA fine-tuning)

| | lfx | MetaClaw |
|---|---|---|
| Integration | SDK wrapper (1 line) | API proxy (infra change) |
| Learning layers | 3 independent (prompt, routing, weights) | 2 conflated (skills + LoRA) |
| Rollback | Atomic per-layer + cross-layer transactions | Manual |
| Regression safety | Built-in gate_for_deploy | None |
| Reward system | Composable signals with priority + defaults | PRM only (majority vote) |
| Provider lock-in | None (any LLM) | Tinker cloud for training |
| Observability | Export to Langfuse/any platform | Built-in dashboard |

### vs. building it yourself

Prompt engineering by hand works until it doesn't scale. lfx automates the feedback loop: observe → extract signals → reflect → update → validate → deploy. The Harness layer alone replaces manual prompt iteration with data-driven evolution.

---

## Monetization

### Open Core Model

**Free (open source):**
- Harness layer (prompt/playbook evolution)
- Live mode pipeline (EpisodeCollector, AsyncLearner, `lfx.wrap()`)
- All reward extractors and exporters (Langfuse, SkyRL)

**Paid tiers:**

| Tier | What you get | Price |
|------|-------------|-------|
| **Pro** | Router layer, managed judge LLM, dashboard & analytics | $49/mo |
| **Team** | Multi-agent coordination, shared playbooks, A/B testing | $199/mo |
| **Enterprise** | Weights layer (managed LoRA training), SLA, on-prem | Custom |

### Revenue Drivers

1. **Compute margin on judge calls** — lfx provides the judge LLM for Pro+ tiers. Lazy invocation keeps costs low.
2. **Training infrastructure** — Weights layer requires GPUs. Managed training-as-a-service at margin.
3. **Router optimization savings** — Router saves users money by routing simple queries to cheaper models. lfx captures a fraction of the savings.

### GTM Strategy

1. **Developer adoption** — `pip install lfx` + one-line wrap. Free Harness layer shows value immediately.
2. **Usage-based upgrade** — Once agents process enough traffic, Router and Weights unlock measurable improvements.
3. **Framework partnerships** — Pre-built integrations with litellm, n8n, LangChain, CrewAI.

---

## Roadmap

### Shipped (v0.1)
- Harness layer with prompt evolution and playbook memory
- Composable reward signals with priority cascade
- Live mode: `lfx.wrap()` → EpisodeCollector → AsyncLearner
- Default extractors (execution + user feedback)
- SkyRL exporter for GRPO training

### Next (v0.2)
- Langfuse exporter (episodes + signals → dashboard)
- litellm callback integration
- Weights layer: LoRA fine-tuning from live data
- Rich wrapper (tool calls, tokens, model ID, timing)

### Future
- API proxy mode (zero-code integration)
- Dashboard & analytics (Pro tier)
- Multi-agent shared playbooks (Team tier)
