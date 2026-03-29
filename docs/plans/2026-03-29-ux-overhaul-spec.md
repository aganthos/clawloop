# ClawLoop Developer Experience Overhaul

Date: 2026-03-29
Branch: `chore/ux-overhaul`

## Goal

Make ClawLoop's value obvious in <2 minutes and get a first successful run
in <5 minutes. Users in any integration category (Python SDK, n8n, OpenClaw,
OpenAI-compatible, weight training) find their path in <30 seconds.

## Audience (priority order)

1. **Framework developers** — have an agent, want it to learn
2. **Technical decision makers / VCs** — need to understand value fast
3. **ML researchers** — internal, lower priority for public docs

## Lead story

"Agents that learn from experience." The hook is the learning loop, not any
single integration path. ClawLoop observes agent-environment interactions,
learns from them (prompt evolution, model routing, weight training), and
makes the agent better.

## README.md — Complete Rewrite

### Narrative arc

Logo → one-liner → architecture diagram → install → 10-second demo →
choose your path → how it works → environments → LLM providers → advanced

### Architecture diagram (Mermaid)

Full system showing:
- **Integration paths** (left): Python SDK, n8n/webhooks, OpenAI-compatible
  (litellm), OpenClaw Proxy
- **ClawLoop engine** (center): Episode Collector → three learning layers
  (Harness, Router, Weights) → unified protocol (forward_backward → optim_step)
- **Experience sources** (right): Environments (math, harbor, CRMArena, custom),
  Production traces (proxy, webhooks, callbacks)
- **Loop**: Updated agent state feeds back to environment interaction

Weights layer described as "finetune & RL (GRPO, PPO, full finetune)" — not
just LoRA/GRPO.

### Choose your path table

| You have... | Start here |
|---|---|
| A Python agent | `examples/demo_math.py` |
| An n8n or workflow platform | `examples/n8n/` |
| An OpenAI-compatible agent or API | CARBench / CRMArena via litellm |
| Want zero-code-change learning | `examples/openclaw_demo.py` |
| GPU resources for weight training | `examples/recipes/` |

### Sections

1. Logo (clawloop.png)
2. Tagline + subtitle
3. Architecture diagram (Mermaid)
4. Install (`pip install clawloop`)
5. Try it in 10 seconds (`demo_math.py --dry-run`, show output)
6. Choose your path (table)
7. How it works (4 paragraphs: loop, harness, router, weights, protocol)
8. Environments (table)
9. LLM Providers (litellm, short)
10. Advanced (collapsible): adding environments, config reference, architecture

## Examples Changes

### New: `examples/n8n/`

- `README.md` — adapted from `n8n-workflows/README.md`, CLIProxyAPI refs
  replaced with generic OpenAI-compatible upstream
- `customer-support.json` — n8n workflow, repointed to generic upstream
- `demo.py` — adapted from `scripts/demo.py`
- `demo_tickets.json` — from `config/demo_tickets.json`

### Rewrite: `examples/README.md`

Routing page, not wall of text. Same path table with slightly more detail.

### Fix: `examples/demo_math.py`

Real-mode defaults use litellm provider routing (`anthropic/claude-haiku-4-5-20251001`)
instead of localhost:11434. Setting `ANTHROPIC_API_KEY` should just work.

### Fix: `examples/playbook_demo.py`

Add `--dry-run` mode with mock LLM clients (same pattern as demo_math.py).

### Rename: `examples/openclaw_proxy_demo.py` → `examples/openclaw_demo.py`

Clearer name.

## What Stays Private

- `n8n-workflows/` — original with CLIProxyAPI refs
- `scripts/` — CLIProxyAPI test scripts
- `config/` — demo_tickets.json gets copied to examples/n8n/
- `resources/` — diagrams recreated as Mermaid

## What Does NOT Change

- `clawloop/` library code
- `tests/`
- `examples/recipes/`
- `examples/train_runner.py` + `configs/`
- `.publicpaths` (examples/n8n/ already covered by `examples/`)

## Patterns Applied (from competitive research)

- Code-first opening (Instructor pattern)
- "(This example is complete, it can be run as-is)" trust signal (Pydantic AI)
- Collapsible advanced sections (smolagents)
- Ruthless brevity at top, depth pushed down (LangChain)
- Sticky tagline: "Agents that learn from experience"
