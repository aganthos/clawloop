# CAR-bench Experiment Tracking

## Setup

- **Benchmark**: CAR-bench (254 tasks: base/hallucination/disambiguation)
- **Agent under test**: Haiku (`openai/claude-haiku-4-5-20251001`) via CLIProxyAPI
- **User simulator**: Gemini (`gemini/gemini-2.5-flash`) via GEMINI_API_KEY
- **Policy evaluator**: Gemini (same)
- **Reflector**: Sonnet (`openai/claude-sonnet-4-20250514`) via CLIProxyAPI
- **Learning**: lfx harness layer — ACE playbook (incremental insights) + GEPA prompt evolution

## Experiments

### EXP-001: smoke_001 — Pipeline validation

| Field | Value |
|-------|-------|
| **Date** | 2026-03-13 |
| **Dir** | `runs/car/smoke_001/` |
| **Config** | 1 iter, 2 base tasks, test split |
| **Hypothesis** | Validate the lfx → agentbeats-run → lfx_server.py pipeline works end-to-end |
| **Result** | Pipeline works. 1 task evaluated (base_1), reward=0.0. Reflector generated 3 insights. |

**Findings:**
- Haiku calls tools with `proxy_` prefix (`proxy_open_close_sunshade` instead of `open_close_sunshade`) → always fails `r_tool_subset`
- All other metrics passed: `r_actions_intermediate=1.0`, `r_tool_execution=1.0`, `r_policy=1.0`, `r_user_end_conversation=1.0`
- base_0 missing from results because CLI `--task-split` default ("test") overrode config ("train")
- Reflector (Sonnet) correctly identified the tool-naming failure pattern

---

### EXP-002: train_icl_001 — First ICL learning run

| Field | Value |
|-------|-------|
| **Date** | 2026-03-13 |
| **Dir** | `runs/car/train_icl_001/` |
| **Config** | 3 iters x 5 base tasks, train split, Haiku agent, Sonnet reflector |
| **Hypothesis** | The Reflector will identify Haiku's tool-naming bug and generate playbook entries that improve subsequent iterations. Reward should increase from iter 0 → iter 2. |
| **Result** | Iter 0 produced real data (3/5 tasks, 7-13 turns each). Iters 1-2 failed: Gemini 429. |

**Findings:**
- **Iter 0**: 3 tasks evaluated (base_0, base_2, base_4). All reward=0.0. Same `proxy_` tool name bug. Policy, tool execution, intermediate actions all pass.
- **Iters 1-2**: Gemini free tier quota exhausted (20 req/day limit). All tasks got empty trajectories (0 turns). The Reflector correctly detected the "empty trace" pattern.
- **Playbook grew**: 0 → 3 → 6 entries. Reflector generated debugging-oriented insights about empty traces. Correctly marked iter-1 entries as harmful (h=0,harm=5) when they didn't help.
- **Bugs found**: (a) Reflector crashed on `None` message content — fixed in `_sanitize_str`. (b) 2/5 task IDs don't exist in train split (base_1, base_3 missing).
- **Blocker**: Need paid Gemini API key or route user sim/policy eval through CLIProxyAPI.

**Key question (still open):** Can the playbook teach Haiku to use correct tool names? Not testable yet due to Gemini quota.

---

### EXP-003: train_icl_003 — Clean 3-iteration ICL run

| Field | Value |
|-------|-------|
| **Date** | 2026-03-13 |
| **Dir** | `runs/car/train_icl_003/` |
| **Config** | 3 iters x 5 base tasks (3 valid), train split, Haiku agent, Sonnet reflector |
| **Hypothesis** | Same as EXP-002. With Gemini rate limit fixed and dynamic ports, all iterations should produce real data. |
| **Result** | All 3 iterations clean. Reward stays 0.0. Playbook grows 0→3→6→8. |

**Findings:**
- **Pipeline fully validated**: 3 iterations, 3 tasks each, all with real trajectories (7-12 turns). No port conflicts, no rate limits, no crashes.
- **Reward stays at 0.0**: Every task fails on `r_tool_subset` — Haiku calls `proxy_open_close_sunshade` instead of `open_close_sunshade`. This is a model-level tool-naming behavior.
- **Playbook cannot fix tool-naming**: The harness layer injects playbook into the system prompt, but Haiku ignores hints about correct tool names. The `proxy_` prefix is baked into the model's behavior when going through CLIProxyAPI.
- **Reflector works well**: Generates relevant insights each iteration. Correctly marks unhelpful entries as harmful (harm=5→10 across iterations).
- **Turn count decreases**: base_0 went 12→10→7 turns — the playbook teaches Haiku to give up faster on broken tools (not ideal, but shows the harness IS influencing behavior).
- **All other metrics pass**: `r_policy=1.0`, `r_tool_execution=1.0`, `r_actions_intermediate=1.0` across all tasks and iterations.

**Key insight:** The `proxy_` prefix is likely added by CLIProxyAPI when proxying tool calls. This is NOT a Haiku bug — it's a CLIProxyAPI artifact. The fix is either (a) strip `proxy_` prefix in the tool execution layer, or (b) use Haiku directly via Anthropic API instead of through CLIProxyAPI.

**Bugs fixed in this run:**
- Dynamic port allocation (no more `[Errno 48] address already in use`)
- Strip `GOOGLE_API_KEY` from subprocess env (was overriding tier-2 `GEMINI_API_KEY`)
- `_sanitize_str` handles `None` message content

---

### EXP-004: train_verify_001 — Post proxy_ fix verification

| Field | Value |
|-------|-------|
| **Date** | 2026-03-13 |
| **Dir** | `runs/car/train_verify_001/` |
| **Config** | 3 iters x 5 base tasks (3 valid), train split, Haiku agent, Sonnet reflector |
| **Hypothesis** | With CLIProxyAPI `tool_prefix_disabled: true`, `r_tool_subset` should pass and reward should be > 0. |
| **Result** | base_4 = 1.0 across all iterations. base_0 and base_2 remain 0.0. Avg reward = 0.2 (1/5, or 1/3 valid). |

**Findings:**
- **proxy_ fix confirmed**: `r_tool_subset` now passes on base_2 and base_4. Previously ALL tasks failed this metric.
- **Remaining failures are real task difficulty**, not infrastructure bugs:
  - `base_0` (sunroof): fails `r_tool_subset` + `r_policy` — multi-step dependency task, Haiku doesn't call all required tools
  - `base_2` (trunk): fails `r_policy` only — tool subset passes but Haiku violates task policy
  - `base_4` (air circulation): **reward=1.0** — all metrics pass, simple single-tool task
- **Playbook grows**: 0 → 3 → 4 → 5 entries. Reflector generates relevant insights each iteration.
- **Harmful counters work**: entries accumulate harmful signals when reward doesn't improve (harm=8 by iter 2).
- **base_1/base_3**: still don't exist in train split (empty episodes, known issue).

**Per-task metric breakdown (iter 0):**

| Task | reward | r_tool_subset | r_policy | r_actions | r_tool_execution | r_user_end |
|------|--------|---------------|----------|-----------|------------------|------------|
| base_0 | 0.0 | 0.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| base_2 | 0.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 |
| base_4 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**CLIProxyAPI fix applied in this run:**
- Upgraded CLIProxyAPI 6.7.40 → 6.8.50 via `brew upgrade`
- Added `"metadata":{"tool_prefix_disabled":true}` to `~/.cli-proxy-api/claude-robert@aganthos.com.json`
- See `docs/cliproxyapi.md` for setup details

---

## How to analyze a run

```bash
# Terminal summary with reward curves, playbook evolution, per-task metrics
uv run python scripts/analyze_run.py runs/car/<run_name>
```

## Artifact guide

Each run directory contains:

```
runs/car/<experiment>/
├── experiment.jsonl          # One JSON line per iteration (the main log)
├── iter_0/
│   ├── scenario.toml         # Exact scenario passed to agentbeats-run
│   ├── harness_prompt.json   # Harness prompt injected into agent (input)
│   ├── harness_state.json    # Full harness state snapshot after episode collection
│   ├── results.json          # CAR-bench evaluation output (rewards + trajectories)
│   └── green_agent.log       # stdout/stderr from agentbeats-run
├── iter_1/
│   ├── ...                   # Same structure, but harness_prompt.json now has playbook
├── ...
```

### What each file tells you

| File | What to look for |
|------|-----------------|
| **experiment.jsonl** | Reward trend across iterations. `avg_reward`, `playbook_size`, `insights_generated`. This is the first thing to check. |
| **harness_prompt.json** | The exact text injected before the system prompt. Empty on iter 0 (baseline). Compare across iterations to see what the Reflector learned. |
| **harness_state.json** | Full harness state at time of episode collection. Playbook entries with helpful/harmful counters show which insights are working. |
| **results.json** | Per-task rewards and full trajectories. Look at `detailed_results_by_split.base[*].reward_info.info` for metric breakdown. `trajectory` array has the full conversation. |
| **scenario.toml** | The exact configuration used. Verify model, ports, task filters, harness-file path. |
| **green_agent.log** | Raw agentbeats-run output. Check for errors, timeouts, agent readiness issues. |

### Reading experiment.jsonl

Each line is one iteration:
```json
{
  "iteration": 0,
  "avg_reward": 0.0,
  "min_reward": 0.0,
  "max_reward": 0.0,
  "n_episodes": 5,
  "playbook_size": 3,
  "playbook_entries": [{"id": "...", "content": "...", "helpful": 0, "harmful": 0}],
  "fb_results": {"harness": {"metrics": {"insights_generated": 3}}}
}
```

**Key signals:**
- `avg_reward` increasing → learning is working
- `playbook_size` growing → Reflector is generating insights
- `helpful > harmful` on entries → insight is validated by outcomes
- `insights_generated = 0` → Reflector didn't fire (check intensity settings)

### Reading per-task metrics in results.json

CAR-bench scores are all-or-nothing: reward=1.0 only if ALL metrics pass.

| Metric | What it measures |
|--------|-----------------|
| `r_actions_final` | Did the agent complete the final required action? |
| `r_actions_intermediate` | Were intermediate steps correct? |
| `r_tool_subset` | Did the agent use all required tools? |
| `r_tool_execution` | Were tool calls executed without errors? |
| `r_policy` | Did the agent follow the task policy? |
| `r_user_end_conversation` | Did the conversation end naturally? |

### Comparing iterations

```bash
# See what the harness learned between iterations
diff <(cat runs/car/train_icl_001/iter_0/harness_prompt.json | python3 -m json.tool) \
     <(cat runs/car/train_icl_001/iter_1/harness_prompt.json | python3 -m json.tool)
```
