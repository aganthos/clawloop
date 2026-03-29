# Playbook & Prompt Optimization — Literature Landscape

> Last updated: 2026-03-28. Review of 9 systems relevant to ClawLoop's harness learning architecture.

## Our Architecture (ClawLoop)

```
Static prompt (GEPA evolves)     →  Role, persona, general framing. All queries.
Playbook (Reflector evolves)     →  Specific strategies, per-query via embedding retrieval.
Cross-visibility                 →  GEPA sees playbook, Reflector sees base prompt. Novel.
```

Three mechanisms unified behind the Evolver protocol:
1. **GEPA** — evolutionary prompt optimization (mutation + crossover on Pareto front)
2. **ACE Playbook** — structured memory with helpful/harmful scoring + temporal decay
3. **Paradigm shift** — opt-in nuclear reset on stagnation (not literature-validated, see below)

## Systems Comparison

### ACE — Agentic Context Engineering (2024)
- **Paper**: https://arxiv.org/abs/2510.04618
- **What**: Evolving playbook via Generator → Reflector → Curator pipeline
- **Key result**: +10.6% on agentic benchmarks, +8.6% on domain-specific
- **Relevance**: We implement this. Our playbook IS the ACE architecture.
- **Their insight**: Incremental deltas prevent context collapse vs full rewrites

### EvoPrompt (ICLR 2024)
- **Paper**: https://arxiv.org/abs/2309.08532
- **What**: Evolutionary prompt optimization — LLM does mutation + crossover on prompt population
- **Key result**: Up to 25% improvement over human-designed prompts on BBH
- **Relevance**: Our GEPA is EvoPrompt-style with Pareto front selection
- **Their insight**: Balance exploration (diverse mutations) vs exploitation (refining best)

### PromptBreeder (2023)
- **Paper**: https://arxiv.org/abs/2309.16797
- **What**: Self-referential prompt evolution — evolves BOTH task-prompts AND mutation-prompts
- **Key result**: Outperforms CoT on arithmetic and commonsense benchmarks
- **Relevance**: We don't have self-referential mutation yet. Relevant to hyperagent roadmap.
- **Their insight**: The optimizer that optimizes itself escapes local optima naturally

### Reflexion (2023)
- **Paper**: https://arxiv.org/abs/2303.11366
- **What**: Per-instance verbal reflections stored as episodic memory
- **Key result**: 91% on HumanEval (vs 80% baseline)
- **Relevance**: Weaker than ours — instance-level, session-bound. Our Reflector extracts general strategies.
- **Their insight**: Natural language reflections as "verbal reinforcement learning"

### ERL — Experiential Reflective Learning (2025)
- **Paper**: https://arxiv.org/abs/2603.24639
- **What**: Heuristic generation from task outcomes + embedding retrieval at test time
- **Key result**: +7.8% over ReAct on GAIA2. Failure heuristics > success heuristics.
- **Relevance**: Very close to our playbook + embedding retrieval. Validates our approach.
- **Their insight**: Selective retrieval is essential. Heuristics transfer better than raw trajectories.

### MNL — Mistake Notebook Learning (2025)
- **Paper**: https://arxiv.org/abs/2512.11485
- **What**: Batch-clusters failures by semantic subject, generates 5-part structured notes
- **Key result**: 12 memory entries vs ACE's 58k tokens. Competitive accuracy.
- **Relevance**: More sophisticated playbook management than ours. Key gap to close.
- **Their mechanisms**:
  - Subject clustering before reflection (not flat batch)
  - 5-part notes: corrected examples, approach, mistake summary, strategy, anti-patterns
  - Merge-or-append with semantic similarity check
  - Accept-if-improves criterion (Δℬ > 0 or rollback)
  - Batch size 16 is optimal
- **Their insight**: Batch abstraction reduces variance. Domain-wise error taxonomy is the key.

### GUM — Generalized User Models (2025)
- **Paper**: https://arxiv.org/abs/2505.10831
- **What**: Per-proposition memory with decay, revision-over-addition, BM25+rerank
- **Key result**: Brier score 0.17 (well-calibrated confidence)
- **Relevance**: We adopted their decay model. Our retrieval is embedding-based (they suggest this).
- **Their insight**: Never evict — zero-confidence stays for audit. Per-entry decay rate.

### TextGrad (2024)
- **Paper**: Stanford HAI
- **What**: Textual gradients from LLM feedback for prompt optimization
- **Relevance**: Risk of task-specific overfitting. No exploration mechanism.
- **Their insight**: Gradient-based is efficient but brittle without diversity

### DEEVO — Tournament of Prompts (2025)
- **Paper**: https://arxiv.org/html/2506.00178v2
- **What**: Multi-agent debate + Elo rating for evolutionary prompt selection
- **Relevance**: Hybrid approach — combines debate with evolution
- **Their insight**: Structured debate as fitness function eliminates need for labeled data

## Competitive Assessment

### Where ClawLoop is strong
- ✅ ACE playbook (full pipeline)
- ✅ GEPA + Pareto front (EvoPrompt-level)
- ✅ Cross-visibility (GEPA sees playbook, Reflector sees base prompt) — **NOVEL**
- ✅ Embedding-based retrieval (validated by ERL)
- ✅ Temporal decay + helpful/harmful scoring (aligned with GUM, MNL)

### Where ClawLoop is behind
- ❌ No batch failure clustering (MNL clusters by semantic subject first)
- ❌ Reflector generates flat insights, not 5-part structured notes (MNL)
- ❌ No accept-if-improves gate (MNL rolls back if performance drops)
- ❌ No self-referential mutation (PromptBreeder evolves the mutator)
- ❌ No domain-wise error taxonomy (MNL's subject mapper)

## Paradigm Shift — Literature Verdict

**No system has an explicit paradigm shift mechanism.** Stagnation is handled by:

| System | Mechanism |
|--------|-----------|
| EvoPrompt | Population diversity via evolutionary operators |
| PromptBreeder | Self-referential mutation of the mutator itself |
| MNL | Batch clustering surfaces patterns individual episodes miss |
| GUM | Natural decay + revision makes old beliefs fade |
| ERL | Failure heuristics naturally explore new directions |

Our paradigm shift (deprecate ALL non-paradigm entries, inject bold new strategies) is the most aggressive and least validated. **Recommendation: keep as opt-in, invest in gentler mechanisms instead.**

## Roadmap — What to Adopt

### Near-term (before hyperagent)
1. **Domain-wise error taxonomy / subject clustering** — cluster failures before reflecting. MNL's subject mapper as inspiration. Wire into Reflector.
2. **Structured notes** — extend Insight to include: approach, mistake summary, strategy, anti-patterns (PlaybookEntry already has the fields)
3. **Accept-if-improves** — gate in optim_step: only commit if batch performance didn't drop

### Medium-term (hyperagent phase)
4. **Self-referential mutation** — PromptBreeder-style: evolve the Reflector prompt itself
5. **Debate-based fitness** — DEEVO-style: use multi-agent debate instead of single-LLM reflection
6. **Learned evolver** — train the evolver on EvolutionLog (state, action, reward_delta) tuples

### Key Insight from MNL

> "Batch-level abstraction reduces spurious updates via concentration inequalities"

A domain-wise error taxonomy is the single highest-value improvement. Instead of reflecting on a flat batch of episodes, first cluster failures by domain/type, then extract patterns per cluster. This gives:
- Higher signal per LLM call
- Natural error taxonomy that transfers across tasks
- Structured notes that include what NOT to do (anti-patterns)
- Foundation for the learned evolver
