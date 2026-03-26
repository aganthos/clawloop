# Competitive Learnings Roadmap

**Date**: 2026-03-17 (updated 2026-03-20)
**Sources**: Honcho (Plastic Labs), MetaClaw v0.2–v0.3.2, GUM paper (arXiv 2505.10831), MetaClaw paper (arXiv 2603.17187)
**Status**: PR1 ✅ MERGED, PR2 ✅ MERGED, PR3 → superseded by `2026-03-20-next-steps-roadmap.md`
**Codex review**: SHIP IT (3 rounds)

> **This document is now a completed reference for PR1/PR2 engineering specs and competitive
> analysis.** For the active launch plan (community release, competition, cloud pro), see
> `docs/plans/2026-03-20-next-steps-roadmap.md`. The PR3 section and Cloud vs Community table
> below are superseded by the three-layer split in the next-steps doc.

---

## PR 1: Playbook Health — Conflict, Redundancy, Attribution, Curation ✅ MERGED

**Branch**: `feat/playbook-curator` → merged to main
**Agent prompt**: `docs/plans/2026-03-19-pr1-playbook-curator-agent-prompt.md`

The playbook (`harness.py:Playbook`) had zero conflict detection, zero redundancy detection, broken attribution (all entries get same signal from every episode), `prune()` was never called, and entries grew without bound. This PR fixed all of it.

### Task 1.1: `PlaybookEntry` schema additions

**File**: `layers/harness.py` (lines 80-111)

Add fields to `PlaybookEntry`. Adopt structured format inspired by MetaClaw's skill schema
(arXiv 2603.17187 §3.1) — their skills have name/description/content/category with anti-pattern
sections, but NO quality scoring, NO dedup, NO conflict detection (just monotonic accumulation).
We improve on this with scoring, decay, and curation:

```python
@dataclass
class PlaybookEntry:
    id: str
    content: str
    helpful: int = 0
    harmful: int = 0
    tags: list[str] = field(default_factory=list)
    source_episode_ids: list[str] = field(default_factory=list)
    # --- NEW ---
    name: str = ""                                           # short slug (MetaClaw-inspired)
    description: str = ""                                    # when-to-trigger one-liner
    anti_patterns: str = ""                                  # what NOT to do (MetaClaw-inspired)
    category: str = "general"                                # structured category for retrieval
    created_at: float = field(default_factory=time.time)     # for decay
    last_activated: float = field(default_factory=time.time) # last episode where relevant
    generation: int = 0                                      # playbook generation when created
    embedding: list[float] | None = None                     # cached semantic embedding
    embedding_model_id: str | None = None                    # tracks which model produced embedding
    embedding_updated_at: float | None = None                # invalidate on content/model change
    decay_rate: float = 0.01                                 # per-entry decay (GUM-inspired)
    superseded_by: str | None = None                         # points to entry that replaced this
```

The `name`/`description`/`anti_patterns`/`category` fields are optional for backward compat.
The Reflector prompt will be updated to produce structured entries when available.

Add `effective_score()` that applies temporal decay:
```python
def effective_score(self) -> float:
    # Decay from last activation; never-used entries decay from creation
    anchor = self.last_activated if self.last_activated != self.created_at else self.created_at
    age_days = (time.time() - anchor) / 86400
    raw = float(self.helpful - self.harmful)
    return raw * math.exp(-self.decay_rate * age_days)
```

Add embedding invalidation:
```python
def needs_reembed(self, current_model_id: str) -> bool:
    """True if embedding is missing, stale (content changed), or from a different model."""
    if self.embedding is None:
        return True
    if self.embedding_model_id != current_model_id:
        return True
    return False
```

Update `to_dict()` / `load_state()` to round-trip all new fields.

### Task 1.2: Semantic similarity infrastructure

**New file**: `core/embeddings.py`

Lightweight embedding interface for playbook entry similarity:
```python
class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

def cosine_similarity(a: list[float], b: list[float]) -> float: ...

def find_similar(
    query_embedding: list[float],
    entries: list[PlaybookEntry],
    threshold: float = 0.75,
) -> list[tuple[PlaybookEntry, float]]: ...
```

Options for embedding provider:
- `LiteLLMEmbedding` — wraps `litellm.embedding()` (already a dependency)
- `MockEmbedding` — returns random vectors for tests

Cache embeddings on `PlaybookEntry.embedding` so we only embed once per entry (re-embed on content update).

### Task 1.3: Retrieve-Classify-Revise pipeline (PlaybookCurator)

**New file**: `core/curator.py`

The GUM paper's three-step pipeline adapted for playbooks:

```python
@dataclass
class CurationResult:
    action: str           # "add" | "merge" | "conflict_resolved" | "skip_redundant"
    entries_affected: list[str]  # IDs
    new_entry: PlaybookEntry | None

class PlaybookCurator:
    def __init__(self, embeddings: EmbeddingProvider, llm: LLMClient, config: CuratorConfig): ...

    def curate_insight(self, insight: Insight, playbook: Playbook) -> CurationResult:
        """Before adding any insight, run retrieve-classify-revise."""
        # 1. RETRIEVE: find similar existing entries by embedding cosine sim
        similar = find_similar(self._embed(insight.content), playbook.entries, threshold=0.6)

        if not similar:
            return CurationResult(action="add", ...)

        # 2. CLASSIFY: LLM classifies relationship
        #    "identical" -> skip (redundant)
        #    "conflicting" -> resolve
        #    "complementary" -> merge
        #    "unrelated" -> add
        classification = self._classify(insight, similar)

        # 3. REVISE based on classification
        if classification == "identical":
            # Boost helpful counter of existing, skip add
            return CurationResult(action="skip_redundant", ...)
        elif classification == "conflicting":
            # LLM resolves conflict using episode evidence as tiebreaker
            resolved = self._resolve_conflict(insight, similar)
            return CurationResult(action="conflict_resolved", new_entry=resolved, ...)
        elif classification == "complementary":
            # Merge into a single stronger entry
            merged = self._merge(insight, similar)
            return CurationResult(action="merge", new_entry=merged, ...)
        else:
            return CurationResult(action="add", ...)
```

**Classification uses heuristics first, LLM only for ambiguous cases** (Codex feedback: avoid LLM for all similar entries):
- `cosine_sim > 0.95` → classify as `identical` without LLM
- `cosine_sim > 0.8` + contradiction keywords detected → classify as `conflicting` without LLM
- `cosine_sim 0.6–0.8` → LLM classifies (ambiguous zone)
- Below threshold → `unrelated`

**Conflict resolution preserves originals** (Codex feedback: never delete on first pass):
- On merge/resolve: create new entry, mark originals with `superseded_by` pointing to new entry
- Superseded entries are hidden from `render()` but retained for audit
- Only removed during consolidation pass after re-validation

**Fallback path** (Codex feedback): if embedding provider or LLM is unavailable, curator degrades to direct `add` with a warning log. Never hard-fail the insight pipeline.

### Task 1.4: Entry-level attribution (fix broken signal)

**File**: `layers/harness.py`, `forward_backward()` (lines 471-505)

Current bug: every episode's reward signal is applied to ALL playbook entries indiscriminately. Fix:

```python
def forward_backward(self, data: Datum) -> Future[FBResult]:
    for episode in data.episodes:
        reward = episode.summary.effective_reward()
        if reward == 0:
            continue
        # NEW: only attribute to relevant entries
        relevant_entries = self._attribute_entries(episode)
        for entry in relevant_entries:
            prev_h, prev_harm = self._pending.playbook_signals.get(entry.id, (0, 0))
            if reward > 0:
                self._pending.playbook_signals[entry.id] = (prev_h + 1, prev_harm)
            else:
                self._pending.playbook_signals[entry.id] = (prev_h, prev_harm + 1)
            # Update last_activated timestamp
            entry.last_activated = time.time()
    ...
```

Attribution strategy (cheapest first, can upgrade later):
1. **Tag match** (free): if episode has `bench` or task tags, match against entry tags
2. **Embedding similarity** (cheap): cosine sim between entry embedding and episode trace embedding
3. **LLM attribution** (expensive, optional): ask LLM "was this entry relevant to this episode?" — only for high-value episodes (|reward| > 0.8)

**Attribution confidence** (Codex feedback): each match produces a `relevance_score: float`. Only update helpful/harmful counters when `relevance_score > attribution_threshold` (default: 0.4). Below threshold → skip, don't attribute noise.

**Support-query separation + generation versioning** (MetaClaw paper arXiv 2603.17187, Algorithm 1):
MetaClaw's key correctness insight: failures drive skill evolution (support set), successes drive
RL (query set). When skills evolve (generation g→g+1), ALL training samples stamped ≤g are flushed.
This is cleaner than a sliding window — it guarantees no stale rewards leak into training.

Adopt for lfx:
- Add `playbook_generation: int` to `Harness` (increments when playbook structurally changes via
  insights, not just score updates). This is distinct from `playbook_version` which tracks any change.
- Add `scored_at_generation: int` to `EpisodeSummary` — stamped when rewards are computed.
- In `forward_backward`: skip episodes where `scored_at_generation < self.playbook_generation`.
  No sliding window needed — generation boundary is the natural cutoff.
- **For Weights layer**: when playbook generation advances, flush stale episodes from the
  Harbor/SkyRL training buffer (mirrors MetaClaw's buffer flush on skill evolution).
- Failed episodes (reward < 0) feed the Reflector (support set). Successful episodes (reward > 0)
  feed the Weights layer (query set). This replaces the current pattern where all episodes feed
  all layers indiscriminately.

Default: tag match + embedding similarity. LLM attribution behind a config flag.

### Task 1.5: Structured playbook rendering

**File**: `layers/harness.py`, `Playbook.render()` (lines 149-159)

Update `render()` to use structured format when entries have name/description/anti_patterns:

```python
def render(self) -> str:
    if not self.entries:
        return ""
    lines = ["## PLAYBOOK"]
    for e in self.entries:
        if e.superseded_by:
            continue  # hide superseded entries
        if e.name and e.description:
            # Structured format (MetaClaw-inspired)
            lines.append(f"### {e.name}")
            lines.append(f"**When**: {e.description}")
            lines.append(e.content)
            if e.anti_patterns:
                lines.append(f"**Anti-pattern**: {e.anti_patterns}")
        else:
            # Legacy flat format
            tag_str = f" [{', '.join(e.tags)}]" if e.tags else ""
            lines.append(f"[{e.id}]{tag_str} :: {e.content}")
    return "\n".join(lines)
```

Update Reflector prompt to request structured output (name, description, content, anti_patterns, category) when generating "add" insights. Backward-compatible: entries without name/description still render in flat format.

### Task 1.6: Playbook consolidation pass ("dreaming")

**File**: `core/curator.py` (extend PlaybookCurator)

Inspired by Honcho's Dreamer agent. Periodic batch consolidation:

```python
def consolidate(self, playbook: Playbook, max_entries: int = 50) -> ConsolidationReport:
    """Periodic 'dreaming' pass — merge, prune, resolve conflicts."""
    # 1. Cluster entries by embedding similarity (simple agglomerative, threshold 0.7)
    clusters = self._cluster_entries(playbook.entries)

    # 2. Within each cluster:
    #    - If >1 entry: LLM merges into single stronger entry
    #    - Detect contradictions within cluster and resolve
    merged_entries = []
    for cluster in clusters:
        if len(cluster) == 1:
            merged_entries.append(cluster[0])
        else:
            merged = self._merge_cluster(cluster)
            merged_entries.append(merged)

    # 3. Prune low-scoring entries (effective_score < 0 after decay)
    pruned = [e for e in merged_entries if e.effective_score() >= 0]

    # 4. If still over budget, keep top max_entries by effective_score
    if len(pruned) > max_entries:
        pruned.sort(key=lambda e: e.effective_score(), reverse=True)
        pruned = pruned[:max_entries]

    return ConsolidationReport(
        before=len(playbook.entries),
        after=len(pruned),
        merged=..., pruned=..., conflicts_resolved=...,
    )
```

Trigger conditions (similar to Honcho's dreaming):
- Every N `optim_step` calls (default: 10)
- When playbook exceeds `max_entries` token budget
- Manually via `harness.consolidate()`

### Task 1.7: Wire up prune() + max size in optim_step

**File**: `layers/harness.py`, `optim_step()` (lines 507-559)

After applying signals and insights, add:
```python
# Auto-prune entries with negative effective_score
pruned = self.playbook.prune_by_effective_score(min_score=0.0)
if pruned:
    log.info("Auto-pruned %d low-scoring playbook entries", pruned)

# Hard cap on entries
MAX_PLAYBOOK_ENTRIES = 100  # configurable
if len(self.playbook.entries) > MAX_PLAYBOOK_ENTRIES:
    self.playbook.entries.sort(key=lambda e: e.effective_score(), reverse=True)
    overflow = len(self.playbook.entries) - MAX_PLAYBOOK_ENTRIES
    self.playbook.entries = self.playbook.entries[:MAX_PLAYBOOK_ENTRIES]
    log.info("Capped playbook at %d entries (removed %d)", MAX_PLAYBOOK_ENTRIES, overflow)
```

### Task 1.8: GEPA-Playbook coherence gate

**File**: `layers/harness.py`, `update_pareto()` (lines 442-452)

When a new GEPA candidate is promoted to `system_prompts[bench]`, check for conflicts with playbook:
```python
def update_pareto(self, bench: str, candidate: PromptCandidate) -> None:
    ...
    best = self.pareto_fronts[bench].best()
    if best:
        self.system_prompts[bench] = best.text
        # NEW: check coherence with playbook
        if self._curator:
            conflicts = self._curator.check_prompt_playbook_coherence(
                best.text, self.playbook
            )
            if conflicts:
                log.warning("GEPA-Playbook conflicts detected: %s", conflicts)
                # Auto-resolve: playbook entries that conflict with the promoted
                # prompt get tagged for review or superseded
```

### Task 1.9: Paradigm breakthrough deprecation

**File**: `core/paradigm.py` + `layers/harness.py`

When paradigm breakthrough fires, it currently only adds new entries. It should also deprecate the old entries it's meant to replace:

- Tag existing non-paradigm entries as `superseded_by` the new paradigm entry ID
- Don't delete them (they may still be useful), but reduce their effective_score multiplier
- In `core/loop.py` line 308: fix `tried_paradigms=[]` TODO — track paradigm contents in a list on AgentState or as a loop-local accumulator

### Task 1.10: Integrate curator into apply_insights

**File**: `layers/harness.py`, `apply_insights()` (lines 339-377)

Route all "add" insights through the curator pipeline before appending:
```python
def apply_insights(self, insights: list[Insight]) -> int:
    insights = self._validate_insights(insights)
    applied = 0
    for insight in insights:
        if insight.action == "add":
            if self._curator:
                result = self._curator.curate_insight(insight, self.playbook)
                # handle result.action: add/merge/skip_redundant/conflict_resolved
                ...
            else:
                # Fallback: direct add (current behavior)
                entry = PlaybookEntry(...)
                self.playbook.add(entry)
            applied += 1
        ...
```

### Task 1.11: Curator observability / metrics

**File**: `core/curator.py`

Track and expose (Codex feedback: need observability for LLM-mediated pipeline):
```python
@dataclass
class CuratorMetrics:
    insights_processed: int = 0
    added: int = 0
    skipped_redundant: int = 0
    merged: int = 0
    conflicts_resolved: int = 0
    fallback_direct_adds: int = 0     # embedding/LLM unavailable
    consolidation_runs: int = 0
    entries_pruned: int = 0
    stale_episodes_skipped: int = 0
```

Expose via `harness.curator_metrics` for experiment logging. `ExperimentLog` should include curator metrics per iteration.

### Task 1.12: Tests

- `test_curator.py` — retrieve-classify-revise with mock embeddings and mock LLM. Use **deterministic recorded LLM responses** (golden fixtures), not live calls
- `test_attribution.py` — verify entry-level attribution only credits relevant entries; verify `relevance_score` threshold filtering
- `test_consolidation.py` — clustering, merging, pruning, max-size cap
- `test_decay.py` — effective_score decreases with age, last_activated resets; never-used entries decay from created_at
- `test_coherence.py` — GEPA-playbook conflict detection
- `test_embedding_invalidation.py` — `needs_reembed()` on content change, model change
- `test_fallback.py` — curator degrades gracefully when embedding/LLM unavailable
- `test_staleness.py` — stale episodes skipped in forward_backward (moved from PR2)
- Update existing `test_harness.py` for new fields, prune wiring, paradigm deprecation, superseded rendering

---

## PR 2: GEPA Completion + Learning Quality ✅ MERGED

**Branch**: `feat/gepa-learning-quality` → merged to main
**Agent prompt**: `docs/plans/2026-03-19-pr2-gepa-learning-quality-agent-prompt.md`
**Depends on**: PR 1 (uses curator, attribution, decay)

### Task 2.1: Data routing for multi-layer learning

**Files**: `core/loop.py`, `learner.py`

**STATUS: NEEDS REWORK** — the original support-query split was implemented
incorrectly and produces zero gradients for GRPO weight training.

**What MetaClaw actually does** (Algorithm 1, arxiv 2603.17187):
- Binary failure detection: task passed or failed
- Failures → support set → skill evolver (prompt/harness learning)
- Non-failures (including partial successes) → RL buffer (weight training)
- Non-failures have **varying PRM rewards** → GRPO has reward variance

**Why our implementation is wrong:**
1. We use `effective_reward() < 0` as threshold. With binary rewards (0→-1,
   1→+1), ALL failures go to harness, ALL successes go to weights.
2. GRPO needs **reward variance within groups**. Single-reward groups give
   advantage=0, grad_norm=0.
3. MetaClaw's RL buffer gets variance from PRM scores. We need it from
   **group sampling** (N responses per prompt, like SkyRL's n_samples_per_prompt).
4. The split applies even in weight-only mode, starving weights of data.

**Correct data flow:**
```python
# Harness: failures only (reflector analyzes what went wrong)
harness_datum = Datum(episodes=support_episodes)
# Weights: ALL episodes (GRPO needs full reward distribution for advantages)
weights_datum = Datum(episodes=episodes)
# Router: all episodes
router_datum = Datum(episodes=episodes)
```

**Additionally needed:** group sampling — collect N episodes per task so GRPO
has intra-group variance. Without this, even with all episodes, per-task
groups of size 1 give zero advantages.

**What IS correct (keep):**
When `playbook_generation` advances (structural playbook change from insights):
- Flush all episodes in Weights training buffer with `scored_at_generation < current_generation`
- This prevents RL from optimizing against pre-adaptation behavior
- Log: `"Generation %d->%d: flushed %d stale episodes from weights buffer"`

This is MetaClaw's key correctness insight and our biggest architectural improvement
from the paper. Without this, the Weights layer can learn to reproduce behaviors that
the playbook has already corrected.

### Task 2.2: GEPA mutation operator

**File**: `layers/harness.py` or new `core/evolution.py`

The Pareto archive exists but has no mechanism to produce new candidates from existing ones. Add:

```python
class PromptEvolver:
    def __init__(self, llm: LLMClient, config: EvolverConfig): ...

    def mutate(self, parent: PromptCandidate, feedback: list[Episode]) -> PromptCandidate:
        """LLM-based targeted mutation. Like GEPA's ASI-guided mutation."""
        # Prompt: "Here is a system prompt and episodes where it failed.
        #          Propose a targeted modification to improve performance."
        # Returns new candidate with parent_id=parent.id, generation=parent.generation+1

    def crossover(self, a: PromptCandidate, b: PromptCandidate) -> PromptCandidate:
        """Combine strengths of two non-dominated candidates."""
        # Prompt: "Candidate A excels at [tasks]. Candidate B excels at [tasks].
        #          Create a hybrid that combines their strengths."
        # Returns new candidate with generation=max(a,b)+1
```

Wire into the learning loop: after `optim_step`, if GEPA is active and new task scores are available, generate 1-2 mutations from the current best + bottom performers.

### Task 2.3: Activity-aware learning intensity

**File**: `core/intensity.py`

From MetaClaw's OMLS. Don't run optim during active user interaction — quality dips during updates hurt UX.

```python
@dataclass
class AdaptiveIntensity:
    ...
    # --- NEW ---
    cooldown_after_request: float = 30.0  # seconds to wait after last user request
    _last_user_request: float = 0.0

    def record_user_activity(self) -> None:
        self._last_user_request = time.time()

    def should_reflect(self, iteration: int) -> bool:
        # Existing logic PLUS:
        if self._last_user_request > 0:
            elapsed = time.time() - self._last_user_request
            if elapsed < self.cooldown_after_request:
                return False  # user is active, defer
        ...
```

Wire `record_user_activity()` into the wrapper/collector when a new user message arrives.

### Task 2.4: Background job system with pluggable tasks

**File**: new `core/background.py`

Codex feedback: "dreaming" appears in both PR1 (playbook consolidation) and PR2 (episode dreaming). Unify into a single background scheduler:

```python
class BackgroundTask(Protocol):
    def should_run(self, state: BackgroundState) -> bool: ...
    def run(self, state: BackgroundState) -> None: ...

@dataclass
class BackgroundState:
    episodes_since_last_run: int
    time_since_last_run: float
    is_user_idle: bool
    playbook: Playbook
    recent_episodes: list[Episode]

class BackgroundScheduler:
    """Single scheduler for all periodic background work."""
    tasks: list[BackgroundTask]

    def tick(self, state: BackgroundState) -> None:
        for task in self.tasks:
            if task.should_run(state):
                task.run(state)
```

Pluggable tasks:
- `PlaybookConsolidation` (from PR1 Task 1.5) — cluster, merge, prune
- `EpisodeDreamer` — cross-episode meta-pattern analysis:
  ```python
  class EpisodeDreamer(BackgroundTask):
      def should_run(self, s): return s.episodes_since_last_run >= 20 and s.is_user_idle
      def run(self, s):
          # LLM analyzes patterns across recent episodes
          # Returns insights tagged "meta-pattern"
  ```

Trigger conditions shared: episode count threshold, time gap, user idle check.

### Task 2.5: Agentic reflector (stretch goal)

**File**: `core/reflector.py`

From Honcho's Dialectic Agent. Instead of a single LLM call, make the reflector an agent loop with tools:

```python
class AgenticReflector(Reflector):
    """Reflector that can search past episodes and reason iteratively."""

    def reflect(self, episodes, playbook, **kw) -> list[Insight]:
        # Agent loop with tools:
        # - search_episodes(query) -> list[Episode]  (semantic search over episode store)
        # - search_playbook(query) -> list[PlaybookEntry]
        # - compare_entries(id_a, id_b) -> diff
        # The agent decides what to investigate, not a fixed pipeline.
```

This is a stretch goal — the standard Reflector works fine for now, but this enables much smarter playbook evolution as the episode history grows.

### Task 2.6: Tests

- `test_support_query_separation.py` — failures go to harness only, successes to weights only
- `test_generation_flush.py` — generation advance flushes stale episodes from weights buffer
- `test_evolution.py` — mutation produces valid candidates with lineage, crossover combines
- `test_activity_intensity.py` — cooldown defers reflection, expired cooldown allows it
- `test_background.py` — scheduler runs tasks when conditions met, skips when not
- `test_dreamer.py` — cross-episode patterns produce tagged insights
- Update `test_loop.py` for tried_paradigms tracking

---

## PR 3: Platform Expansion ⚠️ SUPERSEDED

> **This section is outdated.** Analysis on 2026-03-19 found: OPDAdapter design is wrong
> (SkyRL uses ref model, not endpoint collection), server has NO proxy endpoints (not even
> OpenAI), and some tiered onboarding tasks are already done. PR3 was restructured into three
> tracks in `docs/plans/2026-03-20-next-steps-roadmap.md`:
> - **Track 1: Community Release** — packaging, license, README, quick_start()
> - **Track 2: AgentBeats Competition** — CRM arena, tau2-bench, purple agents
> - **Track 3: Cloud Pro** — auth, persistence, API, billing
>
> The original PR3 spec below is preserved for reference only.

**Branch**: `feat/platform-expansion`
**Independent of PR 1 & 2** (can be parallelized)

### Task 3.1: OPD training paradigm

**File**: `backends/harbor.py` or new `backends/opd.py`

From MetaClaw v0.2. On-Policy Distillation — teacher model provides per-token logprobs on student generations, KL penalty steers student toward teacher distribution.

```python
@dataclass
class OPDConfig:
    teacher_endpoint: str          # vLLM/SGLang endpoint for teacher
    kl_weight: float = 0.1         # KL penalty coefficient
    teacher_model: str = ""        # model name at the endpoint

class OPDAdapter:
    """Collect teacher logprobs for student generations, feed to SkyRL OPD trainer."""

    def collect_teacher_logprobs(self, student_generations: list[str]) -> list[dict]:
        """Query teacher endpoint for per-token logprobs on student outputs."""
        ...
```

SkyRL already has an OPD example at `skyrl/examples/train/on_policy_distillation/`. Wire lfx's Weights layer to optionally use OPD instead of GRPO:
- Add `paradigm: Literal["grpo", "opd"]` to Weights config
- When `opd`, collect teacher logprobs during episode collection and pass to SkyRL's OPD trainer

### Task 3.2: Anthropic Messages API proxy

**File**: `server.py`

From MetaClaw v0.3.2. Currently `server.py` only handles OpenAI-compatible `/v1/chat/completions`. Add `/v1/messages` for Claude-native agent frameworks:

```python
@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Proxy Anthropic Messages API format, inject playbook, collect episodes."""
    body = await request.json()
    # Convert Messages API format to internal format
    # Inject playbook into system prompt
    # Forward to actual Anthropic endpoint
    # Collect episode on response
    ...
```

Key differences from OpenAI format:
- System prompt is a top-level `system` field, not a message
- Tool use is `tool_use` content blocks, not function calls
- Streaming uses SSE with `content_block_delta` events

### Task 3.3: Tiered onboarding

**Files**: `wrapper.py`, `__init__.py`, docs

From MetaClaw's `skills_only` vs `rl` split. Position Harness as zero-config default:

```python
# Tier 1: Just playbook learning (no GPU, no training infra)
import lfx
client = lfx.wrap(openai_client)  # Already works, but make it the highlighted path

# Tier 2: + Router (model selection optimization)
client = lfx.wrap(openai_client, layers=["harness", "router"])

# Tier 3: + Weights (GRPO/OPD, requires GPU/Harbor)
client = lfx.wrap(openai_client, layers=["harness", "router", "weights"])
```

Changes:
- Default `active_layers` in `wrapper.py` to `["harness"]` instead of all three
- Add `lfx.quick_start()` that sets up Tier 1 with sensible defaults
- Update `__init__.py` exports to surface the tiered API
- README / docstring updates explaining the tiers

### Task 3.4: Tests

- `test_opd.py` — OPD adapter collects teacher logprobs, passes to trainer
- `test_messages_proxy.py` — Anthropic Messages API format correctly proxied
- `test_tiered_onboarding.py` — default is harness-only, explicit layers override

---

## Dependency Graph

```
PR 1 (Playbook Health)
  └── PR 2 (GEPA + Learning Quality)  [uses curator, attribution, decay]

PR 3 (Platform Expansion)  [independent, can start in parallel]
```

## Execution Order

1. **PR 1** first — foundation for everything else
2. **PR 2** after PR 1 merges — builds on curator and attribution
3. **PR 3** anytime — fully independent

## Token Budget Estimates

| Component | LLM calls per optim_step | Notes |
|-----------|-------------------------|-------|
| Curator classify | 0-3 (per new insight) | Only when similar entries found |
| Curator resolve/merge | 0-1 | Only on conflicts |
| Consolidation | 1-5 (per cluster) | Only every 10th optim_step |
| Attribution (LLM mode) | 0-N | Optional, off by default |
| Dreamer | 1 | Only every 20 episodes, during idle |
| GEPA mutate | 1-2 | Per iteration when active |
| GEPA crossover | 0-1 | When 2+ non-dominated candidates |

Default configuration (tag+embedding attribution, no LLM attribution) adds ~3-5 LLM calls per optim_step for curation, plus ~1 call per 10 steps for consolidation. Dreaming adds ~1 call per 20 episodes.

---

## Cloud vs Community Edition ⚠️ SUPERSEDED

> **This table is superseded** by the full three-layer split (Harness + Router + Weights)
> in `docs/plans/2026-03-20-next-steps-roadmap.md`. The table below only covered Harness
> features. The next-steps doc covers all three layers plus the upsell funnel.

Features in this roadmap that could differentiate cloud from community:

| Feature | Community (OSS) | Cloud |
|---------|----------------|-------|
| Curator classify | Heuristic-only (cosine thresholds) | + LLM classification for ambiguous cases |
| Conflict resolution | Manual (log warnings, user resolves) | Auto-resolve via LLM + audit trail |
| Consolidation / dreaming | Manual trigger only | Scheduled background service, managed |
| Attribution | Tag match only | + Embedding similarity + LLM attribution |
| Embedding provider | BYO (user provides API key) | Managed embeddings, hosted vector store |
| Episode storage | Local JSONL | Managed episode DB with semantic search |
| GEPA evolution | Archive only (no auto-mutation) | Full mutation + crossover with managed compute |
| Observability | Metrics dict in code | Dashboard, alerting, anomaly detection |
| OPD training | Self-hosted teacher endpoint | Managed teacher inference |
| Background scheduler | In-process thread | Managed serverless workers |

**Principle**: community edition is fully functional for single-agent, local-first use. Cloud adds managed infrastructure, smarter (LLM-powered) curation, multi-agent coordination, and observability. The protocol and data formats are identical — cloud is a superset, not a fork.

---

## Competitive Positioning After This Roadmap

Based on deep reading of MetaClaw paper (arXiv 2603.17187) + Honcho + GUM:

| Capability | MetaClaw (paper) | Honcho | GUM | lfx (after roadmap) |
|------------|-----------------|--------|-----|-------------------|
| Skill/playbook structure | Flat, name/desc/content/category | Propositions with confidence | Confidence-weighted NL | Structured entries with scoring, decay, categories, anti-patterns |
| Conflict detection | None (prompt-only dedup) | Implicit via Revise | Implicit via Revise | Active: heuristic + LLM classify |
| Redundancy handling | None (monotonic growth) | MMR diversity on retrieval | MMR + classify | Active: retrieve-classify-revise + consolidation |
| Pruning | Never | Never (confidence→0) | Never | Auto-prune + max cap + decay |
| Attribution | N/A (no per-skill scoring) | N/A | N/A | Per-entry with relevance threshold |
| Staleness | Generation flush (their best idea) | Decay function | Decay function | Generation flush + per-entry decay (both) |
| Support-query separation | Yes (core novelty) | N/A | N/A | Yes (adopted from MetaClaw) |
| Training paradigms | GRPO only (no hyperparams disclosed) | N/A | N/A | GRPO + OPD |
| Rollback | None | None | None | Atomic snapshot-rollback per layer |
| Regression safety | None | None | None | gate_for_deploy() |
| Multi-objective | Single PRM | N/A | N/A | Composable reward signals with priority |
| User modeling | None | Full (their core value) | Full (their core value) | Via Honcho integration (future) |

**MetaClaw's actual gaps** (from reading the paper, not summaries):
- Skills grow unbounded with NO pruning — will hit context limits
- NO per-skill quality scoring — can't tell which skills help
- PRM is a complete black box — no architecture disclosed
- NO GRPO/LoRA hyperparameters disclosed — hard to reproduce
- NO conflict detection — contradictory skills can coexist silently
- NO rollback — bad skill evolution is irreversible

**lfx's structural advantages after this roadmap**:
1. Only system with active conflict/redundancy resolution
2. Only system with per-entry attribution and quality scoring
3. Only system with atomic rollback + regression gating
4. Only system combining generation flush + temporal decay (both MetaClaw + GUM approaches)
5. Three independent learning layers vs. MetaClaw's two conflated mechanisms
6. Multi-objective reward composition vs. single PRM
