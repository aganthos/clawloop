# Playbook Curator — How It Works

The playbook is a list of reusable strategies injected into agent system prompts. The curator controls how entries are added, updated, merged, pruned, and capped.

## Operating Modes

### Lightweight (no embeddings, no LLM)

Best for narrow agents (n8n workflows, single-task agents) where the playbook stays small and all entries fit in context.

```python
from lfx.core.curator import PlaybookCurator

# Option A: convenience constructor
harness._curator = PlaybookCurator.lightweight(max_entries=50)

# Option B: no curator at all (default Harness behavior)
harness = Harness()  # all entries credited, no dedup
```

**Behavior:**
- Every insight is added directly (no similarity check)
- Every episode credits every active entry (broadcast attribution)
- Auto-prune removes entries with `score < 0` once they have 3+ signals
- Hard cap removes lowest-scoring entries when over `max_entries`
- Consolidation only prunes and caps (no clustering or merging)

### Embeddings-only (no LLM)

Adds similarity-based dedup without LLM costs. Uses heuristic thresholds.

```python
from lfx.core.embeddings import GeminiEmbedding
from lfx.core.curator import PlaybookCurator

harness._curator = PlaybookCurator(
    embeddings=GeminiEmbedding(),  # or MockEmbedding() for tests
)
```

**Behavior:**
- Identical insights (cosine sim > 0.95) are skipped automatically
- Conflicting insights (sim > 0.8 + contradiction keywords) classified by heuristic
- Ambiguous cases (sim 0.6–0.8) classified as "unrelated" (no LLM to ask)
- Attribution uses tag match first, then embedding similarity

### Full (embeddings + LLM)

Complete retrieve-classify-revise pipeline with LLM for ambiguous classification and conflict resolution.

```python
from lfx.core.embeddings import GeminiEmbedding
from lfx.core.curator import PlaybookCurator
from lfx.llm import LiteLLMClient

harness._curator = PlaybookCurator(
    embeddings=GeminiEmbedding(),
    llm=LiteLLMClient(model="gemini/gemini-2.0-flash"),
)
```

**Behavior:**
- Full retrieve-classify-revise: identical/complementary/conflicting/unrelated
- Conflict resolution creates new entry, marks originals as superseded
- Complementary insights merged into stronger combined entry
- Consolidation clusters similar entries and LLM-merges them
- GEPA-playbook coherence checks on prompt promotion

## Update Frequency: The `batch_size` Parameter

The `batch_size` on `EpisodeCollector` determines how often the playbook gets updated. It controls how many episodes are collected before triggering a learning cycle (forward_backward → optim_step).

```python
from lfx.collector import EpisodeCollector

collector = EpisodeCollector(
    pipeline=reward_pipeline,
    batch_size=8,       # learn every 8 episodes (default: 16)
    on_batch=learner.on_batch,
)
```

| Setting | `batch_size` | Effect |
|---------|-------------|--------|
| Aggressive | 1–4 | Updates frequently, noisy signal, more LLM calls |
| Balanced | 8–16 | Default range, good signal-to-noise |
| Conservative | 32–64 | Slower updates, more stable, fewer API calls |

**For n8n / narrow agents:** Use `batch_size=4` or lower — tasks are homogeneous so even small batches give good signal.

**For general agents (OpenClaw-style):** Use `batch_size=16+` — diverse tasks need more data to separate signal from noise.

### Related controls

- **`AdaptiveIntensity.reflect_every_n`** — how often the Reflector fires (default: every 3rd iteration). Higher = fewer LLM calls for insight generation.
- **`CuratorConfig.consolidation_interval`** — how often the dreaming pass runs (default: every 10 optim_steps).
- **`CuratorConfig.max_playbook_entries`** — hard cap on active entries (default: 100).

## Data Flow

```
User request → LLM call → Episode collected
                              ↓
                    EpisodeCollector buffer
                              ↓ (batch_size reached)
                    forward_backward()
                    ├── Skip stale episodes (scored_at_generation < playbook_generation)
                    ├── Attribute rewards to relevant entries (tag match → embedding sim → fallback all)
                    └── Reflector generates Insights (if intensity allows)
                              ↓
                    optim_step()
                    ├── Apply playbook signals (helpful/harmful)
                    ├── Apply insights via curator:
                    │   ├── RETRIEVE similar entries by embedding
                    │   ├── CLASSIFY: identical / complementary / conflicting / unrelated
                    │   └── REVISE: skip / merge / resolve / add
                    ├── Auto-prune (score < 0, signal count >= 3)
                    └── Hard cap (max_entries, lowest effective_score removed)
                              ↓
                    Updated playbook → next system prompt
```

## Embedding Providers

| Provider | Use case | Dependency |
|----------|----------|------------|
| `MockEmbedding` | Tests only | None |
| `GeminiEmbedding` | Production (free tier available) | `GOOGLE_API_KEY` env var |
| `LiteLLMEmbedding` | OpenAI / other providers | `litellm` + provider API key |

```python
# Gemini (recommended — batch API, rate-limit friendly)
from lfx.core.embeddings import GeminiEmbedding
emb = GeminiEmbedding(model="gemini-embedding-001")

# OpenAI via litellm
from lfx.core.embeddings import LiteLLMEmbedding
emb = LiteLLMEmbedding(model="text-embedding-3-small", api_key="sk-...")
```

`GeminiEmbedding` uses the `batchEmbedContents` endpoint (up to 100 texts per call) to minimize API calls under rate limits.

### Hosted / multi-tenant embedding batching

For the hosted version with multiple users, embedding requests should be batched across tenants. The current per-curator batching (`_ensure_embeddings` collects all un-embedded entries into one `embed()` call) works for single-agent deployments. For multi-tenant, a shared embedding queue would collect requests from multiple users and flush in bulk — reducing per-user rate-limit pressure and amortizing API costs. This is a natural fit for the background scheduler planned in PR2 (`BackgroundTask` protocol): a `BatchEmbeddingWorker` that drains the queue every N seconds or when the batch hits a size threshold.

## Temporal Decay

Every entry has a `decay_rate` (default: 0.01/day). `effective_score()` applies exponential decay from the last activation:

```
effective_score = (helpful - harmful) × exp(-decay_rate × days_since_last_activated)
```

Entries that keep getting activated stay strong. Unused entries fade. The auto-prune in `optim_step` removes entries with sustained negative raw score (not decay — to avoid pruning new entries before they've had enough signal).

## Generation Versioning

`playbook_generation` increments on structural changes (insights applied, entries pruned/capped). Episodes stamped with `scored_at_generation` before the current generation are skipped in `forward_backward` — this prevents stale rewards from leaking into training after the playbook has evolved.
