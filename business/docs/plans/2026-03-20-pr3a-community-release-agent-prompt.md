# Agent Prompt: PR3a — Community Release Polish

## Your Task

Prepare the lfx package for public release (v0.1.0). This is packaging, metadata, and API surface cleanup — no architectural changes. The repo must be presentable on GitHub (required for AgentBeats competition eligibility).

## Branch

Create and work on `chore/community-release-v0.1` off `main`.

## What to Do

### 1. License (BSL 1.1)

Create `LICENSE` at repo root. Use the Business Source License 1.1 template:
- **Licensor**: Aganthos AG
- **Licensed Work**: lfx v0.1.0
- **Additional Use Grant**: Non-commercial and evaluation use permitted
- **Change Date**: 2030-03-20 (4 years)
- **Change License**: Apache 2.0

Update `pyproject.toml`: change `license = "MIT"` to `license = "BSL-1.1"`.

### 2. README.md

Create `README.md` at repo root (`/Users/robertmueller/Desktop/aganthos/README.md`). Include:
- One-paragraph description: lfx = Learning from Experience, unified learning API for AI agents
- Three learning layers (Harness, Router, Weights) — one sentence each
- Quickstart (3 code blocks): pip install, lfx.wrap(), lfx-server
- Architecture diagram (ASCII art or mermaid)
- Link to docs/plans/ for detailed specs
- License notice (BSL 1.1)

Keep it under 150 lines. No fluff.

### 3. pyproject.toml metadata

Update `/Users/robertmueller/Desktop/aganthos/pyproject.toml`:
```toml
[project]
description = "Learning from Experience — unified learning API for AI agents"
authors = [{name = "Aganthos AG"}]
license = "BSL-1.1"

[project.urls]
Homepage = "https://github.com/aganthos/lfx"
Repository = "https://github.com/aganthos/lfx"

[project.scripts]
lfx = "lfx.cli:main"
lfx-server = "lfx.server:main"
```

Rename the `[project.optional-dependencies]` group `n8n` to `server` (it contains starlette/uvicorn which are needed for any server use, not just n8n). Keep `n8n` as an alias that includes `server` deps.

Add Python version classifiers.

### 4. `__init__.py` exports

Update `/Users/robertmueller/Desktop/aganthos/lfx/__init__.py`. Add these missing exports:
- `Harness`, `Playbook`, `PlaybookEntry` from `lfx.layers.harness`
- `Reflector`, `ReflectorConfig` from `lfx.core.reflector`
- `AdaptiveIntensity` from `lfx.core.intensity`
- `Episode`, `EpisodeSummary` from `lfx.core.episode`
- `Datum` from `lfx.core.types`

Add `__all__` list. Keep `__version__ = "0.1.0"`.

### 5. Server entry point

Add a `main()` function to `lfx/server.py` (if not already present) that can be called as `lfx-server` console script. Should accept `--host`, `--port`, `--seed-prompt` args matching existing server config.

### 6. `lfx.quick_start()`

Add to `lfx/__init__.py` or `lfx/wrapper.py`:

```python
def quick_start(client, *, seed_prompt: str = "", model: str = ""):
    """One-liner setup: wrap a client with harness learning.

    Returns (wrapped_client, learner) — call learner.start() to begin
    background learning.
    """
    from lfx.collector import EpisodeCollector
    from lfx.learner import AsyncLearner
    from lfx.core.loop import AgentState
    from lfx.layers.harness import Harness

    state = AgentState(harness=Harness(system_prompts={"default": seed_prompt}))
    collector = EpisodeCollector()
    learner = AsyncLearner(agent_state=state, active_layers=["harness"])
    collector.on_batch = learner.on_batch
    wrapped = wrap(client, collector=collector)
    return wrapped, learner
```

Export `quick_start` from `__init__.py`.

### 7. Consistent defaults

In `core/loop.py`, change `learning_loop()` parameter default from `active_layers: list[str] | None = None` to `active_layers: list[str] | None = None` but add at the top of the function body:
```python
if active_layers is None:
    active_layers = ["harness"]
```

This makes `learning_loop` consistent with `AsyncLearner` and `LfxServer` which already default to harness-only.

## Testing

```bash
cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/ -x -v
```

All existing tests MUST pass. Add a test for `quick_start()` in `tests/test_wrapper.py` or new `tests/test_quickstart.py`.

Verify the package builds: `pip install -e ".[dev]"` and `python -c "import lfx; print(lfx.__version__)"`.

## Commit Style

- Format: `fix:`, `feat:`, or `chore:` + one line description
- NO Co-Authored-By lines in commits
- NO multi-line bodies
- Small, focused commits per task

## What NOT to Do

- Do NOT change the learning engine (Harness, Reflector, Curator, etc.)
- Do NOT add proxy endpoints to server.py
- Do NOT change the reward pipeline or extractors
- Do NOT create documentation beyond the README
- Do NOT set up CI/CD (that's a separate task)
