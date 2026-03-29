# Open-Core Repo Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Set up the aganthos/clawloop public repo extraction pipeline and enterprise/ skeleton so we can ship the community edition and layer enterprise (hyperagent) on top without divergence.

**Architecture:** Private monorepo `aganthos/aganthos` is the single source of truth. A `.publicpaths` manifest + sync script projects a subset to the public `aganthos/clawloop` repo. Enterprise code lives in `enterprise/` (minimal skeleton now, interfaces defined when first backend exists). CI enforces the public/private boundary.

**Tech Stack:** Python 3.11+, bash, git, hatchling (build), pytest

**Design doc:** Based on brainstorming sessions 2026-03-27 (Claude + Codex, 3 design rounds + 4 review rounds)

**Note:** This plan file is a local gitignored working file (docs/ is in .gitignore). It is NOT part of the repository.

**Parallelism:** Tasks are grouped into phases. Tasks within a phase can run in parallel. Tasks across phases are sequential.

**Version:** v0.0.1 (test release, repo starts PRIVATE to verify filtering before going public)

```
Phase 1 (parallel):  Task 1 (repo hygiene) + Task 2 (enterprise skeleton)        ✅ DONE
                              │
Phase 2 (sequential): Task 3 (release files) → Task 4 (sync pipeline + private repo + CI)
                              │
Phase 3 (standalone): Task 5 (cloud hook)
                              │
--- future (separate plans) ---
Phase 4: Evolution interfaces (Evolver Protocol, PatchSet, TraceSchema)
Phase 5: First Evolver backend (reflector)
Phase 6: Cloud API (enterprise/cloud/)
Phase 7: Guided mutation backend (GEPA++)
Phase 8: Hyperagent backend (DGM-H evolutionary loop, arxiv.org/abs/2603.19461)
```

---

## Phase 1: Foundation (parallel)

### Task 1: Repo hygiene — clean public-facing code

**Files:**
- Modify: `.gitignore`
- Modify: 14 Python files in `clawloop/` (54 residual `lfx` references)
- Modify: `clawloop/guide/playbook-curator.md` (~10 import examples)
- Modify: `clawloop/static/index.html:285,618` (logo text, confirm dialog)
- Modify: `examples/demo_math.py:16-18,217-218` (remove hardcoded key/URL defaults)
- Modify: `examples/recipes/arithmetic.py:197`, `a2a_crmarena.py:172`, `guess_number.py:189`, `harbor_bfcl.py:208`
- Move: `clawloop/static/diagrams/05-editions.html` → `pitch/diagrams/` (pricing)
- Move: `clawloop/static/diagrams/slide-03-editions.html` → `pitch/diagrams/` (pricing)

**Context:** 54 residual `lfx` references across 14 Python files. Examples hardcode `kuhhandel-bench-key` and `127.0.0.1:8317`. Static diagrams contain `$149/mo` pricing. `.gitignore` doesn't cover root `.env`.

- [ ] **Step 1: Add `.env` to root `.gitignore`**

Add after line 5 in `.gitignore`:

```
# Secrets
.env
```

- [ ] **Step 2: Run `git diff .gitignore` to verify only the `.env` line was added**

Run: `git diff .gitignore`
Expected: one added line `+.env`

- [ ] **Step 3: Rename all `lfx` → `clawloop` in Python files (strings/comments only)**

For each file, replace residual `lfx` references:
- Thread names: `"lfx-learner"` → `"clawloop-learner"`
- Logger names: `logging.getLogger("lfx")` → `logging.getLogger("clawloop")`
- Prog names: `prog="lfx"` → `prog="clawloop"`
- Docstrings: `"lfx-server"` → `"clawloop-server"`, `"lfx learning pipeline"` → `"clawloop learning pipeline"`
- Comments: `"lfx-specific"` → `"clawloop-specific"`, `"lfx/core/layer.py"` → `"clawloop/core/layer.py"`
- Pip install: `"pip install 'lfx[otel]'"` → `"pip install 'clawloop[otel]'"`
- Pip install in harbor.py: `"pip install lfx[harbor]"` → `"pip install harbor"` (no `[harbor]` extra exists in pyproject.toml)
- Agent names: `"lfx-purple-agent"` → `"clawloop-purple-agent"`, `"lfx-entropic-purple-agent"` → `"clawloop-entropic-purple-agent"`
- Code examples in comments: `lfx.wrap()` → `clawloop.wrap()`
- File references: `lfx/adapters/...` → `clawloop/adapters/...`, `"lfx RewardSignals"` → `"clawloop RewardSignals"`
- `__init__.py` docstring: `"""LfX — ..."` → `"""ClawLoop — Learning from Experience unified learning API."""`

Use `replace_all` where the pattern is unambiguous within a file.

- [ ] **Step 4: Update `clawloop/guide/playbook-curator.md`**

Replace all `from lfx.` → `from clawloop.` in import examples (~10 occurrences).

- [ ] **Step 5: Update `clawloop/static/index.html`**

Line 285: `<span class="logo">LFX</span>` → `<span class="logo">ClawLoop</span>`
Line 618: `"Reset lfx server?"` → `"Reset ClawLoop server?"`

- [ ] **Step 6: Sanitize example defaults**

In `examples/demo_math.py` lines 217-218, change:
```python
api_base = os.environ.get("CLAWLOOP_API_BASE", "http://127.0.0.1:8317/v1")
api_key = os.environ.get("CLAWLOOP_API_KEY", "kuhhandel-bench-key")
```
to:
```python
api_base = os.environ.get("CLAWLOOP_API_BASE", "http://localhost:11434/v1")
api_key = os.environ.get("CLAWLOOP_API_KEY", "your-api-key")
```

Update docstring at lines 16-17 to match. Same pattern for all 4 recipe files (change default `--api-base` from `http://127.0.0.1:8317/v1` to `http://localhost:11434/v1`).

- [ ] **Step 7: Move pricing diagrams out of public tree**

```bash
mkdir -p /Users/robertmueller/Desktop/aganthos/pitch/diagrams
mv clawloop/static/diagrams/05-editions.html pitch/diagrams/
mv clawloop/static/diagrams/slide-03-editions.html pitch/diagrams/
```

Also remove pricing reference from `clawloop/static/clawloop-architecture.html` — find the `$149+/mo` text and replace with `Cloud Pro` (no price).

- [ ] **Step 8: Run tests to verify nothing is broken**

Run: `cd /Users/robertmueller/Desktop/aganthos && python -m pytest tests/ -x -q`
Expected: all tests pass (string/comment changes should not affect behavior)

- [ ] **Step 9: Commit**

```bash
git add .gitignore clawloop/ examples/ pitch/diagrams/
git commit -m "chore: clean residual lfx refs and sanitize examples for public release"
```

---

### Task 2: Enterprise skeleton + AGENTS.md boundary rules

**Files:**
- Create: `enterprise/__init__.py`
- Create: `enterprise/tests/__init__.py`
- Create: `enterprise/tests/conftest.py`
- Modify: `ressources/AGENTS.md`

**Context:** Minimal enterprise skeleton — just enough to establish the filesystem boundary and test infrastructure. NO interfaces, NO Protocol definitions (deferred until first backend exists per Codex review round 1-4). AGENTS.md makes the boundary unambiguous for coding agents.

- [ ] **Step 1: Create enterprise directory structure**

```bash
cd /Users/robertmueller/Desktop/aganthos
mkdir -p enterprise/tests
```

- [ ] **Step 2: Create `enterprise/__init__.py`**

```python
"""Aganthos enterprise — proprietary learning algorithms. NEVER published.

This directory contains enterprise-only code that is never synced to
the public aganthos/clawloop repo. See ressources/AGENTS.md for boundary rules.

Future structure (created when first backend is built):
  enterprise/evolution/interfaces/  — Evolver protocol, PatchSet, TraceSchema
  enterprise/evolution/backends/    — reflector, guided_mutation, dgm_h (hyperagent)
  enterprise/evolution/core/        — archive, selection, evaluation, lineage
  enterprise/cloud/                 — API server + async improvement workers
"""
```

- [ ] **Step 3: Create `enterprise/tests/__init__.py`**

Empty file.

- [ ] **Step 4: Create `enterprise/tests/conftest.py`**

```python
"""Enterprise test configuration.

Enterprise tests CAN import from both clawloop and enterprise.
Community tests (tests/) must NEVER import from enterprise.
"""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-mark all enterprise tests."""
    for item in items:
        item.add_marker(pytest.mark.enterprise)
```

- [ ] **Step 5: Update `ressources/AGENTS.md` with boundary rules**

Append to the existing file:

```markdown

## Public/Private Boundary

This monorepo contains both public (community) and private (enterprise) code.

### Directory classification
- `clawloop/`, `tests/`, `examples/` → **PUBLIC** (synced to github.com/aganthos/clawloop)
- `enterprise/` → **PRIVATE** (proprietary algorithms, never published)
- Everything else (`docs/`, `pitch/`, `configs/`, `benchmarks/`, `ressources/`, `scripts/`) → **PRIVATE**

### Rules for coding agents
1. Code in `clawloop/` must NEVER import from `enterprise/`
2. Code in `enterprise/` CAN import from `clawloop` (it extends community)
3. Tests in `tests/` must NEVER import from `enterprise/`
4. Enterprise tests live in `enterprise/tests/`
5. New files in `clawloop/` automatically become public
6. New files in `enterprise/` stay private
7. The `.publicpaths` manifest is the source of truth for what gets synced

### Architecture (future — interfaces defined when first backend exists)
- `clawloop.core.layer.Layer` Protocol = what executes (community)
- Enterprise will use an Evolver Protocol = what proposes improvements to what executes
- Planned evolution backends: reflector → guided_mutation → dgm_h (hyperagent)
- Cloud hook: `clawloop.wrap(client, cloud_url=..., cloud_api_key=...)` sends traces, pulls patches
- See arxiv.org/abs/2603.19461 (HyperAgents) for the long-term hyperagent vision
```

- [ ] **Step 6: Commit**

```bash
git add enterprise/ ressources/AGENTS.md
git commit -m "feat: enterprise skeleton with boundary rules"
```

---

## Phase 2: Sync Infrastructure (sequential, after Phase 1)

**Note:** Tasks 3 and 4 must run in this order. Task 3 creates the sync infrastructure. Task 4 creates the release files (LICENSE, CONTRIBUTING.md, CHANGELOG.md) that `.publicpaths` references and then performs the actual public repo extraction.

### Task 3: Public sync pipeline — `.publicpaths` + scripts + CI guard

**Files:**
- Create: `.publicpaths`
- Create: `scripts/sync_public.sh`
- Create: `scripts/audit_public.sh`
- Create: `tests/conftest.py` (add import guard)

**Context:** The sync script uses a staging-dir approach to prevent stale files. The audit script runs in CI to catch leaks. A conftest guard prevents community tests from importing enterprise code. `.publicpaths` is the SINGLE source of truth — no hardcoded exclusions anywhere else.

- [ ] **Step 1: Create `.publicpaths`**

```
# Files and directories synced to aganthos/clawloop public repo.
# Default = private. Only listed paths go public.
# One path per line. Directories end with /. Comments start with #.

clawloop/
tests/
examples/
README.md
LICENSE
pyproject.toml
CONTRIBUTING.md
CHANGELOG.md
.gitignore
.gitmodules
```

- [ ] **Step 2: Create `scripts/sync_public.sh`**

```bash
#!/usr/bin/env bash
# Sync public files from monorepo to the aganthos/clawloop public repo.
# Uses a staging directory to prevent stale files.
# .publicpaths is the SINGLE source of truth — no hardcoded exclusions.
#
# Usage: ./scripts/sync_public.sh /path/to/clawloop-public-repo
set -euo pipefail

REPO_ROOT="$(git -C "$(dirname "$0")/.." rev-parse --show-toplevel)"
PUBLIC_REPO="${1:?Usage: sync_public.sh /path/to/clawloop-public}"
MANIFEST="$REPO_ROOT/.publicpaths"
STAGING="$(mktemp -d)"

trap 'rm -rf "$STAGING"' EXIT

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: $MANIFEST not found" >&2
    exit 1
fi

# Reject dangerous paths
while IFS= read -r p; do
    [[ -z "$p" || "$p" =~ ^[[:space:]]*# ]] && continue
    p="${p%/}"
    if [[ "$p" == /* || "$p" == *..* ]]; then
        echo "ERROR: unsafe path in .publicpaths: $p" >&2
        exit 1
    fi
    if [[ -d "$REPO_ROOT/$p" ]]; then
        rsync -a --copy-links --safe-links \
            --exclude "__pycache__" \
            --exclude ".DS_Store" \
            --exclude "*.pyc" \
            --exclude ".git" \
            --exclude ".claude" \
            --max-size=50M \
            "$REPO_ROOT/$p/" "$STAGING/$p/"
    elif [[ -f "$REPO_ROOT/$p" ]]; then
        mkdir -p "$(dirname "$STAGING/$p")"
        cp "$REPO_ROOT/$p" "$STAGING/$p"
    else
        echo "WARNING: $p not found in repo" >&2
    fi
done < "$MANIFEST"

# Verify public repo state before syncing
cd "$PUBLIC_REPO"
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")
if [[ "$CURRENT_BRANCH" != "main" ]]; then
    echo "ERROR: Public repo is not on main branch (on $CURRENT_BRANCH)" >&2
    exit 1
fi
git fetch origin main --quiet 2>/dev/null || true
if git log --oneline origin/main..HEAD 2>/dev/null | grep -q .; then
    echo "ERROR: Public repo has unpushed commits — diverged from origin" >&2
    exit 1
fi
if git log --oneline HEAD..origin/main 2>/dev/null | grep -q .; then
    echo "ERROR: Public repo is behind origin/main — pull first" >&2
    exit 1
fi
cd "$REPO_ROOT"

# Sync staging to public repo (--delete removes stale files)
rsync -a --delete \
    --exclude ".git" \
    "$STAGING/" "$PUBLIC_REPO/"

echo "Synced to $PUBLIC_REPO ($(find "$STAGING" -type f | wc -l | tr -d ' ') files)"
```

- [ ] **Step 3: Make sync script executable**

```bash
chmod +x scripts/sync_public.sh
```

- [ ] **Step 4: Create `scripts/audit_public.sh`**

```bash
#!/usr/bin/env bash
# Leak detection — run in CI on every push to main.
# Checks that public code doesn't reference enterprise internals.
set -euo pipefail

ERRORS=0

echo "=== Audit: public/private boundary ==="

# Use rg if available, fall back to grep
if command -v rg &>/dev/null; then
    SEARCH_CMD="rg -l"
else
    SEARCH_CMD="grep -rl"
fi

# 1. No enterprise imports in public code
if $SEARCH_CMD "from enterprise\|import enterprise" clawloop/ tests/ examples/ 2>/dev/null; then
    echo "FAIL: Public code imports from enterprise/"
    ERRORS=$((ERRORS + 1))
fi

# 2. No enterprise references in public docs
if $SEARCH_CMD "enterprise/" README.md CONTRIBUTING.md examples/ 2>/dev/null; then
    echo "FAIL: Public docs reference enterprise/"
    ERRORS=$((ERRORS + 1))
fi

# 3. No internal secrets/URLs in public code
if $SEARCH_CMD "kuhhandel-bench-key" clawloop/ tests/ examples/ README.md 2>/dev/null; then
    echo "FAIL: Internal API key found in public code"
    ERRORS=$((ERRORS + 1))
fi

# 4. No symlinks pointing outside public tree
while IFS= read -r link; do
    target="$(readlink "$link")"
    if [[ "$target" == *enterprise* || "$target" == *docs/* || "$target" == *pitch/* ]]; then
        echo "FAIL: Symlink $link points to private path: $target"
        ERRORS=$((ERRORS + 1))
    fi
done < <(find clawloop tests examples -type l 2>/dev/null || true)

# 5. Build and inspect package for leaks
rm -rf dist/
python -m build --sdist --wheel 2>/dev/null
python3 -c "
import tarfile, zipfile, glob, sys
PRIVATE = ('enterprise', 'pitch', 'configs/', 'benchmarks/', 'ressources/')
for f in glob.glob('dist/*.tar.gz'):
    t = tarfile.open(f)
    for m in t.getmembers():
        if any(p in m.name for p in PRIVATE):
            print(f'LEAK in {f}: {m.name}')
            sys.exit(1)
for f in glob.glob('dist/*.whl'):
    z = zipfile.ZipFile(f)
    for n in z.namelist():
        if any(p in n for p in PRIVATE):
            print(f'LEAK in {f}: {n}')
            sys.exit(1)
print('Package audit: clean')
"

if [[ $ERRORS -gt 0 ]]; then
    echo "=== FAILED: $ERRORS leak(s) detected ==="
    exit 1
fi

echo "=== PASSED: no leaks detected ==="
```

- [ ] **Step 5: Make audit script executable**

```bash
chmod +x scripts/audit_public.sh
```

- [ ] **Step 6: Add import guard to `tests/conftest.py`**

If `tests/conftest.py` exists, append to it. If not, create it:

```python
def pytest_collection_modifyitems(session, config, items):
    """Guard: community tests must never import from enterprise."""
    import ast
    from pathlib import Path

    for item in items:
        fpath = Path(item.fspath)
        # Only check tests/ (not enterprise/tests/)
        if "enterprise" in str(fpath):
            continue
        try:
            tree = ast.parse(fpath.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("enterprise"):
                raise ValueError(
                    f"BOUNDARY VIOLATION: {fpath} imports from enterprise "
                    f"(line {node.lineno}). Community tests must NEVER "
                    f"import from enterprise/."
                )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("enterprise"):
                        raise ValueError(
                            f"BOUNDARY VIOLATION: {fpath} imports from enterprise "
                            f"(line {node.lineno}). Community tests must NEVER "
                            f"import from enterprise/."
                        )
```

- [ ] **Step 7: Run audit script locally**

Run: `cd /Users/robertmueller/Desktop/aganthos && bash scripts/audit_public.sh`
Expected: `PASSED: no leaks detected`

- [ ] **Step 8: Run tests to verify conftest guard works**

Run: `python -m pytest tests/ -x -q`
Expected: all pass (no enterprise imports in community tests)

- [ ] **Step 9: Commit**

```bash
git add .publicpaths scripts/ tests/conftest.py
git commit -m "feat: public sync pipeline with leak detection and import guard"
```

---

### Task 4: Public repo extraction — create aganthos/clawloop

**Files:**
- Copy: `clawloop/LICENSE` → `LICENSE` (repo root)
- Create: `CONTRIBUTING.md`
- Create: `CHANGELOG.md`
- Modify: `pyproject.toml` (add metadata)

**Context:** This task creates the actual public repo on GitHub and does the first sync. LICENSE copies to repo root (standard location). pyproject.toml gets author/URL metadata. CONTRIBUTING.md and CHANGELOG.md are minimal stubs.

- [ ] **Step 1: Copy LICENSE to repo root**

```bash
cp clawloop/LICENSE LICENSE
```

Keep the copy in `clawloop/` for now (hatchling includes it in the wheel).

- [ ] **Step 2: Update pyproject.toml metadata**

Add the following fields inside `[project]` (after line 10 `license = "BSL-1.1"`, before line 11 `dependencies`):

```toml
authors = [{name = "Aganthos GmbH"}]
readme = "README.md"
keywords = ["ai", "agents", "learning", "llm", "reinforcement-learning"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
```

Add a new `[project.urls]` section between `[project.scripts]` and `[tool.hatch.build.targets.wheel]` (i.e., between current lines 41 and 43):

```toml
[project.urls]
Homepage = "https://github.com/aganthos/clawloop"
Repository = "https://github.com/aganthos/clawloop"
```

Rename the `n8n` optional dependency group to `server`:

```toml
server = [
    "starlette>=0.27",
    "uvicorn>=0.20",
    "httpx>=0.24",
]
```

Add `clawloop-server` script entry:
```toml
[project.scripts]
clawloop = "clawloop.cli:main"
clawloop-server = "clawloop.server:main"
```

- [ ] **Step 3: Create `CONTRIBUTING.md`**

```markdown
# Contributing to ClawLoop

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/aganthos/clawloop.git
cd clawloop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/
```

## Guidelines

- Run `pytest tests/ -x` before submitting a PR
- Follow existing code patterns
- One commit per logical change: `feat:`, `fix:`, or `chore:` prefix

## License

By contributing, you agree that your contributions will be licensed under the BSL 1.1 license.
```

- [ ] **Step 4: Create `CHANGELOG.md`**

```markdown
# Changelog

## 0.0.1 (2026-03-27)

Initial public release.

- Three learning layers: Harness (prompt optimization), Router (model selection), Weights (training)
- Unified Layer Protocol with atomic rollback and regression gating
- Live mode: `clawloop.wrap()` + `EpisodeCollector` + `AsyncLearner`
- Benchmark adapters: CRM Arena, tau2-bench (stub), Harbor
- SkyRL backend for LoRA/GRPO training
- OTel/OpenInference export
- n8n integration server
```

- [ ] **Step 5: Run tests and audit**

```bash
python -m pytest tests/ -x -q
bash scripts/audit_public.sh
```

Expected: all pass, no leaks.

- [ ] **Step 6: Commit the metadata changes**

```bash
git add LICENSE CONTRIBUTING.md CHANGELOG.md pyproject.toml
git commit -m "chore: add release metadata, LICENSE at root, CONTRIBUTING, CHANGELOG"
```

- [ ] **Step 7: Create the public repo on GitHub (idempotent)**

```bash
if ! gh repo view aganthos/clawloop &>/dev/null 2>&1; then
    gh repo create aganthos/clawloop --public --description "ClawLoop — Learning from Experience unified learning API"
else
    echo "Repo aganthos/clawloop already exists, skipping creation"
fi
```

- [ ] **Step 8: Clone the public repo and run first sync**

```bash
cd /tmp
git clone git@github.com:aganthos/clawloop.git clawloop-public
cd /Users/robertmueller/Desktop/aganthos
bash scripts/sync_public.sh /tmp/clawloop-public
```

- [ ] **Step 9: Add `/enterprise/` to the PUBLIC repo's .gitignore (belt-and-suspenders)**

```bash
echo -e "\n# Enterprise code must never exist in public repo\n/enterprise/" >> /tmp/clawloop-public/.gitignore
```

- [ ] **Step 10: Verify the public repo contents**

```bash
cd /tmp/clawloop-public
ls  # should show: clawloop/ tests/ examples/ README.md LICENSE pyproject.toml ...
# Verify NO enterprise, docs, pitch, configs, benchmarks
ls enterprise 2>&1 | grep -q "No such file"
ls docs 2>&1 | grep -q "No such file"
# Verify no pricing diagrams (moved out in Task 1)
ls clawloop/static/diagrams/05-editions.html 2>&1 | grep -q "No such file"
```

- [ ] **Step 11: Commit and push the public repo**

```bash
cd /tmp/clawloop-public
git add -A
git commit -m "feat: initial public release of ClawLoop v0.0.1"
git tag v0.0.1
git push origin main
git push origin v0.0.1
```

- [ ] **Step 12: Verify on GitHub**

```bash
gh repo view aganthos/clawloop --web
```

---

## Phase 3: Extension Points (standalone, after Phase 2)

### Task 5: Cloud hook — add `cloud_url` and `trace_level` to `wrap()`

**Files:**
- Modify: `clawloop/wrapper.py`
- Create: `tests/test_cloud_hook.py`

**Context:** When `cloud_url` is set on `wrap()`, episodes are async-posted to the cloud API. The cloud returns improvements via a pull cycle. `trace_level` controls how much data is sent. Strategy selection is server-side (not in client code). This is the thin transport layer — no enterprise logic.

- [ ] **Step 1: Write failing test — wrap() accepts cloud_url param**

In `tests/test_cloud_hook.py`:

```python
"""Tests for the cloud_url hook in wrap()."""

from unittest.mock import MagicMock

from clawloop.wrapper import wrap, WrappedClient
from clawloop.collector import EpisodeCollector
from clawloop.core.reward import RewardPipeline


def _make_collector() -> EpisodeCollector:
    """Create a minimal EpisodeCollector with a mock pipeline."""
    return EpisodeCollector(pipeline=MagicMock(spec=RewardPipeline))


def test_wrap_accepts_cloud_url():
    """wrap() should accept cloud_url and cloud_api_key params."""

    class FakeClient:
        def complete(self, messages, **kw):
            return "hello"

    wrapped = wrap(
        FakeClient(),
        _make_collector(),
        cloud_url="https://api.clawloop.com",
        cloud_api_key="cl-test-key",
    )
    assert isinstance(wrapped, WrappedClient)
    assert wrapped._cloud_url == "https://api.clawloop.com"
    assert wrapped._cloud_api_key == "cl-test-key"


def test_wrap_default_no_cloud():
    """Without cloud_url, no cloud config is set."""

    class FakeClient:
        def complete(self, messages, **kw):
            return "hello"

    wrapped = wrap(FakeClient(), _make_collector())
    assert wrapped._cloud_url is None
    assert wrapped._cloud_api_key is None


def test_wrap_accepts_trace_level():
    """wrap() should accept trace_level param."""

    class FakeClient:
        def complete(self, messages, **kw):
            return "hello"

    wrapped = wrap(
        FakeClient(),
        _make_collector(),
        cloud_url="https://api.clawloop.com",
        cloud_api_key="cl-key",
        trace_level="full",
    )
    assert wrapped._trace_level == "full"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cloud_hook.py -v`
Expected: FAIL — `wrap()` doesn't accept `cloud_url` yet.

- [ ] **Step 3: Add cloud params to `wrap()` and `WrappedClient`**

In `clawloop/wrapper.py`, update `WrappedClient.__init__`:

```python
def __init__(
    self,
    client: Any,
    collector: EpisodeCollector,
    *,
    tracer: Any = None,
    intensity: AdaptiveIntensity | None = None,
    cloud_url: str | None = None,
    cloud_api_key: str | None = None,
    trace_level: str = "minimal",
) -> None:
    self._client = client
    self._collector = collector
    self._tracer = tracer
    self._intensity = intensity
    self._cloud_url = cloud_url
    self._cloud_api_key = cloud_api_key
    self._trace_level = trace_level

    self._llm_kind_attr: str | None = None
    self._llm_kind_value: str | None = None
    if tracer:
        self._llm_kind_attr, self._llm_kind_value = resolve_oi_span_kind()
```

Update the `wrap()` function signature:

```python
def wrap(
    client: Any,
    collector: EpisodeCollector,
    *,
    tracer: Any = None,
    intensity: AdaptiveIntensity | None = None,
    cloud_url: str | None = None,
    cloud_api_key: str | None = None,
    trace_level: str = "minimal",
) -> WrappedClient:
    """Wrap an LLMClient with live-mode episode collection.

    Usage::

        # Local learning only:
        wrapped = clawloop.wrap(my_client, collector=collector)

        # With cloud learning (better algorithms, cross-client transfer):
        wrapped = clawloop.wrap(
            my_client, collector=collector,
            cloud_url="https://api.clawloop.com",
            cloud_api_key="cl-...",
            trace_level="full",  # minimal | standard | full
        )
    """
    return WrappedClient(
        client, collector,
        tracer=tracer, intensity=intensity,
        cloud_url=cloud_url, cloud_api_key=cloud_api_key,
        trace_level=trace_level,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cloud_hook.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite for regressions**

Run: `pytest tests/ -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add clawloop/wrapper.py tests/test_cloud_hook.py
git commit -m "feat: add cloud_url and trace_level params to wrap()"
```

---

## Post-Implementation

After all phases complete:

- [ ] **Run full test suite:** `pytest tests/ -x -q`
- [ ] **Run audit:** `bash scripts/audit_public.sh`
- [ ] **Sync to public repo:** `bash scripts/sync_public.sh /path/to/clawloop-public`
- [ ] **Push public repo with new tag**
