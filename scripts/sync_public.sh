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
            --exclude ".env*" \
            --exclude "*.lock" \
            --exclude "skyrl" \
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

# Sync staging to public repo (--delete removes stale files).
# Exclude paths managed directly in the public repo (not synced from monorepo):
#   .github/      — CI workflows, issue templates
#   .gitmodules   — submodule references (benchmarks, skyrl)
#   benchmarks/   — git submodules
#   clawloop/skyrl/ — git submodule (NovaSky SkyRL, Apache 2.0)
#
# WARNING: --delete removes any file not in staging or excluded here.
# Contributor-added files (SECURITY.md, etc.) will be deleted on next sync
# unless excluded below. Run with --dry-run first to preview changes.
rsync -a --delete \
    --exclude ".git" \
    --exclude ".github" \
    --exclude ".gitignore" \
    --exclude ".gitmodules" \
    --exclude "benchmarks" \
    --exclude "clawloop/skyrl" \
    "$STAGING/" "$PUBLIC_REPO/"

echo "Synced to $PUBLIC_REPO ($(find "$STAGING" -type f | wc -l | tr -d ' ') files)"
