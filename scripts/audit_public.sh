#!/usr/bin/env bash
# Leak detection — run in CI on every push to main.
# Checks that public code doesn't reference enterprise internals.
set -euo pipefail

ERRORS=0

echo "=== Audit: public/private boundary ==="

# 1. No enterprise imports in public code (actual Python import statements)
#    Uses line-anchored pattern to avoid matching strings/comments in guard code.
#    Excludes .claude/ dirs (IDE config, not shipped).
IMPORT_PATTERN='^\s*(from|import)\s+enterprise'
MATCHES=$(grep -rn --include="*.py" -E "$IMPORT_PATTERN" clawloop/ tests/ examples/ 2>/dev/null \
    | grep -v '\.claude/' || true)
if [[ -n "$MATCHES" ]]; then
    echo "FAIL: Public code imports from enterprise:"
    echo "$MATCHES"
    ERRORS=$((ERRORS + 1))
fi

# 2. No enterprise references in public docs
MATCHES=$(grep -rl "enterprise/" README.md CONTRIBUTING.md examples/ 2>/dev/null \
    | grep -v '\.claude/' || true)
if [[ -n "$MATCHES" ]]; then
    echo "FAIL: Public docs reference enterprise/"
    echo "$MATCHES"
    ERRORS=$((ERRORS + 1))
fi

# 3. No internal secrets/URLs in public code (exclude .claude/ IDE config)
MATCHES=$(grep -rl "kuhhandel-bench-key" clawloop/ tests/ examples/ README.md 2>/dev/null \
    | grep -v '\.claude/' || true)
if [[ -n "$MATCHES" ]]; then
    echo "FAIL: Internal API key found in public code"
    echo "$MATCHES"
    ERRORS=$((ERRORS + 1))
fi

# 4. No residual lfx branding in public Python code
#    Allows: variable names referencing external scripts (e.g., lfx_server in car.py)
MATCHES=$(grep -rn --include="*.py" -iw 'lfx' clawloop/ tests/ examples/ 2>/dev/null \
    | grep -v '\.claude/' || true)
if [[ -n "$MATCHES" ]]; then
    echo "FAIL: Residual lfx branding found in public code"
    echo "$MATCHES"
    ERRORS=$((ERRORS + 1))
fi

# 5. AST-based enterprise import scan (catches multi-line imports)
#    Scans clawloop/, tests/, examples/ for any static import from enterprise.
python3 -c "
import ast, sys
from pathlib import Path

errors = []
for d in ['clawloop', 'tests', 'examples']:
    for f in Path(d).rglob('*.py'):
        # Only skip the enterprise_clawloop directory itself, not substring matches
        if str(f).startswith('enterprise_clawloop/') or '/enterprise_clawloop/' in str(f):
            continue
        if '.claude' in str(f):
            continue
        # Flag enterprise-themed filenames inside public dirs
        if 'enterprise' in f.name:
            errors.append(f'{f}: enterprise-themed filename in public directory')
            continue
        try:
            tree = ast.parse(f.read_text())
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('enterprise'):
                errors.append(f'{f}:{node.lineno}: from {node.module} import ...')
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('enterprise'):
                        errors.append(f'{f}:{node.lineno}: import {alias.name}')
if errors:
    for e in errors:
        print(f'BOUNDARY VIOLATION: {e}')
    sys.exit(1)
print('AST import scan: clean')
" || { echo "FAIL: AST-based import scan found enterprise imports"; ERRORS=$((ERRORS + 1)); }

# 6. No symlinks pointing outside public tree
while IFS= read -r link; do
    target="$(readlink "$link")"
    if [[ "$target" == *enterprise* || "$target" == *docs/* || "$target" == *pitch/* ]]; then
        echo "FAIL: Symlink $link points to private path: $target"
        ERRORS=$((ERRORS + 1))
    fi
done < <(find clawloop tests examples -type l 2>/dev/null || true)

# 7. Build and inspect package for leaks
rm -rf dist/
python -m build --sdist --wheel
python3 -c "
import tarfile, zipfile, glob, sys, re

# Match top-level private dirs (after the package prefix like 'clawloop-0.0.1/')
# These patterns match paths where the private dir is the first real directory.
PRIVATE_DIRS = ('enterprise', 'enterprise_clawloop', 'pitch', 'configs', 'ressources')

def is_private(path):
    # Strip sdist prefix (e.g., 'clawloop-0.0.1/enterprise/...')
    parts = path.split('/')
    # Skip the first component if it looks like a package prefix (sdist)
    if len(parts) > 1 and '-' in parts[0]:
        parts = parts[1:]
    # Check if first real directory is private
    return len(parts) > 0 and parts[0] in PRIVATE_DIRS

for f in glob.glob('dist/*.tar.gz'):
    t = tarfile.open(f)
    for m in t.getmembers():
        if is_private(m.name):
            print(f'LEAK in {f}: {m.name}')
            sys.exit(1)
for f in glob.glob('dist/*.whl'):
    z = zipfile.ZipFile(f)
    for n in z.namelist():
        if is_private(n):
            print(f'LEAK in {f}: {n}')
            sys.exit(1)
print('Package audit: clean')
"

if [[ $ERRORS -gt 0 ]]; then
    echo "=== FAILED: $ERRORS leak(s) detected ==="
    exit 1
fi

echo "=== PASSED: no leaks detected ==="
