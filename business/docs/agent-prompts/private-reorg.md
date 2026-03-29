# Task: Reorganize aganthos private directory structure

Working directory: /Users/robertmueller/Desktop/aganthos
Branch: create `chore/private-reorg` from current HEAD

## Goal

Consolidate scattered private directories into 3 clean groups:
- `enterprise_clawloop/` — proprietary code + CLIProxyAPI demos + private configs
- `scripts/` — dev/ops tooling only
- `business/` — all non-technical content (sales, marketing, GTM, docs, plans)

## Hard Rules

- NEVER modify .publicpaths
- NEVER modify files in clawloop/ EXCEPT for updating the seed-prompt
  default path in server.py (necessary to not break the server after
  moving config/)
- NEVER modify benchmarks/ (git submodules)
- Use `git mv` for all moves to preserve history
- After all moves, fix ALL broken references

## Step 1: Move CLIProxyAPI demos into enterprise_clawloop/examples/

```bash
mkdir -p enterprise_clawloop/examples
git mv n8n-workflows enterprise_clawloop/examples/n8n_cliproxy
git mv scripts/demo.py enterprise_clawloop/examples/n8n_cliproxy/demo.py
git mv scripts/test_openclaw_learning_loop.py enterprise_clawloop/examples/openclaw_cliproxy.py
```

## Step 2: Move private config into enterprise_clawloop/config/

```bash
mkdir -p enterprise_clawloop/config
git mv config/demo_tickets.json enterprise_clawloop/config/
git mv config/seed_prompt.txt enterprise_clawloop/config/
rm -rf config  # remove even if .DS_Store remains
```

Update the default path in clawloop/server.py:
```python
# Change this line:
parser.add_argument("--seed-prompt", default="config/seed_prompt.txt")
# To:
parser.add_argument("--seed-prompt", default="enterprise_clawloop/config/seed_prompt.txt")
```

## Step 3: Move private training configs

```bash
mkdir -p enterprise_clawloop/examples/configs
git mv configs/car_train.json enterprise_clawloop/examples/configs/
git mv configs/entropic_mini.json enterprise_clawloop/examples/configs/
git mv configs/entropic_smoke.json enterprise_clawloop/examples/configs/
git mv configs/entropic_train.json enterprise_clawloop/examples/configs/
rm -rf configs
```

## Step 4: Move experiment runs

```bash
git mv runs enterprise_clawloop/runs
```

## Step 5: Consolidate business directories

```bash
mkdir -p business
git mv pitch business/
git mv gtm business/
git mv webpage business/
git mv landingpage business/
git mv swiss_genai_price business/awards
git mv communication business/
git mv intake-forms business/
git mv resources business/
git mv docs business/
```

Move orphaned root business files:
```bash
git mv CONTACT_VERIFICATION_GUIDE.md business/
git mv PILOT_CUSTOMER_RESEARCH_SUMMARY.md business/
git mv aganthos_pilot_customers_master.xlsx business/
```

## Step 6: Update pyproject.toml exclude list

The hatch build exclude list references old directory names. Update:
```python
# FROM:
exclude = [
    "benchmarks/",
    "enterprise_clawloop/",
    "pitch/",
    "configs/",
    "resources/",
    "scripts/",
    "docs/",
]
# TO:
exclude = [
    "benchmarks/",
    "enterprise_clawloop/",
    "business/",
    "scripts/",
]
```

## Step 7: Update .gitignore

Add new directory patterns, remove old ones. Ensure these are ignored:
- `business/pitch/node_modules/`
- `enterprise_clawloop/runs/`
- `playbook.json` (test artifact)
- `dist/`

## Step 8: Fix ALL broken references

Run this comprehensive search:
```bash
rg -n "n8n-workflows|scripts/demo|config/seed_prompt|config/demo_tickets|configs/car|configs/entropic|swiss_genai_price|intake-forms|resources/|pitch/|gtm/|runs/car|runs/entropic|docs/plans|docs/specs|docs/research|docs/agent-prompts" \
  --glob '!node_modules' --glob '!.git' --glob '!.kd' --glob '!*.lock' .
```

For EVERY match:
- If the file was moved: update the path to the new location
- Key files that WILL have broken refs:
  - `enterprise_clawloop/examples/n8n_cliproxy/README.md` — paths to
    config/, scripts/demo.py
  - `enterprise_clawloop/examples/n8n_cliproxy/demo.py` — tickets path
  - `enterprise_clawloop/examples/openclaw_cliproxy.py` — any path refs
  - `docker-compose.yml` — check for any config/ or scripts/ refs
  - `business/resources/AGENTS.md` — documents private dir layout
  - `business/gtm/*.md` — may reference resources/, pitch/
  - `business/pitch/*.md` — may reference resources/
  - Any moved .md files referencing other moved .md files
  - `clawloop/guide/*.md` — may reference docs/

Do NOT update references inside .kd/ (kingdom design artifacts are snapshots).

## Step 9: Verify scripts/ is clean

After moves, scripts/ should contain exactly:
- sync_public.sh
- audit_public.sh
- analyze_run.py
- gpu_validation/

```bash
ls scripts/
```

## Step 10: Verification (run ALL, no skipping)

1. `git diff --stat HEAD` — review all changes
2. `PYTHONPATH=. python -c "import clawloop; import enterprise_clawloop"` — imports work
3. `PYTHONPATH=. python -m pytest tests/ -x -q --timeout=30` — public tests pass
4. `cat .publicpaths` — unchanged
5. No proprietary leakage in public paths:
   ```bash
   rg "8317|CLIProxy|cliproxy|kuhhandel" examples/ clawloop/ tests/ README.md \
     --glob '!node_modules' || echo "CLEAN"
   ```
6. Old dirs gone:
   ```bash
   for d in n8n-workflows config configs runs pitch gtm webpage landingpage \
     swiss_genai_price communication intake-forms resources docs; do
     [ -e "$d" ] && echo "FAIL: $d still exists at root" || echo "CLEAN: $d gone"
   done
   ```
7. No broken references remain:
   ```bash
   rg "n8n-workflows/|configs/car|configs/entropic|swiss_genai_price/" \
     --glob '!node_modules' --glob '!.git' --glob '!.kd' . \
     && echo "FAIL: stale refs found" || echo "CLEAN"
   ```

## Commit

```bash
git add -A
git commit -m "chore: consolidate private dirs into enterprise_clawloop/ and business/"
```
