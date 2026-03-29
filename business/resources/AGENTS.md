# Resources Folder

When working on slides, presentations, or pitch decks in this folder, use the `/frontend-slides` skill for creating or converting HTML presentations. This skill provides curated design presets, PowerPoint import, and animation-rich single-file HTML output.

## Brand Guidelines
- Aganthos brand uses a dark red/black gradient aesthetic
- Logo SVG available at `../landingpage/logo.svg` (within business/)
- Website: aganthos.com
- Tagline: "Learning from Experience Lab"

## Public/Private Boundary

This monorepo contains both public (community) and private (enterprise) code.

### Directory classification
- `clawloop/`, `tests/`, `examples/` → **PUBLIC** (synced to github.com/aganthos/clawloop)
- `enterprise_clawloop/` → **PRIVATE** (proprietary code, CLIProxyAPI demos, private configs)
- `business/` → **PRIVATE** (sales, marketing, GTM, docs, plans, resources)
- `scripts/` → **PRIVATE** (dev/ops tooling)
- `benchmarks/` → **PRIVATE** (git submodules)

### Rules for coding agents
1. Code in `clawloop/` must NEVER import from `enterprise_clawloop/`
2. Code in `enterprise_clawloop/` CAN import from `clawloop` (it extends community)
3. Tests in `tests/` must NEVER import from `enterprise_clawloop/`
4. Enterprise tests live in `enterprise_clawloop/tests/`
5. New files in `clawloop/` automatically become public
6. New files in `enterprise_clawloop/` stay private
7. The `.publicpaths` manifest is the source of truth for what gets synced

### Architecture (future — interfaces defined when first backend exists)
- `clawloop.core.layer.Layer` Protocol = what executes (community)
- Enterprise will use an Evolver Protocol = what proposes improvements to what executes
- Planned evolution backends: reflector → guided_mutation → dgm_h (hyperagent)
- Cloud hook: `clawloop.wrap(client, cloud_url=..., cloud_api_key=...)` sends traces, pulls patches
- See arxiv.org/abs/2603.19461 (HyperAgents) for the long-term hyperagent vision
