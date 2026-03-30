# Third-Party Notices

ClawLoop references the following third-party components as git submodules.
These are **not bundled** in the ClawLoop package or source distribution —
they are fetched from their upstream repositories when you run
`git submodule update --init`. Each component is licensed under its original
license by its respective authors.

## SkyRL (`clawloop/skyrl/`)

- **Source:** https://github.com/NovaSky-AI/SkyRL
- **License:** Apache License 2.0
- **Used for:** Weight training (LoRA, full fine-tuning, GRPO, PPO, SFT)
- **Note:** SkyRL itself includes adapted code from several projects
  (VERL, NeMo-Aligner, OpenRLHF, Unsloth Zoo, and others). See individual
  file headers within the submodule for specific attributions and licenses.

## Benchmarks (`benchmarks/`)

Optional benchmark environments, used for evaluation only.

| Submodule | Source | License |
|---|---|---|
| Harbor | https://github.com/harbor-framework/harbor | See upstream |
| entropic-crmarenapro | https://github.com/rkstu/entropic-crmarenapro | MIT (per README) |
| CAR-bench | https://github.com/CAR-bench/car-bench-agentbeats | See upstream |

## pip Dependencies

ClawLoop's pip dependencies (litellm, pydantic, etc.) are installed
separately via `pip install` and are not included in this source tree.
See `pyproject.toml` for the full dependency list.
