# Contributing to ClawLoop

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/aganthos/clawloop.git
cd clawloop
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -x
```

## Architecture Overview

ClawLoop has three learning layers that all follow the same protocol:

```
clawloop/
  core/         # Data structures, protocols, learning loop
  layers/       # Harness (prompt), Router (model selection), Weights (RL/finetune)
  adapters/     # Benchmark connectors (entropic, car, openclaw)
  evolvers/     # Evolution backends (LocalEvolver)
  envs/         # Task environments (math, harbor)
  extractors/   # Reward signal extraction
  backends/     # SkyRL GRPO integration
  exporters/    # OTel, SkyRL, router export
  callbacks/    # litellm integration
  utils/        # Async bridge
```

**Key types:** `Episode`, `EpisodeSummary`, `Datum`, `AgentState`, `StateID`

**Layer Protocol:** Every layer implements `forward_backward()` (accumulate
updates without mutation) and `optim_step()` (apply atomically, rollback on
failure). See `clawloop/core/layer.py`.

**Learning loop:** `clawloop/core/loop.py` — collects episodes, distributes
them as `Datum` objects, runs forward_backward then optim_step on each layer.

## Adding a New Environment

1. Create an adapter in `clawloop/adapters/` implementing `EnvAdapter`
2. Your `run_episode()` must return an `Episode` with messages, steps, and
   an `EpisodeSummary` containing reward signals
3. Register it in `clawloop/train.py` via `ENV_BUILDERS`
4. See `clawloop/envs/math.py` for a minimal example (~80 lines)

## Testing

```bash
# Run all tests
pytest tests/ -x

# Run a specific test file
pytest tests/test_agent.py -x

# Run a specific test
pytest tests/test_agent.py::TestClawLoopAgent::test_learn_basic -x

# Run with verbose output
pytest tests/ -x -v --timeout=30
```

Tests use `MockLLMClient` from `clawloop/llm.py` — no API keys needed. The
`tests/conftest.py` has a boundary guard that prevents tests from importing
private modules.

## Code Style

- Follow existing code patterns
- Use type hints on all public functions and methods
- Add docstrings to public classes and functions
- Use `from __future__ import annotations` for forward references
- Use `Protocol` for interfaces, `@dataclass` for value types
- No linter is enforced yet — just keep it consistent with surrounding code

## Commits

One commit per logical change with a prefix:

- `feat:` new functionality
- `fix:` bug fix
- `chore:` maintenance, docs, CI

## Pull Requests

- Run `pytest tests/ -x` before submitting
- Keep PRs focused — one concern per PR
- Describe what changed and why in the PR description

## Issues

- **Bug reports:** include steps to reproduce, expected vs actual behavior,
  and your Python version
- **Feature requests:** describe the use case, not just the solution

## License

By contributing, you agree that your contributions will be licensed under
the [BSL 1.1](LICENSE) license.
