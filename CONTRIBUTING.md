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
