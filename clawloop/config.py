"""Environment config loading for ClawLoop.

Loads `.env` so users don't have to source/export by hand. Idempotent; safe
to call multiple times. Lookup order (first hit wins, later files do not
override earlier ones):

1. ``$CLAWLOOP_ENV_FILE`` — explicit path override.
2. ``clawloop/.env`` — package-scoped, colocated with code.
3. Nearest ``.env`` walking up from the current working directory.

Missing files are skipped silently. Existing environment variables are never
overridden (so CI/CD injected secrets always win over local ``.env``).
"""
from __future__ import annotations

import os
from pathlib import Path

_loaded: bool = False


def load_env(*, force: bool = False) -> list[Path]:
    """Load .env files. Returns the list of paths actually loaded.

    Args:
        force: If True, reloads even if load_env has already been called in
               this process. Default False for cheap idempotency.
    """
    global _loaded
    if _loaded and not force:
        return []

    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        _loaded = True
        return []

    loaded: list[Path] = []

    override = os.environ.get("CLAWLOOP_ENV_FILE")
    if override:
        p = Path(override)
        if p.is_file():
            load_dotenv(p, override=False)
            loaded.append(p)

    pkg_env = Path(__file__).resolve().parent.parent / ".env"
    if pkg_env.is_file():
        load_dotenv(pkg_env, override=False)
        loaded.append(pkg_env)

    nearest = find_dotenv(usecwd=True)
    if nearest and Path(nearest).resolve() not in {p.resolve() for p in loaded}:
        load_dotenv(nearest, override=False)
        loaded.append(Path(nearest))

    _loaded = True
    return loaded
