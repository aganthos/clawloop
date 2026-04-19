"""Safe async-to-sync bridge."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

_EXECUTOR = ThreadPoolExecutor(max_workers=1)


def run_async(coro):
    """Run an async coroutine from sync code safely.

    No event loop running: uses asyncio.run() (fast path).
    Event loop running (Jupyter, async orchestration): runs in a
    thread with its own event loop.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        future = _EXECUTOR.submit(asyncio.run, coro)
        return future.result()
