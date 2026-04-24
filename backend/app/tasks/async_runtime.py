"""Shared async runtime for sync Celery tasks.

Celery prefork workers execute synchronous task functions. Multiple modules may
need to run async coroutines against the same asyncpg-backed SQLAlchemy engine.
Using different event loops across modules causes cross-loop Future errors.
"""

from __future__ import annotations

import asyncio


_WORKER_LOOP: asyncio.AbstractEventLoop | None = None


def run_async(coro):
    """Run a coroutine on a single process-global event loop."""
    global _WORKER_LOOP
    if _WORKER_LOOP is None or _WORKER_LOOP.is_closed():
        _WORKER_LOOP = asyncio.new_event_loop()
    return _WORKER_LOOP.run_until_complete(coro)
