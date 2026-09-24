"""Global semaphore for LLM/embedding calls to bound Gemini concurrency under load."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Awaitable, Dict, TypeVar

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_semaphores: Dict[int, asyncio.Semaphore] = {}
_semaphores_guard = threading.Lock()


def _llm_semaphore() -> asyncio.Semaphore:
    """Return an asyncio.Semaphore bound to the current running event loop."""
    loop = asyncio.get_running_loop()
    key = id(loop)
    with _semaphores_guard:
        sem = _semaphores.get(key)
        if sem is None:
            limit = max(1, int(getattr(settings, "llm_max_concurrent", 3) or 3))
            sem = asyncio.Semaphore(limit)
            _semaphores[key] = sem
        return sem


async def with_llm_slot(awaitable: Awaitable[T], *, label: str = "llm") -> T:
    """Acquire an LLM/embed slot, log queue_wait when contention is high, then run awaitable."""
    sem = _llm_semaphore()
    t0 = time.perf_counter()
    await sem.acquire()
    waited = time.perf_counter() - t0
    threshold = float(getattr(settings, "llm_queue_wait_log_seconds", 2.0) or 2.0)
    if waited >= threshold:
        logger.info(
            f"[Latency] queue_wait={waited:.3f}s label={label} "
            f"max_concurrent={getattr(settings, 'llm_max_concurrent', 3)}"
        )
    try:
        return await awaitable
    finally:
        sem.release()
