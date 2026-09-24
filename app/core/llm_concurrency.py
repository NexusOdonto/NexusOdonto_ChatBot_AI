"""Global semaphore for LLM/embedding calls to bound Gemini concurrency under load.

Also tracks an extendable graph deadline so time spent waiting for an LLM slot
does not burn the graph timeout budget (cite/agenda turns often need several
LLM→tools loops under concurrent load).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextvars import ContextVar
from typing import Awaitable, Dict, Optional, TypeVar

from app.core.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")

_semaphores: Dict[int, asyncio.Semaphore] = {}
_semaphores_guard = threading.Lock()

# Absolute monotonic deadline for the current graph invoke (None = no deadline).
_graph_deadline_mono: ContextVar[Optional[float]] = ContextVar(
    "graph_deadline_mono", default=None
)
_graph_queue_wait_total: ContextVar[float] = ContextVar(
    "graph_queue_wait_total", default=0.0
)
# Snapshot after run_with_graph_deadline finishes (survives clear_graph_deadline).
_last_graph_queue_wait: ContextVar[float] = ContextVar(
    "last_graph_queue_wait", default=0.0
)


class GraphDeadlineExceeded(TimeoutError):
    """Raised when the extendable graph work budget is exhausted (not LLM queue wait)."""


class LLMCallTimeoutError(TimeoutError):
    """Raised when a single LLM/embed call exceeds llm_call_timeout_seconds after acquiring a slot."""


def _llm_semaphore() -> asyncio.Semaphore:
    """Return an asyncio.Semaphore bound to the current running event loop."""
    loop = asyncio.get_running_loop()
    key = id(loop)
    with _semaphores_guard:
        sem = _semaphores.get(key)
        if sem is None:
            limit = max(1, int(getattr(settings, "llm_max_concurrent", 8) or 8))
            sem = asyncio.Semaphore(limit)
            _semaphores[key] = sem
            logger.info(f"[LLM] concurrency semaphore bound limit={limit}")
        return sem


def begin_graph_deadline(timeout_seconds: float) -> None:
    """Start a wall-clock budget for graph work (excludes later queue waits)."""
    seconds = max(1.0, float(timeout_seconds))
    _graph_deadline_mono.set(time.monotonic() + seconds)
    _graph_queue_wait_total.set(0.0)
    _last_graph_queue_wait.set(0.0)


def clear_graph_deadline() -> None:
    _graph_deadline_mono.set(None)
    _graph_queue_wait_total.set(0.0)


def extend_graph_deadline(seconds: float) -> None:
    """Push the deadline forward (used to exclude LLM queue wait from budget)."""
    if seconds <= 0:
        return
    deadline = _graph_deadline_mono.get()
    if deadline is None:
        return
    _graph_deadline_mono.set(deadline + seconds)
    _graph_queue_wait_total.set(_graph_queue_wait_total.get() + seconds)


def graph_deadline_remaining() -> Optional[float]:
    deadline = _graph_deadline_mono.get()
    if deadline is None:
        return None
    return deadline - time.monotonic()


def graph_queue_wait_total() -> float:
    """Queue wait accumulated during the active (or just-finished) graph run."""
    active = float(_graph_queue_wait_total.get() or 0.0)
    if active > 0:
        return active
    return float(_last_graph_queue_wait.get() or 0.0)


async def with_llm_slot(awaitable: Awaitable[T], *, label: str = "llm") -> T:
    """Acquire an LLM/embed slot, log queue_wait when contention is high, then run awaitable.

    Time spent waiting for the semaphore is added back to the active graph deadline
    so concurrent queueing does not trigger MENSAJE_GRAPH_TIMEOUT prematurely.
    """
    sem = _llm_semaphore()
    t0 = time.perf_counter()
    await sem.acquire()
    waited = time.perf_counter() - t0
    if waited > 0:
        extend_graph_deadline(waited)
    threshold = float(getattr(settings, "llm_queue_wait_log_seconds", 2.0) or 2.0)
    if waited >= threshold:
        logger.info(
            f"[Latency] queue_wait={waited:.3f}s label={label} "
            f"max_concurrent={getattr(settings, 'llm_max_concurrent', 8)} "
            f"queue_wait_total={float(_graph_queue_wait_total.get() or 0.0):.3f}s"
        )
    try:
        call_timeout = float(getattr(settings, "llm_call_timeout_seconds", 0) or 0)
        if call_timeout > 0:
            try:
                return await asyncio.wait_for(awaitable, timeout=call_timeout)
            except asyncio.TimeoutError as exc:
                logger.warning(
                    f"[Latency] llm_call_timeout={call_timeout:.1f}s label={label}"
                )
                raise LLMCallTimeoutError(
                    f"LLM call timed out after {call_timeout:.0f}s ({label})"
                ) from exc
        return await awaitable
    finally:
        sem.release()


async def run_with_graph_deadline(coro: Awaitable[T], *, timeout_seconds: float) -> T:
    """Run coro until done or the extendable graph deadline expires.

    Unlike asyncio.wait_for, this re-checks a ContextVar deadline that
    with_llm_slot extends while waiting for a concurrency slot.
    Raises GraphDeadlineExceeded (not bare TimeoutError) so callers can
    distinguish from per-call LLM timeouts.
    """
    begin_graph_deadline(timeout_seconds)
    task = asyncio.create_task(coro)  # type: ignore[arg-type]
    try:
        while True:
            remaining = graph_deadline_remaining()
            if remaining is not None and remaining <= 0:
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
                raise GraphDeadlineExceeded(
                    f"Graph deadline exceeded ({timeout_seconds:.0f}s work budget)"
                )

            wait_slice = 0.5 if remaining is None else min(max(remaining, 0.05), 0.5)
            done, _ = await asyncio.wait({task}, timeout=wait_slice)
            if task in done:
                return task.result()
    finally:
        _last_graph_queue_wait.set(float(_graph_queue_wait_total.get() or 0.0))
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        clear_graph_deadline()
