"""Request coalescing (Singleflight pattern) engine for Token Optimiser.

Prevents cache stampedes and eliminates redundant upstream model inference by merging
concurrent in-flight executions for identical cache keys into a single leader task.
Followers await the leader's completion without hanging or blocking.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar

from gateway.core.exceptions import CoalescingError, LeaderExecutionError

logger = logging.getLogger("gateway.coalescing")

T = TypeVar("T")

__all__ = [
    "CoalescingError",
    "LeaderExecutionError",
    "RequestCoalescer",
    "get_request_coalescer",
]


class RequestCoalescer:
    """Singleflight pattern for cache lookups and expensive provider inference.

    If multiple concurrent requests arrive with the exact same key simultaneously,
    only the first request (the leader) triggers execution. All other requests (followers)
    await the in-flight future and share the identical result, eliminating cache
    stampedes and redundant upstream model inference.

    If the leader encounters an exception, that exception is cleanly propagated to all
    awaiting followers wrapped in a `LeaderExecutionError` without hanging or deadlock.
    Follower cancellations are shielded so they do not abort the leader's execution.
    """

    def __init__(self) -> None:
        self._in_flight: dict[str, asyncio.Future[Any]] = {}
        self._lock = asyncio.Lock()
        self._total_requests: int = 0
        self._coalesced_requests: int = 0
        self._leader_executions: int = 0
        self._leader_failures: int = 0

    @property
    def active_in_flight(self) -> int:
        """Returns the number of currently executing leader keys."""
        return len(self._in_flight)

    def get_metrics(self) -> dict[str, Any]:
        """Returns runtime singleflight deduplication telemetry and hit ratios."""
        hit_ratio = (self._coalesced_requests / self._total_requests) if self._total_requests > 0 else 0.0
        return {
            "total_requests": self._total_requests,
            "coalesced_requests": self._coalesced_requests,
            "leader_executions": self._leader_executions,
            "leader_failures": self._leader_failures,
            "active_in_flight": len(self._in_flight),
            "coalesce_hit_ratio": round(hit_ratio, 4),
        }

    def reset_metrics(self) -> None:
        """Resets telemetry counters."""
        self._total_requests = 0
        self._coalesced_requests = 0
        self._leader_executions = 0
        self._leader_failures = 0

    def reset(self) -> None:
        """Resets both in-flight tracking and telemetry."""
        self._in_flight.clear()
        self.reset_metrics()

    async def do(
        self,
        key: str,
        coro_fn: Callable[[], Coroutine[Any, Any, T]],
    ) -> tuple[T, bool]:
        """Executes coro_fn if key is not already active; otherwise waits for active future.

        Args:
            key: Deterministic cache or execution key used for deduplication.
            coro_fn: Async zero-argument factory callable for the primary task.

        Returns:
            Tuple of (result: T, was_coalesced: bool).

        Raises:
            LeaderExecutionError: If a coalesced follower was awaiting a leader that failed.
            BaseException: If the caller was the leader and coro_fn raised.
        """
        async with self._lock:
            self._total_requests += 1
            if key in self._in_flight:
                future = self._in_flight[key]
                is_leader = False
                self._coalesced_requests += 1
            else:
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._in_flight[key] = future
                is_leader = True
                self._leader_executions += 1

        if not is_leader:
            try:
                # Shield prevents follower cancellation from cancelling the leader's future
                result = await asyncio.shield(future)
                return result, True
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                raise LeaderExecutionError(key=key, original_error=exc) from exc

        try:
            result = await coro_fn()
            if not future.done():
                future.set_result(result)
            return result, False
        except BaseException as exc:
            self._leader_failures += 1
            if not future.done():
                future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)


_GLOBAL_COALESCER = RequestCoalescer()


def get_request_coalescer() -> RequestCoalescer:
    """Returns the process-wide RequestCoalescer singleton instance."""
    return _GLOBAL_COALESCER
