from __future__ import annotations

import asyncio
from typing import Any, Callable, Coroutine


class RequestCoalescer:
    """Singleflight pattern for cache lookups and expensive operations.

    If multiple concurrent requests arrive with the exact same key simultaneously,
    only the first request triggers the execution; all other requests await the
    in-flight future and share the identical result, eliminating cache stampedes.
    """

    def __init__(self) -> None:
        self._in_flight: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()

    async def do(
        self,
        key: str,
        coro_fn: Callable[[], Coroutine[Any, Any, Any]],
    ) -> tuple[Any, bool]:
        """Executes coro_fn if key is not already active; otherwise waits for active future.

        Returns:
            (result, was_coalesced: bool)
        """
        async with self._lock:
            if key in self._in_flight:
                future = self._in_flight[key]
                is_leader = False
            else:
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._in_flight[key] = future
                is_leader = True

        if not is_leader:
            result = await future
            return result, True

        try:
            result = await coro_fn()
            future.set_result(result)
            return result, False
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            async with self._lock:
                self._in_flight.pop(key, None)


_GLOBAL_COALESCER = RequestCoalescer()


def get_request_coalescer() -> RequestCoalescer:
    return _GLOBAL_COALESCER
