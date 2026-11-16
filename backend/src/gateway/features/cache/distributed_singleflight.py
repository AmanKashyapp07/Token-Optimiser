from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, Callable, Coroutine

from gateway.features.cache.coalescing import (
    LeaderExecutionError,
    RequestCoalescer,
    get_request_coalescer,
)

logger = logging.getLogger(__name__)


class MockRedisPubSub:
    """In-memory async PubSub channel for local simulation of Redis distributed messaging."""

    def __init__(self, hub: MockRedisClient, channel: str) -> None:
        self.hub = hub
        self.channel = channel
        self.queue: asyncio.Queue[str] = asyncio.Queue()

    async def get_message(self, timeout: float = 10.0) -> dict[str, Any] | None:
        try:
            msg = await asyncio.wait_for(self.queue.get(), timeout=timeout)
            return {"type": "message", "channel": self.channel, "data": msg}
        except asyncio.TimeoutError:
            return None

    async def close(self) -> None:
        self.hub.unsubscribe(self.channel, self)


class MockRedisClient:
    """In-memory Redis client mock for testing distributed locks and pub/sub messaging."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.locks: dict[str, str] = {}
        self.subscribers: dict[str, list[MockRedisPubSub]] = {}
        self._lock = asyncio.Lock()

    async def set(
        self,
        key: str,
        value: str,
        nx: bool = False,
        px: int | None = None,
        ex: int | None = None,
    ) -> bool:
        async with self._lock:
            if nx and (key in self.locks or key in self.store):
                return False
            if nx:
                self.locks[key] = value
            else:
                self.store[key] = value
            return True

    async def get(self, key: str) -> str | None:
        async with self._lock:
            return self.store.get(key)

    async def delete(self, key: str) -> int:
        async with self._lock:
            removed = 0
            if key in self.locks:
                del self.locks[key]
                removed += 1
            if key in self.store:
                del self.store[key]
                removed += 1
            return removed

    async def publish(self, channel: str, message: str) -> int:
        async with self._lock:
            subs = list(self.subscribers.get(channel, []))
        for sub in subs:
            await sub.queue.put(message)
        return len(subs)

    def pubsub(self, channel: str) -> MockRedisPubSub:
        sub = MockRedisPubSub(self, channel)
        if channel not in self.subscribers:
            self.subscribers[channel] = []
        self.subscribers[channel].append(sub)
        return sub

    def unsubscribe(self, channel: str, sub: MockRedisPubSub) -> None:
        if channel in self.subscribers:
            self.subscribers[channel] = [s for s in self.subscribers[channel] if s != sub]


class DistributedRequestCoalescer:
    """Production Redis-backed singleflight request coalescer across multi-worker clusters.

    When multiple gateway instances (processes/containers) receive identical requests:
    1. Workers attempt to acquire a distributed mutex via Redis `SET lock:{key} {worker_id} NX PX 30000`.
    2. The acquiring node becomes the **Distributed Leader** and executes the expensive task.
    3. Other nodes become **Distributed Followers** and subscribe to Redis Pub/Sub `coalesce:channel:{key}`.
    4. Upon completion, the leader publishes the result and unlocks; followers receive the broadcast.
    5. If the leader fails, followers receive structured `LeaderExecutionError` without hanging.
    6. If Redis is unconfigured or offline, seamlessly falls back to the in-memory `RequestCoalescer`.
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        worker_id: str | None = None,
        lock_ttl_ms: int = 30000,
        follower_timeout_s: float = 10.0,
        local_fallback: RequestCoalescer | None = None,
    ) -> None:
        self.redis = redis_client
        self.worker_id = worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.lock_ttl_ms = lock_ttl_ms
        self.follower_timeout_s = follower_timeout_s
        self.local_coalescer = local_fallback or get_request_coalescer()

    async def do(
        self,
        key: str,
        coro_fn: Callable[[], Coroutine[Any, Any, Any]],
    ) -> tuple[Any, bool]:
        """Executes coro_fn with cluster-wide singleflight deduplication.

        Returns:
            (result, was_coalesced: bool)
        """
        if not self.redis:
            # Redis is not configured: fallback to in-process singleflight
            return await self.local_coalescer.do(key, coro_fn)

        lock_key = f"coalesce:lock:{key}"
        channel_key = f"coalesce:channel:{key}"
        result_key = f"coalesce:result:{key}"

        # 1. Attempt distributed leader election
        is_leader = await self.redis.set(lock_key, self.worker_id, nx=True, px=self.lock_ttl_ms)

        if not is_leader:
            # 2. Follower path: subscribe to pub/sub channel and await leader broadcast
            pubsub = self.redis.pubsub(channel_key)
            try:
                msg = await pubsub.get_message(timeout=self.follower_timeout_s)
                if not msg:
                    raise TimeoutError(f"Distributed follower timed out awaiting leader for key '{key}'")

                payload = json.loads(msg["data"])
                if payload.get("status") == "error":
                    orig_err = RuntimeError(payload.get("error_message", "Leader execution failed"))
                    raise LeaderExecutionError(key=key, original_error=orig_err)

                # Fetch cached result
                raw_result = await self.redis.get(result_key)
                if raw_result is None:
                    # Fallback to local execution if result expired
                    return await coro_fn(), True

                result = json.loads(raw_result)
                return result, True
            finally:
                await pubsub.close()

        # 3. Leader path: execute coro_fn, broadcast result, and release lock
        try:
            result = await coro_fn()
            # Store result in Redis with 60s TTL
            try:
                serialized = json.dumps(result) if not isinstance(result, str) else result
                await self.redis.set(result_key, serialized, ex=60)
                await self.redis.publish(
                    channel_key,
                    json.dumps({"status": "success", "worker_id": self.worker_id}),
                )
            except Exception as e:
                logger.warning("Failed to publish distributed coalescer result: %s", e)

            return result, False
        except BaseException as exc:
            # Broadcast failure to all awaiting cluster followers
            try:
                await self.redis.publish(
                    channel_key,
                    json.dumps({
                        "status": "error",
                        "error_message": str(exc),
                        "worker_id": self.worker_id,
                    }),
                )
            except Exception as e:
                logger.warning("Failed to broadcast leader error: %s", e)
            raise
        finally:
            try:
                await self.redis.delete(lock_key)
            except Exception as e:
                logger.warning("Failed to release distributed lock: %s", e)
