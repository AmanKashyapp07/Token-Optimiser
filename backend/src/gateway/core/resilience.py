from __future__ import annotations

import asyncio
import random
import time
from enum import Enum
from typing import Any, Callable


class TimeoutContext:
    """Manages and enforces deadlines across multi-stage pipeline executions."""

    def __init__(self, timeout_seconds: float = 10.0):
        self.timeout = timeout_seconds
        self.start_time = time.perf_counter()
        self.deadline = self.start_time + timeout_seconds

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000

    def remaining_seconds(self) -> float:
        rem = self.deadline - time.perf_counter()
        if rem <= 0:
            raise TimeoutError(f"Request exceeded maximum SLA timeout of {self.timeout}s")
        return rem


async def async_retry_with_backoff(
    coro_fn: Callable[..., Any],
    retries: int = 2,
    base_delay: float = 0.1,
    max_delay: float = 1.0,
    jitter: bool = True,
    retry_exceptions: tuple = (Exception,),
) -> Any:
    """Executes an async callable with exponential backoff and randomized jitter."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return await coro_fn()
        except retry_exceptions as exc:
            last_err = exc
            if attempt == retries:
                break
            delay = min(max_delay, base_delay * (2 ** attempt))
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            await asyncio.sleep(delay)
    raise last_err


class CircuitState(str, Enum):
    CLOSED = "CLOSED"        # Normal operation: requests pass through to primary provider
    OPEN = "OPEN"            # Tripped: requests immediately routed to fallback without calling upstream
    HALF_OPEN = "HALF_OPEN"  # Recovery testing: test single request to see if provider recovered


class CircuitBreaker:
    """Production Circuit Breaker pattern implementation.

    Guards against cascading failures, upstream timeouts, and provider 503 errors.
    Automatically degrades traffic to a local economy fallback tier when the primary
    frontier model encounters repeated failures.
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 5.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()
        self.last_failure_time: float | None = None

    def _update_state(self, new_state: CircuitState) -> None:
        if self.state != new_state:
            self.state = new_state
            self.last_state_change = time.time()

    def record_success(self) -> None:
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self._update_state(CircuitState.CLOSED)

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self._update_state(CircuitState.OPEN)

    def execute(
        self,
        primary_fn: Callable[..., Any],
        fallback_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, bool, dict[str, Any]]:
        """Executes primary_fn if circuit allows, otherwise gracefully falls back to fallback_fn.

        Returns:
            (result, fallback_applied, telemetry_dict)
        """
        now = time.time()

        # Check if recovery timeout has elapsed to test HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (now - self.last_failure_time >= self.recovery_timeout):
                self._update_state(CircuitState.HALF_OPEN)
            else:
                fallback_res = fallback_fn(*args, **kwargs)
                return fallback_res, True, {
                    "circuit_state": CircuitState.OPEN.value,
                    "fallback_reason": "circuit_breaker_open_upstream_unhealthy",
                    "failures_logged": self.failure_count,
                }

        try:
            result = primary_fn(*args, **kwargs)
            self.record_success()
            return result, False, {
                "circuit_state": self.state.value,
                "fallback_reason": None,
                "failures_logged": 0,
            }
        except Exception as exc:
            self.record_failure()
            fallback_res = fallback_fn(*args, **kwargs)
            return fallback_res, True, {
                "circuit_state": self.state.value,
                "fallback_reason": f"primary_failed: {str(exc)}",
                "failures_logged": self.failure_count,
            }

    def reset(self) -> None:
        """Reset circuit breaker to pristine CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()


_GLOBAL_CIRCUIT_BREAKER = CircuitBreaker()


def get_circuit_breaker() -> CircuitBreaker:
    return _GLOBAL_CIRCUIT_BREAKER
