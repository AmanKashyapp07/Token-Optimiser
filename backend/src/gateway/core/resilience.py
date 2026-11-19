"""Enterprise resilience patterns: Circuit Breaker, Exponential Backoff, and Deadline Enforcement.

Guards the Token Optimiser gateway against cascading failures, upstream timeouts,
and repeated 5xx errors from frontier LLM providers. Automatically degrades traffic
to a local economy fallback tier when the primary model encounters repeated failures.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Coroutine, TypeVar

from gateway.core.exceptions import CircuitBreakerOpenError

logger = logging.getLogger("gateway.resilience")

T = TypeVar("T")

__all__ = [
    "TimeoutContext",
    "async_retry_with_backoff",
    "CircuitState",
    "CircuitBreakerOpenError",
    "CircuitBreaker",
    "get_circuit_breaker",
]


class TimeoutContext:
    """Manages and enforces deadlines across multi-stage pipeline executions.

    Attributes:
        timeout: Total permitted time budget in seconds.
        start_time: High-precision monotonic start timestamp.
        deadline: Monotonic deadline timestamp.
    """

    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout = timeout_seconds
        self.start_time = time.perf_counter()
        self.deadline = self.start_time + timeout_seconds

    def elapsed_ms(self) -> float:
        """Returns elapsed duration in milliseconds since creation."""
        return (time.perf_counter() - self.start_time) * 1000

    def remaining_seconds(self) -> float:
        """Returns remaining seconds before the SLA deadline expires.

        Raises:
            TimeoutError: If the deadline has already passed.
        """
        rem = self.deadline - time.perf_counter()
        if rem <= 0:
            raise TimeoutError(f"Request exceeded maximum SLA timeout of {self.timeout}s")
        return rem


async def async_retry_with_backoff(
    coro_fn: Callable[[], Coroutine[Any, Any, T]],
    retries: int = 2,
    base_delay: float = 0.1,
    max_delay: float = 1.0,
    jitter: bool = True,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Executes an async callable with exponential backoff and randomized decorrelated jitter.

    Args:
        coro_fn: Async callable factory to execute.
        retries: Maximum number of retry attempts.
        base_delay: Initial sleep duration in seconds before first retry.
        max_delay: Upper ceiling on backoff delay.
        jitter: Whether to add random noise (0.5 to 1.0 multiplier) to avoid synchronized stampedes.
        retry_exceptions: Tuple of exception types that trigger a retry.

    Returns:
        Result T returned by successful execution of coro_fn.

    Raises:
        The last exception encountered if all retries are exhausted.
    """
    last_err: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            return await coro_fn()
        except retry_exceptions as exc:
            last_err = exc
            if attempt == retries:
                break
            delay = min(max_delay, base_delay * (2**attempt))
            if jitter:
                delay = delay * (0.5 + random.random() * 0.5)
            logger.warning(
                "Retry attempt %d/%d after error: %s (sleeping %.3fs)", attempt + 1, retries, exc, delay
            )
            await asyncio.sleep(delay)

    assert last_err is not None
    raise last_err


class CircuitState(str, Enum):
    """Operational states for the Circuit Breaker finite state machine."""

    CLOSED = "CLOSED"  # Normal operation: requests pass through to primary provider
    OPEN = "OPEN"  # Tripped: requests immediately routed to fallback without calling upstream
    HALF_OPEN = "HALF_OPEN"  # Recovery testing: test single request to see if provider recovered


class CircuitBreaker:
    """Production Circuit Breaker pattern implementation.

    State Transition Invariants:
        [CLOSED]   --- (failures >= failure_threshold) ---> [OPEN]
        [OPEN]     --- (time >= recovery_timeout) --------> [HALF_OPEN]
        [HALF_OPEN]--- (success) -------------------------> [CLOSED]
        [HALF_OPEN]--- (failure) -------------------------> [OPEN]
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 5.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_state_change = time.time()
        self.last_failure_time: float | None = None

    def _update_state(self, new_state: CircuitState) -> None:
        """Transitions circuit to a new state and records state change timestamp."""
        if self.state != new_state:
            logger.info("CircuitBreaker state transitioned: %s -> %s", self.state.value, new_state.value)
            self.state = new_state
            self.last_state_change = time.time()

    def record_success(self) -> None:
        """Records a successful operation and recovers HALF_OPEN state to CLOSED."""
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self._update_state(CircuitState.CLOSED)

    def record_failure(self) -> None:
        """Records a failed operation and trips CLOSED circuit to OPEN if threshold reached."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self._update_state(CircuitState.OPEN)

    def execute(
        self,
        primary_fn: Callable[..., Any],
        fallback_fn: Callable[..., Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, bool, dict[str, Any]]:
        """Synchronously executes primary_fn if circuit permits; otherwise falls back.

        Args:
            primary_fn: Primary operation callable.
            fallback_fn: Optional fallback callable to invoke if circuit is OPEN or primary fails.
            *args: Positional arguments forwarded to callables.
            **kwargs: Keyword arguments forwarded to callables.

        Returns:
            Tuple of (result, fallback_applied: bool, telemetry_dict).

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN and no fallback_fn is supplied.
        """
        now = time.time()

        # Check if recovery timeout has elapsed to test HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (now - self.last_failure_time >= self.recovery_timeout):
                self._update_state(CircuitState.HALF_OPEN)
            else:
                if fallback_fn is None:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN ({self.failure_count} failures). Upstream provider unhealthy."
                    )
                fallback_res = fallback_fn(*args, **kwargs)
                return (
                    fallback_res,
                    True,
                    {
                        "circuit_state": CircuitState.OPEN.value,
                        "fallback_reason": "circuit_breaker_open_upstream_unhealthy",
                        "failures_logged": self.failure_count,
                    },
                )

        try:
            result = primary_fn(*args, **kwargs)
            self.record_success()
            return (
                result,
                False,
                {
                    "circuit_state": self.state.value,
                    "fallback_reason": None,
                    "failures_logged": 0,
                },
            )
        except Exception as exc:
            self.record_failure()
            if fallback_fn is None:
                raise
            fallback_res = fallback_fn(*args, **kwargs)
            return (
                fallback_res,
                True,
                {
                    "circuit_state": self.state.value,
                    "fallback_reason": f"primary_failed: {str(exc)}",
                    "failures_logged": self.failure_count,
                },
            )

    async def async_execute(
        self,
        primary_fn: Callable[..., Any],
        fallback_fn: Callable[..., Any] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, bool, dict[str, Any]]:
        """Asynchronously executes primary_fn guarding against failures.

        Args:
            primary_fn: Primary async callable.
            fallback_fn: Optional async or sync fallback callable.
            *args: Positional arguments forwarded to callables.
            **kwargs: Keyword arguments forwarded to callables.

        Returns:
            Tuple of (result, fallback_applied: bool, telemetry_dict).

        Raises:
            CircuitBreakerOpenError: If circuit is OPEN and no fallback_fn is supplied.
        """
        now = time.time()

        if self.state == CircuitState.OPEN:
            if self.last_failure_time and (now - self.last_failure_time >= self.recovery_timeout):
                self._update_state(CircuitState.HALF_OPEN)
            else:
                if fallback_fn is None:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker is OPEN ({self.failure_count} failures). Upstream provider unhealthy."
                    )
                res = fallback_fn(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    res = await res
                return (
                    res,
                    True,
                    {
                        "circuit_state": CircuitState.OPEN.value,
                        "fallback_reason": "circuit_breaker_open_upstream_unhealthy",
                        "failures_logged": self.failure_count,
                    },
                )

        try:
            res = primary_fn(*args, **kwargs)
            if asyncio.iscoroutine(res):
                res = await res
            self.record_success()
            return (
                res,
                False,
                {
                    "circuit_state": self.state.value,
                    "fallback_reason": None,
                    "failures_logged": 0,
                },
            )
        except Exception as exc:
            self.record_failure()
            if fallback_fn is None:
                raise
            res = fallback_fn(*args, **kwargs)
            if asyncio.iscoroutine(res):
                res = await res
            return (
                res,
                True,
                {
                    "circuit_state": self.state.value,
                    "fallback_reason": f"primary_failed: {str(exc)}",
                    "failures_logged": self.failure_count,
                },
            )

    def reset(self) -> None:
        """Resets circuit breaker to pristine CLOSED state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.last_state_change = time.time()


_GLOBAL_CIRCUIT_BREAKER = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)


def get_circuit_breaker() -> CircuitBreaker:
    """Returns the process-wide default CircuitBreaker singleton."""
    return _GLOBAL_CIRCUIT_BREAKER
