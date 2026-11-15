from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable


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
