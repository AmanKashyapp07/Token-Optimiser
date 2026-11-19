"""Centralized exception hierarchy for the Token Optimiser gateway.

Follows the OpenAI API error response specifications and standard HTTP semantics.
Provides structured error dictionaries with error type, code, and parameter tracking.
"""

from __future__ import annotations

from typing import Any


class GatewayError(Exception):
    """Base exception for all Token Optimiser gateway operations.

    Attributes:
        message: Human-readable error description.
        status_code: Corresponding HTTP status code.
        error_type: Machine-readable error category string (e.g., 'gateway_error').
        code: Specific domain error code or string representation.
        param: The request parameter that triggered the error, if applicable.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "gateway_error",
        code: str | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.code = code or str(status_code)
        self.param = param

    def to_dict(self) -> dict[str, Any]:
        """Serializes exception to OpenAI-compatible error format with legacy FastAPI detail."""
        return {
            "detail": self.message,
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.code,
                "param": self.param,
            },
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, status_code={self.status_code})"


# ==============================================================================
# Upstream Provider & Inference Exceptions
# ==============================================================================


class ProviderError(GatewayError):
    """Raised when an upstream LLM provider fails or returns an error."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        code: str = "provider_error",
        param: str | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_type="provider_error",
            code=code,
            param=param,
        )


class ProviderTimeoutError(ProviderError):
    """Raised when the upstream provider fails to respond within SLA deadline."""

    def __init__(self, message: str = "Upstream inference provider timed out") -> None:
        super().__init__(message=message, status_code=504, code="gateway_timeout")


class ProviderRateLimitError(ProviderError):
    """Raised when the upstream provider rejects a call due to rate limits (HTTP 429)."""

    def __init__(self, message: str = "Upstream inference provider rate limit exceeded") -> None:
        super().__init__(message=message, status_code=429, code="provider_rate_limited")


class ProviderAuthenticationError(ProviderError):
    """Raised when upstream credentials (API key) are rejected (HTTP 401/403)."""

    def __init__(self, message: str = "Upstream provider authentication failed") -> None:
        super().__init__(message=message, status_code=502, code="bad_gateway_auth")


class ProviderUnavailableError(ProviderError):
    """Raised when the upstream provider experiences service outage (HTTP 500/503)."""

    def __init__(self, message: str = "Upstream inference provider service unavailable") -> None:
        super().__init__(message=message, status_code=503, code="service_unavailable")


# ==============================================================================
# Concurrency & Resilience Exceptions
# ==============================================================================


class CircuitBreakerOpenError(GatewayError, RuntimeError):
    """Raised when circuit breaker is in OPEN state and fast-fails traffic.

    Inherits from RuntimeError to maintain complete backward compatibility with
    legacy test assertions.
    """

    def __init__(
        self,
        message: str = "Circuit breaker is OPEN: upstream provider is unavailable",
    ) -> None:
        GatewayError.__init__(
            self,
            message=message,
            status_code=503,
            error_type="circuit_breaker_open",
            code="service_unavailable",
        )


class CoalescingError(GatewayError):
    """Base error for request coalescing (singleflight) failures."""

    def __init__(self, message: str, status_code: int = 502, code: str = "coalescing_error") -> None:
        super().__init__(message=message, status_code=status_code, error_type="coalescing_error", code=code)


class LeaderExecutionError(CoalescingError):
    """Raised when a follower request's coalesced leader encounter an unhandled exception."""

    def __init__(self, key: str, original_error: BaseException) -> None:
        self.key = key
        self.original_error = original_error
        self.leader_failed = True
        msg = f"Singleflight leader execution failed for key '{key}': {original_error}"
        super().__init__(message=msg, status_code=502, code="leader_execution_error")


# ==============================================================================
# Feature-Specific Domain Exceptions
# ==============================================================================


class RateLimitExceededError(GatewayError):
    """Raised when client exceeds maximum allowed requests per minute."""

    def __init__(
        self,
        message: str = "Rate limit exceeded. Too many requests.",
        retry_after: int = 60,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(
            message=message,
            status_code=429,
            error_type="rate_limit_exceeded",
            code="rate_limit_exceeded",
        )


class CacheError(GatewayError):
    """Raised on cache storage or serialization failures."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(
            message=message, status_code=status_code, error_type="cache_error", code="cache_error"
        )


class RoutingError(GatewayError):
    """Raised when model routing classification fails unexpectedly."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(
            message=message, status_code=status_code, error_type="routing_error", code="routing_error"
        )


class BudgetingError(GatewayError):
    """Raised when token budgeting model fails to produce valid allocations."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(
            message=message, status_code=status_code, error_type="budgeting_error", code="budgeting_error"
        )


class InvalidRequestError(GatewayError):
    """Raised when client request payload fails semantic domain validation."""

    def __init__(self, message: str, param: str | None = None) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_type="invalid_request_error",
            code="invalid_request",
            param=param,
        )
