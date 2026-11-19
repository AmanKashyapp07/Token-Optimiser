"""Global managed HTTP client connection pooling for the Token Optimiser gateway.

Provides a shared httpx.AsyncClient singleton across the ASGI application lifespan,
ensuring connection reuse, socket pooling, and clean shutdown lifecycle management.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger("gateway.http")

_CLIENT: httpx.AsyncClient | None = None


def get_http_client(request: Any = None) -> httpx.AsyncClient:
    """Retrieves the active shared httpx.AsyncClient singleton.

    If an active FastAPI request with an app-scoped client is provided, that client
    is returned; otherwise the global module-level singleton is returned or created.

    Args:
        request: Optional FastAPI Request instance.

    Returns:
        Shared httpx.AsyncClient instance with configured connection limits.
    """
    global _CLIENT
    if request is not None and hasattr(request, "app") and hasattr(request.app.state, "http_client"):
        return request.app.state.http_client

    if _CLIENT is None or _CLIENT.is_closed:
        limits = httpx.Limits(max_keepalive_connections=50, max_connections=200, keepalive_expiry=30.0)
        timeout = httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)
        _CLIENT = httpx.AsyncClient(limits=limits, timeout=timeout)
    return _CLIENT


async def init_http_client(
    max_keepalive_connections: int = 50,
    max_connections: int = 200,
    keepalive_expiry: float = 30.0,
    read_timeout: float = 30.0,
) -> httpx.AsyncClient:
    """Initializes the global singleton HTTP client with configured connection limits."""
    global _CLIENT
    if _CLIENT is not None and not _CLIENT.is_closed:
        await _CLIENT.aclose()

    limits = httpx.Limits(
        max_keepalive_connections=max_keepalive_connections,
        max_connections=max_connections,
        keepalive_expiry=keepalive_expiry,
    )
    timeout = httpx.Timeout(connect=5.0, read=read_timeout, write=10.0, pool=5.0)
    _CLIENT = httpx.AsyncClient(limits=limits, timeout=timeout)
    logger.debug("Initialized shared HTTP client connection pool (max_connections=%d)", max_connections)
    return _CLIENT


async def close_http_client() -> None:
    """Gracefully closes all connections in the shared HTTP client pool."""
    global _CLIENT
    if _CLIENT is not None and not _CLIENT.is_closed:
        logger.debug("Closing shared HTTP client connection pool")
        await _CLIENT.aclose()
        _CLIENT = None
