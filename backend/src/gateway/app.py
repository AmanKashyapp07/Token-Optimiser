"""FastAPI application factory and lifecycle management for Token Optimiser.

Manages connection pools, global middleware, exception handlers, and API routing.
Adheres to modern ASGI lifespan conventions and enterprise observability standards.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.benchmarking import router as benchmark_router
from .core.chat import router as chat_router
from .core.config import get_settings
from .core.exceptions import GatewayError, RateLimitExceededError
from .core.http import close_http_client, get_http_client, init_http_client
from .core.middleware import RateLimiterMiddleware, RequestTracingMiddleware
from .core.pipeline import router as gateway_pipeline_router
from .features.cache.api import router as cache_router
from .features.compression.api import router as compression_router
from .features.insights.api import router as insights_router
from .features.routing.api import router as routing_router

logger = logging.getLogger("gateway.app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manages application lifecycle: resource initialization and graceful teardown.

    Initializes shared HTTP connection pools and prepares runtime directories on startup.
    Ensures all active socket connections are gracefully drained and closed on shutdown.
    """
    settings = get_settings()
    logger.info(
        "Starting Token Optimiser gateway [environment=%s, mode=%s]", settings.environment, settings.mode
    )

    # Initialize high-performance shared HTTP connection pool
    client = await init_http_client()
    app.state.http_client = client

    try:
        yield
    finally:
        logger.info("Shutting down Token Optimiser gateway: draining connection pools")
        await close_http_client()


app = FastAPI(
    title="IML Token Optimiser",
    version="0.1.0",
    description="Cost-aware and resilient inference gateway for LLM requests.",
    lifespan=lifespan,
)


# Global custom exception handler formatting OpenAI-compliant error envelopes
@app.exception_handler(GatewayError)
async def gateway_error_handler(request: Request, exc: GatewayError) -> JSONResponse:
    """Catches domain gateway exceptions and converts them to standard error JSON."""
    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitExceededError):
        headers["Retry-After"] = str(getattr(exc, "retry_after", 60))
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict(),
        headers=headers,
    )


# High-precision observability and rate limiting middleware
app.add_middleware(RequestTracingMiddleware)
app.add_middleware(RateLimiterMiddleware, requests_per_minute=240)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
def health() -> dict[str, str | bool]:
    """Health check endpoint reporting gateway mode and upstream readiness."""
    settings = get_settings()
    live = bool(settings.mode == "live" and settings.api_base_url and settings.api_key)
    return {
        "ok": True,
        "mode": "live" if live else "demo",
        "cheap_model": settings.cheap_model,
        "strong_model": settings.strong_model,
        "message": "Ready for local use."
        if live
        else "Running in demo mode with authored fixtures and local utilities.",
    }


# Feature and pipeline routers
app.include_router(chat_router)
app.include_router(benchmark_router)
app.include_router(gateway_pipeline_router)
app.include_router(compression_router)
app.include_router(routing_router)
app.include_router(cache_router)
app.include_router(insights_router)

# Re-export get_http_client for convenience
__all__ = ["app", "get_http_client"]
