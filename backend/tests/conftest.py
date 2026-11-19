"""Global shared pytest configuration and fixtures for Token Optimiser tests.

Provides centralized client lifecycles, environment overrides, automatic singleton resets,
and isolated temporary stores adhering to top-tier production testing standards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient

from gateway.app import app
from gateway.core.config import Settings
from gateway.core.resilience import get_circuit_breaker
from gateway.core.store import Store
from gateway.features.cache.coalescing import get_request_coalescer


@pytest.fixture(autouse=True)
def reset_gateway_singletons() -> Generator[None, None, None]:
    """Ensures circuit breaker and singleflight coalescer are pristine for every test."""
    cb = get_circuit_breaker()
    coalescer = get_request_coalescer()
    cb.reset()
    coalescer.reset()
    yield
    cb.reset()
    coalescer.reset()


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """Session-scoped TestClient reusing the ASGI application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def demo_settings() -> Settings:
    """Returns an isolated Settings instance configured for offline demo execution."""
    return Settings(
        mode="demo",
        environment="demo",
        cheap_model="demo-cheap",
        strong_model="demo-strong",
    )


@pytest.fixture
def live_settings() -> Settings:
    """Returns an isolated Settings instance configured for live provider execution."""
    return Settings(
        mode="live",
        environment="production",
        api_base_url="https://api.openai.com/v1",
        api_key="test-live-key",
        cheap_model="gpt-4o-mini",
        strong_model="gpt-4o",
    )


@pytest.fixture
def temp_store(tmp_path: Path) -> Store:
    """Creates a temporary, isolated SQLite store."""
    db_file = tmp_path / "test_store.sqlite3"
    return Store(db_file)
