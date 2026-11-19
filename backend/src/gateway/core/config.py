"""Application configuration management for the Token Optimiser gateway.

Reads configuration from environment variables and `.env` files using Pydantic Settings.
Enforces type safety, validates network URLs and file paths, and provides clear runtime defaults.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directory layout anchors
ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "backend" / "data"
RUNTIME = DATA / "runtime"
ARTIFACTS = DATA / "artifacts"


class Settings(BaseSettings):
    """Immutable production settings schema for Token Optimiser.

    Attributes:
        mode: Gateway operating mode ('demo' uses local fixtures, 'live' proxies to real providers).
        environment: Deployment environment ('development', 'test', 'demo', 'production').
        redis_url: Optional connection string for distributed caching and singleflight.
        host: Host IP binding for the Uvicorn server.
        port: Listening port for the HTTP server.
        api_base_url: Base URL for upstream OpenAI-compatible provider (e.g. 'https://api.openai.com/v1').
        api_key: Upstream provider secret key. Stored server-side only.
        cheap_model: Model name for economy/simple queries (default 'gpt-4o-mini' or 'demo-cheap').
        strong_model: Model name for complex reasoning queries (default 'gpt-4o' or 'demo-strong').
        cheap_input_price: Cost in USD per 1K input tokens for cheap tier.
        cheap_output_price: Cost in USD per 1K output tokens for cheap tier.
        strong_input_price: Cost in USD per 1K input tokens for strong tier.
        strong_output_price: Cost in USD per 1K output tokens for strong tier.
        encoding: Tiktoken encoding name (e.g. 'cl100k_base' or 'o200k_base').
        cache_threshold: Minimum cosine similarity required to serve a semantic cache hit.
        database_path: SQLite runtime metrics database location.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=False,
    )

    mode: Literal["demo", "live"] = "demo"
    environment: Literal["development", "test", "demo", "production"] = "demo"
    redis_url: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    api_base_url: str = ""
    api_key: str = ""
    cheap_model: str = "demo-cheap"
    strong_model: str = "demo-strong"
    cheap_input_price: float | None = None
    cheap_output_price: float | None = None
    strong_input_price: float | None = None
    strong_output_price: float | None = None
    encoding: str = "cl100k_base"
    cache_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    database_path: Path = RUNTIME / "iml.sqlite3"

    @field_validator("api_base_url", mode="before")
    @classmethod
    def normalize_api_base_url(cls, v: str | None) -> str:
        """Strips trailing slashes and normalizes base URL."""
        if not v:
            return ""
        return str(v).strip().rstrip("/")

    @property
    def is_live(self) -> bool:
        """Returns True if gateway is configured to dispatch live upstream inference calls."""
        return bool(self.mode == "live" and self.api_base_url and self.api_key)

    @property
    def is_production(self) -> bool:
        """Returns True if the runtime environment is set to production."""
        return self.environment == "production"

    @property
    def is_demo(self) -> bool:
        """Returns True if the runtime environment allows demo fixture fallback."""
        return self.environment in ("demo", "development", "test")

    @property
    def has_redis(self) -> bool:
        """Returns True if a Redis connection URL is configured."""
        return bool(self.redis_url and self.redis_url.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retrieves cached application settings singleton and ensures runtime directories exist."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    return Settings()
