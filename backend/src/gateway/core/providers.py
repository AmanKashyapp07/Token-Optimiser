from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx

from gateway.tokenizer import TokenCounter
from .config import ROOT, Settings

FIXTURES = ROOT / "backend" / "data" / "fixtures" / "demo_answers.json"


# ==============================================================================
# Standardized Provider Exceptions
# ==============================================================================

class ProviderError(Exception):
    """Base class for all upstream LLM provider and gateway errors."""

    def __init__(self, message: str, status_code: int = 502, code: str = "provider_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class ProviderTimeoutError(ProviderError):
    def __init__(self, message: str = "Upstream inference provider timed out"):
        super().__init__(message, status_code=504, code="gateway_timeout")


class ProviderRateLimitError(ProviderError):
    def __init__(self, message: str = "Upstream inference provider rate limit exceeded"):
        super().__init__(message, status_code=429, code="provider_rate_limited")


class ProviderAuthenticationError(ProviderError):
    def __init__(self, message: str = "Upstream provider authentication failed"):
        super().__init__(message, status_code=502, code="bad_gateway_auth")


class ProviderUnavailableError(ProviderError):
    def __init__(self, message: str = "Upstream inference provider service unavailable"):
        super().__init__(message, status_code=503, code="service_unavailable")


# ==============================================================================
# Data Contracts
# ==============================================================================

@dataclass
class Generation:
    text: str
    model: str
    mode: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    source: str = "fixture"
    finish_reason: str = "stop"
    latency_ms: float = 0.0


# ==============================================================================
# Consolidated Provider Engine
# ==============================================================================

class Provider:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.fixtures = self._load_fixtures()
        self.counter = TokenCounter()

    def _load_fixtures(self) -> list[dict]:
        if not FIXTURES.exists():
            return []
        try:
            return json.loads(FIXTURES.read_text(encoding="utf-8"))
        except Exception:
            return []

    def is_live(self) -> bool:
        return bool(
            self.settings.mode == "live"
            and self.settings.api_base_url
            and self.settings.api_key
            and self.settings.cheap_model
        )

    def _truncate_to_max_tokens(self, text: str, max_tokens: int) -> tuple[str, bool]:
        """Truncates completion text to strictly obey max_tokens boundary."""
        tokens = self.counter.count(text)
        if tokens <= max_tokens:
            return text, False

        words = text.split()
        truncated_words: list[str] = []
        for word in words:
            candidate = " ".join(truncated_words + [word])
            if self.counter.count(candidate) > max_tokens:
                break
            truncated_words.append(word)

        capped_text = " ".join(truncated_words)
        if capped_text and not capped_text.endswith("."):
            capped_text += "..."
        return capped_text or text[:max_tokens * 4], True

    def answer_fixture(self, text: str, max_tokens: int | None = None) -> Generation:
        start_t = time.perf_counter()
        lowered = text.lower()
        selected_text = None
        source_id = "fallback"

        for item in self.fixtures:
            if any(token in lowered for token in item.get("match", [])):
                selected_text = item["answer"]
                source_id = item.get("id", "fixture")
                break

        if selected_text is None:
            selected_text = (
                "I can demonstrate the optimisation pipeline for this request. In live mode, the selected model generates the final answer; "
                "in demo mode, the app uses curated responses for common support examples."
            )

        finish_reason = "stop"
        if max_tokens is not None and max_tokens > 0:
            selected_text, was_capped = self._truncate_to_max_tokens(selected_text, max_tokens)
            if was_capped:
                finish_reason = "length"

        prompt_tokens = self.counter.count(text)
        completion_tokens = self.counter.count(selected_text)
        latency_ms = (time.perf_counter() - start_t) * 1000

        return Generation(
            text=selected_text,
            model=self.settings.cheap_model,
            mode="demo",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            source=source_id,
            finish_reason=finish_reason,
            latency_ms=round(latency_ms, 2),
        )

    async def generate(
        self,
        text: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        max_retries: int = 2,
    ) -> Generation:
        start_t = time.perf_counter()

        if not self.is_live():
            out = self.answer_fixture(text, max_tokens=max_tokens)
            out.model = model
            return out

        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "temperature": temperature,
        }
        if max_tokens is not None and max_tokens > 0:
            payload["max_tokens"] = max_tokens

        last_err: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        self.settings.api_base_url.rstrip("/") + "/chat/completions",
                        headers={"Authorization": f"Bearer {self.settings.api_key}"},
                        json=payload,
                    )

                if response.status_code == 401 or response.status_code == 403:
                    raise ProviderAuthenticationError(f"Authentication failed: {response.text}")
                elif response.status_code == 429:
                    if attempt < max_retries:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    raise ProviderRateLimitError(f"Provider rate limited: {response.text}")
                elif response.status_code >= 500:
                    if attempt < max_retries:
                        await asyncio.sleep(0.5 * (2 ** attempt))
                        continue
                    raise ProviderUnavailableError(f"Provider error ({response.status_code}): {response.text}")

                response.raise_for_status()
                data = response.json()
                break
            except httpx.TimeoutException as exc:
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                raise ProviderTimeoutError(f"Provider timed out after 30s: {exc}") from exc
            except httpx.HTTPError as exc:
                last_err = exc
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * (2 ** attempt))
                    continue
                raise ProviderError(f"Upstream HTTP error: {exc}") from exc

        choice_obj = data.get("choices", [{}])[0]
        choice = choice_obj.get("message", {}).get("content", "")
        finish_reason = choice_obj.get("finish_reason", "stop")
        usage = data.get("usage") or {}

        prompt_tokens = usage.get("prompt_tokens") or self.counter.count(text)
        completion_tokens = usage.get("completion_tokens") or self.counter.count(choice)
        total_tokens = usage.get("total_tokens") or (prompt_tokens + completion_tokens)
        latency_ms = (time.perf_counter() - start_t) * 1000

        return Generation(
            text=choice,
            model=model,
            mode="live",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            source="live",
            finish_reason=finish_reason,
            latency_ms=round(latency_ms, 2),
        )

    async def stream(
        self,
        text: str,
        model: str,
        max_tokens: int | None = None,
        temperature: float = 0.2,
    ) -> AsyncIterator[str]:
        result = await self.generate(text, model, max_tokens=max_tokens, temperature=temperature)
        for chunk in result.text.split():
            yield chunk + " "
