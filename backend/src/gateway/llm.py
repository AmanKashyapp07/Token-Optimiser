"""Legacy research LLM client adapter for Token Optimiser.

Maintains backward-compatible client interfaces for imported research modules
(m2_compressor, m4_router) while utilizing modern pooled HTTP connections and strict types.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

from gateway.core.http import get_http_client
from gateway.schemas import ChatMessage, Usage

logger = logging.getLogger("gateway.llm")

try:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    def _httpx_retry():
        return retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=4.0),
            retry=retry_if_exception_type((httpx.HTTPError, httpx.ConnectError, httpx.TimeoutException)),
            reraise=True,
        )
except Exception:  # tenacity not installed

    def _httpx_retry():  # type: ignore
        def deco(fn):
            return fn

        return deco


@dataclass(slots=True)
class LLMResult:
    """Container for completion output and token telemetry from legacy clients."""

    content: str
    usage: Usage
    model: str
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLMClient:
    """Abstract base class for legacy research clients."""

    model: str = ""

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        raise NotImplementedError

    async def stream_chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        result = await self.chat(messages, temperature, max_tokens)
        for token in result.content.split(" "):
            yield token + " "


class MockLLMClient(BaseLLMClient):
    """Deterministic, offline client used for testing and dry-run execution."""

    def __init__(self, model: str, canned: str | None = None) -> None:
        self.model = model
        self.canned = canned
        self.last_usage: dict[str, Any] | None = None

    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        last = messages[-1]["content"] if isinstance(messages[-1], dict) else messages[-1].content
        if not self.canned:
            content = f"[mock:{self.model}] {last}"
        else:
            content = self.canned

        prompt_tokens = len(content.split())
        completion_tokens = max(1, len(content.split()) // 2 + 1)
        self.last_usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
        return LLMResult(
            content=content,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            model=self.model,
        )


class OpenAIClient(BaseLLMClient):
    """Async OpenAI-compatible chat client using pooled HTTP connections."""

    def __init__(self, model: str, base_url: str, api_key: str = "", timeout: float = 60.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.last_usage: dict[str, Any] | None = None

    @staticmethod
    def _dump(messages: list[ChatMessage] | list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if isinstance(m, dict):
                out.append(m)
            else:
                out.append(m.model_dump(exclude_none=True))
        return out

    async def stream_chat(  # type: ignore[override]
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._dump(messages),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        self.last_usage = None
        client = get_http_client()
        async with client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    if "usage" in obj and obj["usage"]:
                        self.last_usage = obj["usage"]
                    delta = obj.get("choices", [{}])[0].get("delta", {}) if obj.get("choices") else {}
                    content = delta.get("content")
                    if content:
                        yield content
                except Exception:
                    continue

    @_httpx_retry()
    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> LLMResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._dump(messages),
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        client = get_http_client()
        resp = await client.post(
            f"{self.base_url}/chat/completions", headers=self._headers, json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        content = choice["message"].get("content") or choice["message"].get("reasoning") or ""
        usage = Usage(**data.get("usage", {}))
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        return LLMResult(content=content, usage=usage, model=self.model, raw=data)


def build_client(
    tier_name: str,
    model: str,
    base_url: str,
    api_key: str,
    dry_run: bool,
    canned: str | None = None,
    timeout: float = 30.0,
) -> BaseLLMClient:
    """Builds appropriate LLM client instance based on dry_run mode."""
    if dry_run:
        return MockLLMClient(model=model, canned=canned)
    return OpenAIClient(model=model, base_url=base_url, api_key=api_key, timeout=timeout)


def new_task_id() -> str:
    """Generates a unique task identifier prefixed with 'cmg-'."""
    return "cmg-" + uuid.uuid4().hex[:16]
