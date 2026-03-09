"""Model provider: async client for OpenRouter chat completions and embeddings.

All LLM calls in Smriti (salience filtering, extraction, consolidation)
and all embedding calls route through this single client.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from smriti.config import ModelConfig

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class ProviderError(Exception):
    """Raised when the model provider returns a non-success response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Provider error {status_code}: {detail}")


class ModelProvider:
    """Async OpenRouter client for chat completions and embeddings."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or ModelConfig()
        if not self._config.api_key and http_client is None:
            raise ProviderError(
                0,
                "No API key configured. Set SMRITI_MODEL_API_KEY to your OpenRouter key.",
            )
        self._client = http_client or httpx.AsyncClient(
            base_url=OPENROUTER_BASE_URL,
            headers={
                "Authorization": f"Bearer {self._config.api_key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Chat completion. Returns the assistant's response text."""
        payload: dict[str, Any] = {
            "model": model or self._config.extraction_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }
        data = await self._post("/chat/completions", payload)
        return data["choices"][0]["message"]["content"]

    async def embed(
        self,
        texts: list[str],
        *,
        model: str | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        payload: dict[str, Any] = {
            "model": model or self._config.embedding_model,
            "input": texts,
        }
        data = await self._post("/embeddings", payload)
        sorted_data = sorted(data["data"], key=lambda d: d["index"])
        return [item["embedding"] for item in sorted_data]

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self._client.post(endpoint, json=payload)
        if response.status_code != 200:
            detail = response.text[:500]
            raise ProviderError(response.status_code, detail)
        return response.json()

    async def close(self) -> None:
        await self._client.aclose()
