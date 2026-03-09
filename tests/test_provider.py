"""Tests for ModelProvider (mocked HTTP, no real API calls)."""

from __future__ import annotations

import httpx
import pytest

from smriti.config import ModelConfig
from smriti.provider import ModelProvider, ProviderError

pytestmark = pytest.mark.asyncio


MOCK_BASE = "https://openrouter.ai/api/v1"


def _mock_client(handler) -> httpx.AsyncClient:
    """Create an httpx.AsyncClient with a mock transport and the correct base URL."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=MOCK_BASE,
    )


def _completion_response(content: str = "Hello!") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        },
    )


def _embedding_response(embeddings: list[list[float]] | None = None) -> httpx.Response:
    if embeddings is None:
        embeddings = [[0.1, 0.2, 0.3]]
    data = [{"embedding": emb, "index": i} for i, emb in enumerate(embeddings)]
    return httpx.Response(200, json={"data": data})


class TestModelProvider:
    async def test_complete_returns_content(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            assert "/chat/completions" in str(request.url)
            return _completion_response("The answer is 42.")

        provider = ModelProvider(
            config=ModelConfig(api_key="test-key"), http_client=_mock_client(handler)
        )

        result = await provider.complete([{"role": "user", "content": "What is 6*7?"}])
        assert result == "The answer is 42."

    async def test_complete_sends_correct_model(self) -> None:
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured["body"] = json.loads(request.content)
            return _completion_response()

        config = ModelConfig(api_key="k", extraction_model="anthropic/claude-3.5-haiku")
        provider = ModelProvider(config=config, http_client=_mock_client(handler))

        await provider.complete([{"role": "user", "content": "hi"}])
        assert captured["body"]["model"] == "anthropic/claude-3.5-haiku"

    async def test_complete_with_explicit_model(self) -> None:
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            import json
            captured["body"] = json.loads(request.content)
            return _completion_response()

        provider = ModelProvider(config=ModelConfig(api_key="k"), http_client=_mock_client(handler))

        await provider.complete(
            [{"role": "user", "content": "hi"}], model="google/gemini-2.0-flash"
        )
        assert captured["body"]["model"] == "google/gemini-2.0-flash"

    async def test_embed_returns_vectors(self) -> None:
        expected = [[0.1, 0.2], [0.3, 0.4]]

        async def handler(request: httpx.Request) -> httpx.Response:
            assert "/embeddings" in str(request.url)
            return _embedding_response(expected)

        provider = ModelProvider(config=ModelConfig(api_key="k"), http_client=_mock_client(handler))

        result = await provider.embed(["hello", "world"])
        assert result == expected

    async def test_embed_sorts_by_index(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"embedding": [0.9], "index": 1},
                        {"embedding": [0.1], "index": 0},
                    ]
                },
            )

        provider = ModelProvider(config=ModelConfig(api_key="k"), http_client=_mock_client(handler))

        result = await provider.embed(["a", "b"])
        assert result == [[0.1], [0.9]]

    async def test_error_raises_provider_error(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(429, text="Rate limit exceeded")

        provider = ModelProvider(config=ModelConfig(api_key="k"), http_client=_mock_client(handler))

        with pytest.raises(ProviderError) as exc_info:
            await provider.complete([{"role": "user", "content": "hi"}])
        assert exc_info.value.status_code == 429
        assert "Rate limit" in exc_info.value.detail

    async def test_sends_auth_header(self) -> None:
        captured = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            captured["auth"] = request.headers.get("authorization")
            return _completion_response()

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=MOCK_BASE,
            headers={"Authorization": "Bearer sk-or-test123"},
        )
        provider = ModelProvider(
            config=ModelConfig(api_key="sk-or-test123"), http_client=client
        )

        await provider.complete([{"role": "user", "content": "hi"}])
        assert captured["auth"] == "Bearer sk-or-test123"
