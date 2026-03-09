"""Tests for the FastAPI HTTP API.

Uses httpx.AsyncClient with mocked daemon — no live infrastructure.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import smriti.api as api_module
from smriti.api import create_app
from smriti.config import Settings
from smriti.daemon import PipelineStats


def _mock_daemon() -> MagicMock:
    daemon = MagicMock()
    daemon.stats = PipelineStats(
        batches_processed=5,
        events_consumed=42,
        events_filtered=30,
        memories_created=18,
    )
    daemon.event_bus = AsyncMock()
    daemon.event_bus.publish = AsyncMock(return_value="1-0")
    daemon.event_bus.close = AsyncMock()
    daemon.provider = AsyncMock()
    daemon.provider.close = AsyncMock()
    daemon.provider.embed = AsyncMock(return_value=[[0.1] * 1536])
    daemon.engine = AsyncMock()
    daemon.engine.dispose = AsyncMock()
    daemon.session_factory = MagicMock()
    daemon.shutdown = AsyncMock()
    return daemon


@pytest.fixture
def app():
    """Create app with daemon pre-injected (skip real lifespan)."""
    application = create_app()
    return application


@pytest.fixture
def mock_daemon():
    daemon = _mock_daemon()
    original = api_module._daemon
    api_module._daemon = daemon
    api_module._start_time = 100.0
    yield daemon
    api_module._daemon = original


@pytest.fixture
async def client(app, mock_daemon):
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestHealth:
    @pytest.mark.asyncio
    async def test_returns_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data


class TestIngestEvents:
    @pytest.mark.asyncio
    async def test_accepts_events(self, client: AsyncClient, mock_daemon) -> None:
        resp = await client.post("/v1/events", json={
            "events": [
                {
                    "source": "terminal",
                    "event_type": "command",
                    "raw_content": "git push origin main",
                }
            ]
        })
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 1
        mock_daemon.event_bus.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_rejects_empty_events(self, client: AsyncClient) -> None:
        resp = await client.post("/v1/events", json={"events": []})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_accepts_multiple_events(self, client: AsyncClient, mock_daemon) -> None:
        events = [
            {"source": "terminal", "event_type": "command", "raw_content": f"cmd-{i}"}
            for i in range(3)
        ]
        resp = await client.post("/v1/events", json={"events": events})
        assert resp.status_code == 200
        assert resp.json()["accepted"] == 3
        assert mock_daemon.event_bus.publish.call_count == 3


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_calls_embed(self, client: AsyncClient, mock_daemon) -> None:
        with patch("smriti.api.RetrievalEngine") as MockEngine:
            engine_inst = AsyncMock()
            engine_inst.search = AsyncMock(return_value=[])
            MockEngine.return_value = engine_inst

            session = AsyncMock()
            session.__aenter__ = AsyncMock(return_value=session)
            session.__aexit__ = AsyncMock(return_value=False)
            mock_daemon.session_factory.return_value = session

            resp = await client.post("/v1/search", json={"query": "CORS debugging"})

        assert resp.status_code == 200
        mock_daemon.provider.embed.assert_called_once_with(["CORS debugging"])

    @pytest.mark.asyncio
    async def test_search_rejects_empty_query(self, client: AsyncClient) -> None:
        resp = await client.post("/v1/search", json={"query": ""})
        assert resp.status_code == 422


class TestStats:
    @pytest.mark.asyncio
    async def test_returns_stats(self, client: AsyncClient) -> None:
        resp = await client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["batches_processed"] == 5
        assert data["events_consumed"] == 42
        assert data["events_filtered"] == 30
        assert data["memories_created"] == 18
