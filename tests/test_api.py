"""Tests for the FastAPI HTTP API.

Uses httpx.AsyncClient with mocked daemon — no live infrastructure.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

import smriti.api as api_module
from smriti.api import create_app
from smriti.daemon import PipelineStats
from smriti.db.tables import EdgesTable, MemoriesTable


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


def _make_memory_row(**overrides):
    now = datetime.now(UTC)
    defaults = dict(
        id=uuid.uuid4(),
        tier="episodic",
        content="test memory content",
        facts=None,
        metadata_=None,
        importance=0.7,
        created_at=now,
        accessed_at=now,
        access_count=1,
    )
    defaults.update(overrides)
    row = MagicMock(spec=MemoriesTable)
    for k, v in defaults.items():
        setattr(row, k, v)
    return row


def _mock_session(mock_daemon):
    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    mock_daemon.session_factory.return_value = session
    return session


class TestListMemories:
    @pytest.mark.asyncio
    async def test_list_all(self, client: AsyncClient, mock_daemon) -> None:
        _mock_session(mock_daemon)
        rows = [_make_memory_row(content=f"mem-{i}") for i in range(3)]
        with patch("smriti.api.MemoryRepository") as mock_repo:
            mock_repo.return_value.list_all = AsyncMock(return_value=rows)
            resp = await client.get("/v1/memories")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["memories"]) == 3

    @pytest.mark.asyncio
    async def test_filter_by_tier(self, client: AsyncClient, mock_daemon) -> None:
        _mock_session(mock_daemon)
        row = _make_memory_row(tier="semantic")
        with patch("smriti.api.MemoryRepository") as mock_repo:
            mock_repo.return_value.list_all = AsyncMock(return_value=[row])
            resp = await client.get("/v1/memories?tier=semantic")

        assert resp.status_code == 200
        assert resp.json()["memories"][0]["tier"] == "semantic"

    @pytest.mark.asyncio
    async def test_text_search(self, client: AsyncClient, mock_daemon) -> None:
        _mock_session(mock_daemon)
        row = _make_memory_row(content="debugging CORS")
        with patch("smriti.api.MemoryRepository") as mock_repo:
            mock_repo.return_value.search_text = AsyncMock(return_value=[row])
            resp = await client.get("/v1/memories?q=CORS")

        assert resp.status_code == 200
        assert len(resp.json()["memories"]) == 1


class TestMemoryCounts:
    @pytest.mark.asyncio
    async def test_returns_counts(self, client: AsyncClient, mock_daemon) -> None:
        _mock_session(mock_daemon)
        with patch("smriti.api.MemoryRepository") as mock_repo:
            mock_repo.return_value.count_by_tier = AsyncMock(
                return_value={"episodic": 10, "semantic": 5}
            )
            resp = await client.get("/v1/memories/counts")

        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"]["episodic"] == 10
        assert data["counts"]["semantic"] == 5
        assert data["total"] == 15


class TestDeleteMemory:
    @pytest.mark.asyncio
    async def test_deletes_existing_memory(self, client: AsyncClient, mock_daemon) -> None:
        session = _mock_session(mock_daemon)
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=txn)
        txn.__aexit__ = AsyncMock(return_value=False)
        session.begin = MagicMock(return_value=txn)

        row = _make_memory_row()
        memory_id = str(row.id)

        with (
            patch("smriti.api.MemoryRepository") as mock_mem_repo,
            patch("smriti.api.EdgeRepository") as mock_edge_repo,
        ):
            mock_mem_repo.return_value.get_by_id = AsyncMock(return_value=row)
            mock_mem_repo.return_value.delete_by_id = AsyncMock()
            mock_edge_repo.return_value.delete_edges_for = AsyncMock()

            resp = await client.delete(f"/v1/memories/{memory_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["deleted"] is True
        assert data["id"] == memory_id

    @pytest.mark.asyncio
    async def test_returns_404_for_missing_memory(self, client: AsyncClient, mock_daemon) -> None:
        session = _mock_session(mock_daemon)
        txn = AsyncMock()
        txn.__aenter__ = AsyncMock(return_value=txn)
        txn.__aexit__ = AsyncMock(return_value=False)
        session.begin = MagicMock(return_value=txn)

        missing_id = str(uuid.uuid4())

        with (
            patch("smriti.api.MemoryRepository") as mock_mem_repo,
            patch("smriti.api.EdgeRepository"),
        ):
            mock_mem_repo.return_value.get_by_id = AsyncMock(return_value=None)
            resp = await client.delete(f"/v1/memories/{missing_id}")

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_rejects_invalid_uuid(self, client: AsyncClient, mock_daemon) -> None:
        resp = await client.delete("/v1/memories/not-a-uuid")
        assert resp.status_code == 422


class TestGraph:
    @pytest.mark.asyncio
    async def test_returns_nodes_and_edges(self, client: AsyncClient, mock_daemon) -> None:
        _mock_session(mock_daemon)
        node_a = _make_memory_row(
            tier="semantic", content="TypeScript", metadata_={"label": "TypeScript"},
        )
        node_b = _make_memory_row(
            tier="semantic", content="React", metadata_={"label": "React"},
        )

        edge = MagicMock(spec=EdgesTable)
        edge.source_id = node_a.id
        edge.target_id = node_b.id
        edge.relation = "used_with"
        edge.weight = 0.9

        with (
            patch("smriti.api.MemoryRepository") as mock_mem_repo,
            patch("smriti.api.EdgeRepository") as mock_edge_repo,
        ):
            mock_mem_repo.return_value.list_all = AsyncMock(return_value=[node_a, node_b])
            mock_edge_repo.return_value.get_all_edges = AsyncMock(return_value=[edge])
            resp = await client.get("/v1/graph")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 2
        assert len(data["edges"]) == 1
        assert data["edges"][0]["relation"] == "used_with"

    @pytest.mark.asyncio
    async def test_filters_orphan_edges(self, client: AsyncClient, mock_daemon) -> None:
        """Edges referencing nodes not in the result set are excluded."""
        _mock_session(mock_daemon)
        node = _make_memory_row(tier="semantic", metadata_={"label": "X"})

        orphan_edge = MagicMock(spec=EdgesTable)
        orphan_edge.source_id = node.id
        orphan_edge.target_id = uuid.uuid4()
        orphan_edge.relation = "orphan"
        orphan_edge.weight = 1.0

        with (
            patch("smriti.api.MemoryRepository") as mock_mem_repo,
            patch("smriti.api.EdgeRepository") as mock_edge_repo,
        ):
            mock_mem_repo.return_value.list_all = AsyncMock(return_value=[node])
            mock_edge_repo.return_value.get_all_edges = AsyncMock(return_value=[orphan_edge])
            resp = await client.get("/v1/graph")

        assert resp.status_code == 200
        assert len(resp.json()["edges"]) == 0
