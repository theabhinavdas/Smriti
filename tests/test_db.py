"""Integration tests for the database layer (requires running Postgres)."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from smriti.db.repository import EdgeRepository, MemoryRepository
from smriti.db.tables import EdgesTable, MemoriesTable

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# MemoryRepository
# ---------------------------------------------------------------------------


class TestMemoryRepository:
    async def test_insert_and_get(self, db_session: AsyncSession) -> None:
        repo = MemoryRepository()
        row = MemoriesTable(
            id=uuid.uuid4(),
            tier="episodic",
            content="Debugged CORS issue",
            facts={"key_facts": ["missing header"]},
            importance=0.85,
        )
        await repo.insert(db_session, row)
        fetched = await repo.get_by_id(db_session, row.id)
        assert fetched is not None
        assert fetched.content == "Debugged CORS issue"
        assert fetched.importance == pytest.approx(0.85)

    async def test_list_by_tier(self, db_session: AsyncSession) -> None:
        repo = MemoryRepository()
        for i in range(3):
            await repo.insert(
                db_session,
                MemoriesTable(id=uuid.uuid4(), tier="semantic", content=f"fact-{i}"),
            )
        await repo.insert(
            db_session,
            MemoriesTable(id=uuid.uuid4(), tier="episodic", content="episode"),
        )
        results = await repo.list_by_tier(db_session, "semantic")
        assert len(results) == 3
        assert all(r.tier == "semantic" for r in results)

    async def test_touch_increments_access(self, db_session: AsyncSession) -> None:
        repo = MemoryRepository()
        row = MemoriesTable(id=uuid.uuid4(), tier="episodic", content="test touch")
        await repo.insert(db_session, row)
        assert row.access_count == 0

        await repo.touch(db_session, row.id)
        await db_session.refresh(row)
        assert row.access_count == 1

    async def test_delete(self, db_session: AsyncSession) -> None:
        repo = MemoryRepository()
        row = MemoriesTable(id=uuid.uuid4(), tier="episodic", content="to delete")
        await repo.insert(db_session, row)
        await repo.delete_by_id(db_session, row.id)

        fetched = await repo.get_by_id(db_session, row.id)
        assert fetched is None

    async def test_vector_search(self, db_session: AsyncSession) -> None:
        repo = MemoryRepository()
        dim = 1536
        emb_a = [1.0] + [0.0] * (dim - 1)
        emb_b = [0.0] * (dim - 1) + [1.0]

        await repo.insert(
            db_session,
            MemoriesTable(
                id=uuid.uuid4(), tier="episodic", content="close", embedding=emb_a
            ),
        )
        await repo.insert(
            db_session,
            MemoriesTable(
                id=uuid.uuid4(), tier="episodic", content="far", embedding=emb_b
            ),
        )

        results = await repo.vector_search(db_session, emb_a, limit=2)
        assert len(results) == 2
        assert results[0][0].content == "close"


# ---------------------------------------------------------------------------
# EdgeRepository
# ---------------------------------------------------------------------------


class TestEdgeRepository:
    async def test_insert_and_query(self, db_session: AsyncSession) -> None:
        mem_repo = MemoryRepository()
        edge_repo = EdgeRepository()

        node_a = MemoriesTable(id=uuid.uuid4(), tier="semantic", content="user")
        node_b = MemoriesTable(id=uuid.uuid4(), tier="semantic", content="TypeScript")
        await mem_repo.insert(db_session, node_a)
        await mem_repo.insert(db_session, node_b)

        edge = EdgesTable(
            source_id=node_a.id, target_id=node_b.id, relation="prefers", weight=0.92
        )
        await edge_repo.insert(db_session, edge)

        outgoing = await edge_repo.get_edges_from(db_session, node_a.id)
        assert len(outgoing) == 1
        assert outgoing[0].relation == "prefers"

        incoming = await edge_repo.get_edges_to(db_session, node_b.id)
        assert len(incoming) == 1

    async def test_delete_edges_for_node(self, db_session: AsyncSession) -> None:
        mem_repo = MemoryRepository()
        edge_repo = EdgeRepository()

        a = MemoriesTable(id=uuid.uuid4(), tier="semantic", content="a")
        b = MemoriesTable(id=uuid.uuid4(), tier="semantic", content="b")
        c = MemoriesTable(id=uuid.uuid4(), tier="semantic", content="c")
        for node in (a, b, c):
            await mem_repo.insert(db_session, node)

        await edge_repo.insert(
            db_session, EdgesTable(source_id=a.id, target_id=b.id, relation="knows")
        )
        await edge_repo.insert(
            db_session, EdgesTable(source_id=c.id, target_id=a.id, relation="knows")
        )

        await edge_repo.delete_edges_for(db_session, a.id)

        assert await edge_repo.get_edges_from(db_session, a.id) == []
        assert await edge_repo.get_edges_to(db_session, a.id) == []
