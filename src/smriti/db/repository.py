"""Repository layer for memory and edge CRUD operations.

All public methods accept a session from the caller so that
transactions can be composed externally.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import delete, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from smriti.db.tables import EdgesTable, MemoriesTable


class MemoryRepository:
    """Read/write operations against the memories table."""

    async def insert(self, session: AsyncSession, row: MemoriesTable) -> None:
        session.add(row)
        await session.flush()

    async def get_by_id(self, session: AsyncSession, memory_id: UUID) -> MemoriesTable | None:
        result = await session.get(MemoriesTable, memory_id)
        return result

    async def list_by_tier(
        self,
        session: AsyncSession,
        tier: str,
        *,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoriesTable]:
        stmt = (
            select(MemoriesTable)
            .where(MemoriesTable.tier == tier)
            .order_by(MemoriesTable.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def vector_search(
        self,
        session: AsyncSession,
        embedding: list[float],
        *,
        tier: str | None = None,
        limit: int = 10,
    ) -> list[tuple[MemoriesTable, float]]:
        """Find memories by cosine similarity. Returns (memory, distance) pairs."""
        embedding_str = "[" + ",".join(str(v) for v in embedding) + "]"
        distance_expr = MemoriesTable.embedding.cosine_distance(text(f"'{embedding_str}'::vector"))

        stmt = (
            select(MemoriesTable, distance_expr.label("distance"))
            .where(MemoriesTable.embedding.is_not(None))
        )
        if tier is not None:
            stmt = stmt.where(MemoriesTable.tier == tier)
        stmt = stmt.order_by(distance_expr).limit(limit)

        result = await session.execute(stmt)
        return [(row[0], row[1]) for row in result.all()]

    async def touch(self, session: AsyncSession, memory_id: UUID) -> None:
        """Update accessed_at and increment access_count."""
        stmt = (
            update(MemoriesTable)
            .where(MemoriesTable.id == memory_id)
            .values(
                accessed_at=datetime.now(timezone.utc),
                access_count=MemoriesTable.access_count + 1,
            )
        )
        await session.execute(stmt)

    async def delete_by_id(self, session: AsyncSession, memory_id: UUID) -> None:
        stmt = delete(MemoriesTable).where(MemoriesTable.id == memory_id)
        await session.execute(stmt)


class EdgeRepository:
    """Read/write operations against the edges table."""

    async def insert(self, session: AsyncSession, row: EdgesTable) -> None:
        session.add(row)
        await session.flush()

    async def get_edges_from(
        self,
        session: AsyncSession,
        source_id: UUID,
    ) -> list[EdgesTable]:
        stmt = select(EdgesTable).where(EdgesTable.source_id == source_id)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_edges_to(
        self,
        session: AsyncSession,
        target_id: UUID,
    ) -> list[EdgesTable]:
        stmt = select(EdgesTable).where(EdgesTable.target_id == target_id)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_edges_by_relation(
        self,
        session: AsyncSession,
        relation: str,
    ) -> list[EdgesTable]:
        stmt = select(EdgesTable).where(EdgesTable.relation == relation)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def delete_edges_for(self, session: AsyncSession, node_id: UUID) -> None:
        """Remove all edges where node_id is source or target."""
        await session.execute(
            delete(EdgesTable).where(EdgesTable.source_id == node_id)
        )
        await session.execute(
            delete(EdgesTable).where(EdgesTable.target_id == node_id)
        )
