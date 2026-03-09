"""Tier 4: Semantic Store -- persist and query the long-term knowledge graph.

Translates between SemanticNode/SemanticEdge domain models and the
memories/edges tables, composing both repositories.
"""

from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from smriti.db.repository import EdgeRepository, MemoryRepository
from smriti.db.tables import EdgesTable, MemoriesTable
from smriti.models.memory import MemoryTier, SemanticEdge, SemanticNode


class SemanticStore:
    def __init__(
        self,
        mem_repo: MemoryRepository | None = None,
        edge_repo: EdgeRepository | None = None,
    ) -> None:
        self._mem = mem_repo or MemoryRepository()
        self._edge = edge_repo or EdgeRepository()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    async def save_node(self, session: AsyncSession, node: SemanticNode) -> None:
        row = MemoriesTable(
            id=node.id,
            tier=MemoryTier.SEMANTIC.value,
            content=node.label,
            facts={
                "node_type": node.node_type,
                "properties": node.properties,
                "confidence": node.confidence,
                "source_episodes": [str(ep) for ep in node.source_episodes],
            },
            importance=node.confidence,
            created_at=node.created_at,
            accessed_at=node.updated_at,
        )
        await self._mem.insert(session, row)

    async def get_node(self, session: AsyncSession, node_id: UUID) -> SemanticNode | None:
        row = await self._mem.get_by_id(session, node_id)
        if row is None or row.tier != MemoryTier.SEMANTIC.value:
            return None
        return self._row_to_node(row)

    async def list_nodes(
        self, session: AsyncSession, *, limit: int = 100, offset: int = 0
    ) -> list[SemanticNode]:
        rows = await self._mem.list_by_tier(
            session, MemoryTier.SEMANTIC.value, limit=limit, offset=offset
        )
        return [self._row_to_node(r) for r in rows]

    async def search_nodes_by_vector(
        self,
        session: AsyncSession,
        embedding: list[float],
        *,
        limit: int = 10,
    ) -> list[tuple[SemanticNode, float]]:
        results = await self._mem.vector_search(
            session, embedding, tier=MemoryTier.SEMANTIC.value, limit=limit
        )
        return [(self._row_to_node(row), dist) for row, dist in results]

    async def delete_node(self, session: AsyncSession, node_id: UUID) -> None:
        await self._edge.delete_edges_for(session, node_id)
        await self._mem.delete_by_id(session, node_id)

    # ------------------------------------------------------------------
    # Edges
    # ------------------------------------------------------------------

    async def save_edge(self, session: AsyncSession, edge: SemanticEdge) -> None:
        row = EdgesTable(
            source_id=edge.source_id,
            target_id=edge.target_id,
            relation=edge.relation,
            weight=edge.weight,
            metadata_=edge.metadata or None,
        )
        await self._edge.insert(session, row)

    async def get_outgoing(
        self, session: AsyncSession, node_id: UUID
    ) -> list[SemanticEdge]:
        rows = await self._edge.get_edges_from(session, node_id)
        return [self._row_to_edge(r) for r in rows]

    async def get_incoming(
        self, session: AsyncSession, node_id: UUID
    ) -> list[SemanticEdge]:
        rows = await self._edge.get_edges_to(session, node_id)
        return [self._row_to_edge(r) for r in rows]

    async def get_neighbors(
        self, session: AsyncSession, node_id: UUID
    ) -> list[tuple[SemanticNode, SemanticEdge]]:
        """Return all nodes connected via outgoing edges from node_id."""
        edges = await self.get_outgoing(session, node_id)
        result: list[tuple[SemanticNode, SemanticEdge]] = []
        for edge in edges:
            neighbor = await self.get_node(session, edge.target_id)
            if neighbor is not None:
                result.append((neighbor, edge))
        return result

    # ------------------------------------------------------------------
    # Internal converters
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_node(row: MemoriesTable) -> SemanticNode:
        facts = row.facts or {}
        return SemanticNode(
            id=row.id,
            label=row.content,
            node_type=facts.get("node_type", ""),
            properties=facts.get("properties", {}),
            confidence=facts.get("confidence", row.importance),
            source_episodes=[
                UUID(ep) for ep in facts.get("source_episodes", [])
            ],
            created_at=row.created_at,
            updated_at=row.accessed_at,
        )

    @staticmethod
    def _row_to_edge(row: EdgesTable) -> SemanticEdge:
        return SemanticEdge(
            source_id=row.source_id,
            target_id=row.target_id,
            relation=row.relation,
            weight=row.weight,
            metadata=row.metadata_ or {},
        )
