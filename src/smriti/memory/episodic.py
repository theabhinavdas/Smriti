"""Tier 3: Episodic Store -- persist and query compressed conversation episodes.

Translates between EpisodicMemory domain models and the memories table,
composing MemoryRepository for all database operations.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from smriti.db.repository import MemoryRepository
from smriti.db.tables import MemoriesTable
from smriti.models.memory import EpisodicMemory, MemoryTier


class EpisodicStore:
    def __init__(self, repo: MemoryRepository | None = None) -> None:
        self._repo = repo or MemoryRepository()

    async def save(self, session: AsyncSession, episode: EpisodicMemory) -> None:
        """Persist an episodic memory to Postgres."""
        row = MemoriesTable(
            id=episode.id,
            tier=MemoryTier.EPISODIC.value,
            content=episode.summary,
            embedding=episode.embedding or None,
            facts={
                "key_facts": episode.key_facts,
                "source": episode.source,
                "conversation_id": episode.conversation_id,
                "entities": episode.entities,
                "topics": episode.topics,
                "emotional_valence": episode.emotional_valence,
                "links": [link.model_dump(mode="json") for link in episode.links],
            },
            metadata_={
                "source": episode.source,
                "conversation_id": episode.conversation_id,
            },
            importance=episode.importance,
            created_at=episode.created_at,
            accessed_at=episode.last_accessed,
            access_count=episode.access_count,
        )
        await self._repo.insert(session, row)

    async def get(self, session: AsyncSession, episode_id: UUID) -> EpisodicMemory | None:
        """Load a single episode by ID."""
        row = await self._repo.get_by_id(session, episode_id)
        if row is None or row.tier != MemoryTier.EPISODIC.value:
            return None
        return self._row_to_model(row)

    async def list_recent(
        self, session: AsyncSession, *, limit: int = 20, offset: int = 0
    ) -> list[EpisodicMemory]:
        """List episodes ordered by most recent first."""
        rows = await self._repo.list_by_tier(
            session, MemoryTier.EPISODIC.value, limit=limit, offset=offset
        )
        return [self._row_to_model(r) for r in rows]

    async def search_by_vector(
        self,
        session: AsyncSession,
        embedding: list[float],
        *,
        limit: int = 10,
    ) -> list[tuple[EpisodicMemory, float]]:
        """Find similar episodes by vector. Returns (episode, distance) pairs."""
        results = await self._repo.vector_search(
            session, embedding, tier=MemoryTier.EPISODIC.value, limit=limit
        )
        return [(self._row_to_model(row), dist) for row, dist in results]

    async def touch(self, session: AsyncSession, episode_id: UUID) -> None:
        await self._repo.touch(session, episode_id)

    async def delete(self, session: AsyncSession, episode_id: UUID) -> None:
        await self._repo.delete_by_id(session, episode_id)

    @staticmethod
    def _row_to_model(row: MemoriesTable) -> EpisodicMemory:
        facts = row.facts or {}
        from smriti.models.memory import MemoryLink

        links = [MemoryLink.model_validate(l) for l in facts.get("links", [])]
        return EpisodicMemory(
            id=row.id,
            conversation_id=facts.get("conversation_id", ""),
            source=facts.get("source", ""),
            summary=row.content,
            key_facts=facts.get("key_facts", []),
            embedding=list(row.embedding) if row.embedding is not None else [],
            entities=facts.get("entities", []),
            topics=facts.get("topics", []),
            created_at=row.created_at,
            last_accessed=row.accessed_at,
            access_count=row.access_count,
            importance=row.importance,
            emotional_valence=facts.get("emotional_valence", 0.0),
            links=links,
        )
