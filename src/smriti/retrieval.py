"""Retrieval engine: hybrid search across memory tiers + multi-signal ranking.

Runs on every LLM call to find the most relevant memories for context assembly.

Scoring formula (Park et al. 2023, Generative Agents):
  score = alpha * relevance + beta * recency + gamma * importance

Where:
  relevance  = 1.0 - cosine_distance  (from pgvector)
  recency    = exp(-decay * hours_since_last_access)
  importance = memory.importance * log(1 + access_count)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from smriti.db.repository import MemoryRepository
from smriti.db.tables import MemoriesTable


@dataclass
class RankedMemory:
    """A memory annotated with retrieval scores."""

    row: MemoriesTable
    relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0
    final_score: float = 0.0


@dataclass
class RetrievalConfig:
    """Tunable weights for the scoring function."""

    alpha: float = 0.5
    beta: float = 0.3
    gamma: float = 0.2
    decay_hours: float = 168.0  # half-life ~1 week
    top_k: int = 20


class RetrievalEngine:
    def __init__(
        self,
        repo: MemoryRepository | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._repo = repo or MemoryRepository()
        self._config = config or RetrievalConfig()

    async def search(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        *,
        top_k: int | None = None,
        tier: str | None = None,
    ) -> list[RankedMemory]:
        """Hybrid search: vector similarity + multi-signal ranking."""
        k = top_k or self._config.top_k
        candidates = await self._repo.vector_search(
            session, query_embedding, tier=tier, limit=k * 2,
        )

        now = datetime.now(timezone.utc)
        ranked = [self._score(row, distance, now) for row, distance in candidates]
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked[:k]

    async def search_by_time(
        self,
        session: AsyncSession,
        *,
        tier: str | None = None,
        limit: int = 20,
    ) -> list[RankedMemory]:
        """Retrieve recent memories ranked by recency + importance (no vector query)."""
        tier_key = tier or "episodic"
        rows = await self._repo.list_by_tier(session, tier_key, limit=limit)
        now = datetime.now(timezone.utc)
        ranked = [self._score_without_vector(row, now) for row in rows]
        ranked.sort(key=lambda r: r.final_score, reverse=True)
        return ranked

    def _score(
        self, row: MemoriesTable, distance: float, now: datetime
    ) -> RankedMemory:
        relevance = 1.0 - distance
        recency = self._recency_score(row.accessed_at, now)
        importance = self._importance_score(row.importance, row.access_count)

        c = self._config
        final = c.alpha * relevance + c.beta * recency + c.gamma * importance

        return RankedMemory(
            row=row,
            relevance=relevance,
            recency=recency,
            importance=importance,
            final_score=final,
        )

    def _score_without_vector(self, row: MemoriesTable, now: datetime) -> RankedMemory:
        recency = self._recency_score(row.accessed_at, now)
        importance = self._importance_score(row.importance, row.access_count)

        beta_norm = self._config.beta / (self._config.beta + self._config.gamma)
        gamma_norm = self._config.gamma / (self._config.beta + self._config.gamma)
        final = beta_norm * recency + gamma_norm * importance

        return RankedMemory(
            row=row, recency=recency, importance=importance, final_score=final
        )

    def _recency_score(self, accessed_at: datetime, now: datetime) -> float:
        hours = (now - accessed_at).total_seconds() / 3600.0
        return math.exp(-hours / self._config.decay_hours)

    @staticmethod
    def _importance_score(importance: float, access_count: int) -> float:
        return importance * math.log(1.0 + access_count)
