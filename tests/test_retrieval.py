"""Tests for the retrieval engine: scoring and search."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from smriti.db.repository import MemoryRepository
from smriti.db.tables import MemoriesTable
from smriti.retrieval import RankedMemory, RetrievalConfig, RetrievalEngine


# ---------------------------------------------------------------------------
# Scoring unit tests (no DB needed)
# ---------------------------------------------------------------------------


class TestScoring:
    def _engine(self, **kwargs) -> RetrievalEngine:
        return RetrievalEngine(config=RetrievalConfig(**kwargs))

    def test_recency_recent_is_high(self) -> None:
        engine = self._engine()
        now = datetime.now(timezone.utc)
        score = engine._recency_score(now - timedelta(minutes=5), now)
        assert score > 0.99

    def test_recency_old_is_low(self) -> None:
        engine = self._engine(decay_hours=24.0)
        now = datetime.now(timezone.utc)
        score = engine._recency_score(now - timedelta(days=7), now)
        assert score < 0.01

    def test_recency_decays_exponentially(self) -> None:
        engine = self._engine(decay_hours=24.0)
        now = datetime.now(timezone.utc)
        s1 = engine._recency_score(now - timedelta(hours=24), now)
        s2 = engine._recency_score(now - timedelta(hours=48), now)
        assert s1 == pytest.approx(math.exp(-1), rel=1e-3)
        assert s2 == pytest.approx(math.exp(-2), rel=1e-3)

    def test_importance_zero_access_is_zero(self) -> None:
        assert RetrievalEngine._importance_score(0.9, 0) == 0.0

    def test_importance_grows_with_access(self) -> None:
        s1 = RetrievalEngine._importance_score(0.8, 1)
        s5 = RetrievalEngine._importance_score(0.8, 5)
        assert s5 > s1 > 0

    def test_importance_scales_with_importance(self) -> None:
        s_high = RetrievalEngine._importance_score(0.9, 3)
        s_low = RetrievalEngine._importance_score(0.3, 3)
        assert s_high > s_low

    def test_final_score_weights(self) -> None:
        engine = self._engine(alpha=1.0, beta=0.0, gamma=0.0)
        now = datetime.now(timezone.utc)
        row = MemoriesTable(
            id=uuid.uuid4(), tier="episodic", content="test",
            accessed_at=now, importance=0.9, access_count=5,
        )
        ranked = engine._score(row, distance=0.2, now=now)
        assert ranked.final_score == pytest.approx(0.8, rel=1e-3)
        assert ranked.relevance == pytest.approx(0.8, rel=1e-3)


# ---------------------------------------------------------------------------
# Integration tests (require Postgres)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRetrievalSearch:
    async def test_vector_search_returns_ranked(self, db_session: AsyncSession) -> None:
        repo = MemoryRepository()
        dim = 1536
        emb_close = [1.0] + [0.0] * (dim - 1)
        emb_far = [0.0] * (dim - 1) + [1.0]

        now = datetime.now(timezone.utc)
        await repo.insert(
            db_session,
            MemoriesTable(
                id=uuid.uuid4(), tier="episodic", content="close memory",
                embedding=emb_close, importance=0.9, accessed_at=now, access_count=3,
            ),
        )
        await repo.insert(
            db_session,
            MemoriesTable(
                id=uuid.uuid4(), tier="episodic", content="far memory",
                embedding=emb_far, importance=0.5, accessed_at=now - timedelta(days=30),
                access_count=0,
            ),
        )

        engine = RetrievalEngine(repo=repo)
        results = await engine.search(db_session, emb_close, top_k=2)
        assert len(results) == 2
        assert results[0].row.content == "close memory"
        assert results[0].final_score > results[1].final_score

    async def test_search_by_time(self, db_session: AsyncSession) -> None:
        repo = MemoryRepository()
        now = datetime.now(timezone.utc)

        await repo.insert(
            db_session,
            MemoriesTable(
                id=uuid.uuid4(), tier="episodic", content="old",
                importance=0.5, accessed_at=now - timedelta(days=30), access_count=0,
            ),
        )
        await repo.insert(
            db_session,
            MemoriesTable(
                id=uuid.uuid4(), tier="episodic", content="recent important",
                importance=0.9, accessed_at=now, access_count=5,
            ),
        )

        engine = RetrievalEngine(repo=repo)
        results = await engine.search_by_time(db_session, tier="episodic", limit=2)
        assert len(results) == 2
        assert results[0].row.content == "recent important"

    async def test_top_k_limits_results(self, db_session: AsyncSession) -> None:
        repo = MemoryRepository()
        dim = 1536
        query = [1.0] + [0.0] * (dim - 1)

        for i in range(5):
            await repo.insert(
                db_session,
                MemoriesTable(
                    id=uuid.uuid4(), tier="episodic", content=f"mem-{i}",
                    embedding=query, importance=0.5,
                ),
            )

        engine = RetrievalEngine(repo=repo)
        results = await engine.search(db_session, query, top_k=3)
        assert len(results) == 3
