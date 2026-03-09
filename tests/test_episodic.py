"""Integration tests for Tier 3: Episodic Store (requires running Postgres)."""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from smriti.memory.episodic import EpisodicStore
from smriti.models.memory import EpisodicMemory, MemoryLink

pytestmark = pytest.mark.asyncio


def _make_episode(summary: str = "Debugged CORS issue", **kwargs) -> EpisodicMemory:
    return EpisodicMemory(
        summary=summary,
        source="cursor",
        key_facts=["missing Allow-Origin header", "fixed with middleware"],
        entities=["auth_service", "CORS"],
        topics=["debugging", "backend"],
        importance=0.85,
        **kwargs,
    )


class TestEpisodicStore:
    async def test_save_and_get(self, db_session: AsyncSession) -> None:
        store = EpisodicStore()
        ep = _make_episode()
        await store.save(db_session, ep)

        loaded = await store.get(db_session, ep.id)
        assert loaded is not None
        assert loaded.summary == "Debugged CORS issue"
        assert loaded.source == "cursor"
        assert loaded.key_facts == ["missing Allow-Origin header", "fixed with middleware"]
        assert loaded.entities == ["auth_service", "CORS"]
        assert loaded.importance == pytest.approx(0.85)

    async def test_get_missing_returns_none(self, db_session: AsyncSession) -> None:
        store = EpisodicStore()
        from uuid import uuid4

        assert await store.get(db_session, uuid4()) is None

    async def test_list_recent(self, db_session: AsyncSession) -> None:
        store = EpisodicStore()
        for i in range(5):
            await store.save(db_session, _make_episode(f"episode-{i}"))

        results = await store.list_recent(db_session, limit=3)
        assert len(results) == 3
        assert results[0].summary == "episode-4"

    async def test_delete(self, db_session: AsyncSession) -> None:
        store = EpisodicStore()
        ep = _make_episode()
        await store.save(db_session, ep)
        await store.delete(db_session, ep.id)
        assert await store.get(db_session, ep.id) is None

    async def test_touch_updates_access(self, db_session: AsyncSession) -> None:
        store = EpisodicStore()
        ep = _make_episode()
        await store.save(db_session, ep)
        await store.touch(db_session, ep.id)

        loaded = await store.get(db_session, ep.id)
        assert loaded is not None
        assert loaded.access_count == 1

    async def test_preserves_links(self, db_session: AsyncSession) -> None:
        store = EpisodicStore()
        from uuid import uuid4

        target_id = uuid4()
        ep = _make_episode(links=[MemoryLink(target_id=target_id, relation="follows")])
        await store.save(db_session, ep)

        loaded = await store.get(db_session, ep.id)
        assert loaded is not None
        assert len(loaded.links) == 1
        assert loaded.links[0].target_id == target_id
        assert loaded.links[0].relation == "follows"

    async def test_vector_search(self, db_session: AsyncSession) -> None:
        store = EpisodicStore()
        dim = 1536
        emb_close = [1.0] + [0.0] * (dim - 1)
        emb_far = [0.0] * (dim - 1) + [1.0]

        await store.save(db_session, _make_episode("close", embedding=emb_close))
        await store.save(db_session, _make_episode("far", embedding=emb_far))

        results = await store.search_by_vector(db_session, emb_close, limit=2)
        assert len(results) == 2
        assert results[0][0].summary == "close"
