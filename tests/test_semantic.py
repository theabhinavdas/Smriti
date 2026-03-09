"""Integration tests for Tier 4: Semantic Store (requires running Postgres)."""

from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from smriti.memory.semantic import SemanticStore
from smriti.models.memory import SemanticEdge, SemanticNode

pytestmark = pytest.mark.asyncio


def _node(label: str, node_type: str = "concept", **kwargs) -> SemanticNode:
    return SemanticNode(label=label, node_type=node_type, **kwargs)


class TestSemanticStore:
    async def test_save_and_get_node(self, db_session: AsyncSession) -> None:
        store = SemanticStore()
        node = _node("TypeScript", "language", confidence=0.92)
        await store.save_node(db_session, node)

        loaded = await store.get_node(db_session, node.id)
        assert loaded is not None
        assert loaded.label == "TypeScript"
        assert loaded.node_type == "language"
        assert loaded.confidence == pytest.approx(0.92)

    async def test_get_missing_node_returns_none(self, db_session: AsyncSession) -> None:
        store = SemanticStore()
        assert await store.get_node(db_session, uuid4()) is None

    async def test_list_nodes(self, db_session: AsyncSession) -> None:
        store = SemanticStore()
        for lang in ("Python", "Rust", "Go"):
            await store.save_node(db_session, _node(lang, "language"))

        nodes = await store.list_nodes(db_session, limit=10)
        assert len(nodes) == 3
        labels = {n.label for n in nodes}
        assert labels == {"Python", "Rust", "Go"}

    async def test_save_and_query_edges(self, db_session: AsyncSession) -> None:
        store = SemanticStore()
        user = _node("user", "person")
        ts = _node("TypeScript", "language")
        await store.save_node(db_session, user)
        await store.save_node(db_session, ts)

        edge = SemanticEdge(
            source_id=user.id, target_id=ts.id, relation="prefers", weight=0.92
        )
        await store.save_edge(db_session, edge)

        outgoing = await store.get_outgoing(db_session, user.id)
        assert len(outgoing) == 1
        assert outgoing[0].relation == "prefers"
        assert outgoing[0].target_id == ts.id

        incoming = await store.get_incoming(db_session, ts.id)
        assert len(incoming) == 1
        assert incoming[0].source_id == user.id

    async def test_get_neighbors(self, db_session: AsyncSession) -> None:
        store = SemanticStore()
        user = _node("user", "person")
        py = _node("Python", "language")
        ts = _node("TypeScript", "language")
        await store.save_node(db_session, user)
        await store.save_node(db_session, py)
        await store.save_node(db_session, ts)

        await store.save_edge(
            db_session,
            SemanticEdge(source_id=user.id, target_id=py.id, relation="skilled_in"),
        )
        await store.save_edge(
            db_session,
            SemanticEdge(source_id=user.id, target_id=ts.id, relation="prefers"),
        )

        neighbors = await store.get_neighbors(db_session, user.id)
        assert len(neighbors) == 2
        neighbor_labels = {n.label for n, _ in neighbors}
        assert neighbor_labels == {"Python", "TypeScript"}

    async def test_delete_node_removes_edges(self, db_session: AsyncSession) -> None:
        store = SemanticStore()
        a = _node("a", "concept")
        b = _node("b", "concept")
        await store.save_node(db_session, a)
        await store.save_node(db_session, b)

        await store.save_edge(
            db_session,
            SemanticEdge(source_id=a.id, target_id=b.id, relation="related"),
        )
        await store.delete_node(db_session, a.id)

        assert await store.get_node(db_session, a.id) is None
        assert await store.get_outgoing(db_session, a.id) == []

    async def test_source_episodes_roundtrip(self, db_session: AsyncSession) -> None:
        store = SemanticStore()
        ep_ids = [uuid4(), uuid4()]
        node = _node("dark mode", "preference", source_episodes=ep_ids)
        await store.save_node(db_session, node)

        loaded = await store.get_node(db_session, node.id)
        assert loaded is not None
        assert loaded.source_episodes == ep_ids
