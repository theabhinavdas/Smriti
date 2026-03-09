"""Integration tests for Tier 2: Working Memory Store (requires running Valkey)."""

from __future__ import annotations

import pytest

from smriti.config import ValkeyConfig
from smriti.memory.working import WorkingMemoryStore
from smriti.models.memory import Decision, EntityInfo, Goal, WorkingMemory

pytestmark = pytest.mark.asyncio


@pytest.fixture
async def store():
    config = ValkeyConfig()
    s = await WorkingMemoryStore.from_config(config)
    yield s
    await s._client.delete("smriti:working:test-sess")
    await s._client.delete("smriti:working:other-sess")
    await s.close()


def _make_wm(session_id: str = "test-sess") -> WorkingMemory:
    return WorkingMemory(
        session_id=session_id,
        summary="Debugging auth service",
        entities={"auth_svc": EntityInfo(entity_type="component", status="debugging")},
        active_goals=[Goal(description="fix CORS error")],
        decisions=[Decision(choice="use middleware", reason="simpler")],
        token_count=150,
    )


class TestWorkingMemoryStore:
    async def test_save_and_load(self, store: WorkingMemoryStore) -> None:
        wm = _make_wm()
        await store.save(wm)
        loaded = await store.load("test-sess")
        assert loaded is not None
        assert loaded.session_id == "test-sess"
        assert loaded.summary == "Debugging auth service"
        assert "auth_svc" in loaded.entities
        assert loaded.active_goals[0].description == "fix CORS error"
        assert loaded.decisions[0].choice == "use middleware"
        assert loaded.token_count == 150

    async def test_load_missing_returns_none(self, store: WorkingMemoryStore) -> None:
        result = await store.load("nonexistent-session")
        assert result is None

    async def test_delete(self, store: WorkingMemoryStore) -> None:
        await store.save(_make_wm())
        assert await store.exists("test-sess")
        await store.delete("test-sess")
        assert not await store.exists("test-sess")

    async def test_overwrite_on_save(self, store: WorkingMemoryStore) -> None:
        wm = _make_wm()
        await store.save(wm)

        wm.summary = "Updated summary"
        wm.active_goals.append(Goal(description="deploy to staging"))
        await store.save(wm)

        loaded = await store.load("test-sess")
        assert loaded is not None
        assert loaded.summary == "Updated summary"
        assert len(loaded.active_goals) == 2

    async def test_independent_sessions(self, store: WorkingMemoryStore) -> None:
        await store.save(_make_wm("test-sess"))
        await store.save(WorkingMemory(session_id="other-sess", summary="other"))

        a = await store.load("test-sess")
        b = await store.load("other-sess")
        assert a is not None and b is not None
        assert a.summary != b.summary
