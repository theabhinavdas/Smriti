"""Integration tests for the Valkey-backed event bus (requires running Valkey)."""

from __future__ import annotations

import pytest

from smriti.config import ValkeyConfig
from smriti.event_bus import EventBus
from smriti.models.events import ActivityContext, SourceEvent

pytestmark = pytest.mark.asyncio

TEST_STREAM = "smriti:test:events"
TEST_GROUP = "test-group"


@pytest.fixture
async def bus():
    """Create an EventBus pointed at a disposable test stream."""
    config = ValkeyConfig()
    bus = EventBus(
        client=(await EventBus.from_config(config)).client,
        stream_key=TEST_STREAM,
        group_name=TEST_GROUP,
    )
    yield bus
    await bus.client.delete(TEST_STREAM)
    await bus.close()


def _make_event(content: str = "ls -la", source: str = "terminal") -> SourceEvent:
    return SourceEvent(source=source, event_type="command_executed", raw_content=content)


class TestEventBus:
    async def test_publish_returns_entry_id(self, bus: EventBus) -> None:
        entry_id = await bus.publish(_make_event())
        assert isinstance(entry_id, str)
        assert "-" in entry_id

    async def test_stream_length_increments(self, bus: EventBus) -> None:
        assert await bus.stream_length() == 0
        await bus.publish(_make_event("echo one"))
        await bus.publish(_make_event("echo two"))
        assert await bus.stream_length() == 2

    async def test_consume_returns_published_events(self, bus: EventBus) -> None:
        event = _make_event("git commit -m 'init'")
        await bus.publish(event)

        batch = await bus.consume(batch_size=10, block_ms=500)
        assert len(batch) == 1
        entry_id, consumed = batch[0]
        assert consumed.raw_content == "git commit -m 'init'"
        assert consumed.source == "terminal"
        assert consumed.content_hash == event.content_hash

    async def test_ack_prevents_reconsumption(self, bus: EventBus) -> None:
        await bus.publish(_make_event("pip install numpy"))
        batch = await bus.consume(batch_size=10, block_ms=500)
        assert len(batch) == 1

        entry_ids = [eid for eid, _ in batch]
        acked = await bus.ack(entry_ids)
        assert acked == 1

        second_batch = await bus.consume(batch_size=10, block_ms=500)
        assert len(second_batch) == 0

    async def test_preserves_activity_context(self, bus: EventBus) -> None:
        event = SourceEvent(
            source="cursor",
            event_type="file_edited",
            raw_content="diff --git a/foo.py",
            activity_context=ActivityContext(project="smriti", session_id="sess-42"),
        )
        await bus.publish(event)

        batch = await bus.consume(batch_size=1, block_ms=500)
        _, consumed = batch[0]
        assert consumed.activity_context.project == "smriti"
        assert consumed.activity_context.session_id == "sess-42"

    async def test_ack_empty_list_is_noop(self, bus: EventBus) -> None:
        result = await bus.ack([])
        assert result == 0
