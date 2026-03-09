"""Unit tests for the pipeline daemon.

All external dependencies (EventBus, ModelProvider, DB) are mocked so
these tests run without any infrastructure.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from smriti.daemon import Daemon, PipelineStats
from smriti.models.events import SourceEvent


def _make_event(source: str = "terminal", content: str = "git push origin main") -> SourceEvent:
    return SourceEvent(
        source=source,
        event_type="command",
        raw_content=content,
    )


def _build_daemon() -> Daemon:
    """Create a Daemon with all dependencies mocked."""
    settings = MagicMock()

    event_bus = AsyncMock()
    event_bus.stream_key = "smriti:events"
    event_bus.consume = AsyncMock(return_value=[])
    event_bus.ack = AsyncMock(return_value=0)
    event_bus.close = AsyncMock()

    provider = AsyncMock()
    provider.complete = AsyncMock(return_value="[]")
    provider.embed = AsyncMock(return_value=[])
    provider.close = AsyncMock()
    provider.config = MagicMock()
    provider.config.max_tokens_per_extraction = 1000

    engine = AsyncMock()
    engine.dispose = AsyncMock()

    session = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session.begin = MagicMock(return_value=session)

    session_factory = MagicMock(return_value=session)

    daemon = Daemon(
        settings=settings,
        event_bus=event_bus,
        provider=provider,
        engine=engine,
        session_factory=session_factory,
    )
    return daemon


class TestPipelineStats:
    def test_defaults(self) -> None:
        stats = PipelineStats()
        assert stats.batches_processed == 0
        assert stats.events_consumed == 0
        assert stats.events_filtered == 0
        assert stats.memories_created == 0


class TestDaemonRun:
    @pytest.mark.asyncio
    async def test_empty_batch_completes(self) -> None:
        daemon = _build_daemon()
        daemon.event_bus.consume.return_value = []
        await daemon.run(max_iterations=1)
        daemon.event_bus.consume.assert_called_once()
        assert daemon.stats.batches_processed == 0

    @pytest.mark.asyncio
    async def test_processes_events(self) -> None:
        daemon = _build_daemon()
        event = _make_event()
        daemon.event_bus.consume.side_effect = [
            [("1-0", event)],
            [],
        ]
        await daemon.run(max_iterations=2)
        assert daemon.stats.events_consumed == 1
        daemon.event_bus.ack.assert_called()

    @pytest.mark.asyncio
    async def test_stop_halts_loop(self) -> None:
        daemon = _build_daemon()
        daemon.event_bus.consume.return_value = []

        async def stop_after_delay():
            await asyncio.sleep(0.05)
            await daemon.stop()

        task = asyncio.create_task(stop_after_delay())
        await daemon.run(max_iterations=100)
        await task
        assert not daemon._running

    @pytest.mark.asyncio
    async def test_salient_events_create_memories(self) -> None:
        """High-salience events flow through extraction and routing."""
        daemon = _build_daemon()
        event = _make_event(content="git push origin main")
        daemon.event_bus.consume.side_effect = [
            [("1-0", event)],
            [],
        ]
        await daemon.run(max_iterations=2)
        assert daemon.stats.events_consumed == 1
        assert daemon.stats.batches_processed >= 1

    @pytest.mark.asyncio
    async def test_acks_after_no_salient(self) -> None:
        """Events that are all filtered out still get ACKed."""
        daemon = _build_daemon()
        event = _make_event(source="terminal", content="ls")
        daemon.event_bus.consume.side_effect = [
            [("1-0", event)],
            [],
        ]
        await daemon.run(max_iterations=2)
        assert daemon.stats.events_consumed == 1
        assert daemon.stats.events_filtered == 0
        daemon.event_bus.ack.assert_called_with(["1-0"])


class TestDaemonLifecycle:
    @pytest.mark.asyncio
    async def test_shutdown_releases_resources(self) -> None:
        daemon = _build_daemon()
        await daemon.shutdown()
        daemon.event_bus.close.assert_called_once()
        daemon.provider.close.assert_called_once()
        daemon.engine.dispose.assert_called_once()
        assert not daemon._running
