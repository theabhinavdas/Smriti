"""Pipeline daemon (memoryd): consumes events and writes memories.

The daemon runs a continuous async loop:
  EventBus.consume → SalienceFilter → MemoryExtractor → TierRouter → ACK

It manages the lifecycle of all shared resources (DB engine, Valkey
client, HTTP client) and shuts down gracefully on cancellation.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

import valkey.exceptions

from smriti.config import Settings, load_settings
from smriti.db.engine import create_engine, create_session_factory
from smriti.event_bus import EventBus
from smriti.imports.watcher import ImportWatcher
from smriti.ingestion.extractor import MemoryExtractor
from smriti.ingestion.router import TierRouter
from smriti.ingestion.salience import SalienceFilter
from smriti.provider import ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Counters for the current daemon lifetime."""

    batches_processed: int = 0
    events_consumed: int = 0
    events_filtered: int = 0
    memories_created: int = 0


@dataclass
class Daemon:
    """Async pipeline daemon that wires all components together."""

    settings: Settings
    event_bus: EventBus
    provider: ModelProvider
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]

    salience: SalienceFilter = field(init=False)
    extractor: MemoryExtractor = field(init=False)
    router: TierRouter = field(init=False)
    import_watcher: ImportWatcher | None = field(init=False, default=None)
    stats: PipelineStats = field(default_factory=PipelineStats)

    _running: bool = field(default=False, repr=False)
    _import_task: asyncio.Task[None] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.salience = SalienceFilter(
            provider=self.provider,
            ignored_projects=self.settings.daemon.ignored_projects,
        )
        self.extractor = MemoryExtractor(provider=self.provider)
        self.router = TierRouter()

        if self.settings.imports.enabled:
            watch_dir = Path(self.settings.imports.watch_directory).expanduser()
            self.import_watcher = ImportWatcher(
                watch_dir=watch_dir,
                event_bus=self.event_bus,
                session_factory=self.session_factory,
                poll_interval=self.settings.imports.poll_interval_seconds,
            )

    @classmethod
    async def from_settings(cls, settings: Settings | None = None) -> Daemon:
        """Bootstrap all dependencies from config."""
        settings = settings or load_settings()
        event_bus = await EventBus.from_config(settings.valkey)
        provider = ModelProvider(config=settings.models)
        engine = create_engine(settings.postgres)
        session_factory = create_session_factory(engine)
        return cls(
            settings=settings,
            event_bus=event_bus,
            provider=provider,
            engine=engine,
            session_factory=session_factory,
        )

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def run(self, *, max_iterations: int | None = None) -> None:
        """Start the consume-process loop.

        Args:
            max_iterations: Stop after N iterations (None = run until cancelled).
        """
        self._running = True
        logger.info("memoryd started — consuming from %s", self.event_bus.stream_key)

        if self.import_watcher is not None:
            self._import_task = asyncio.create_task(self.import_watcher.run())
            logger.info("ImportWatcher started as background task")

        iteration = 0
        try:
            while self._running:
                if max_iterations is not None and iteration >= max_iterations:
                    break
                await self._process_batch()
                iteration += 1
        except asyncio.CancelledError:
            logger.info("memoryd cancelled, shutting down")
        except valkey.exceptions.ConnectionError:
            if self._running:
                raise
            logger.info("memoryd connection closed during shutdown")
        finally:
            self._running = False
            if self._import_task is not None and not self._import_task.done():
                self._import_task.cancel()
                try:
                    await self._import_task
                except asyncio.CancelledError:
                    pass

    async def stop(self) -> None:
        """Signal the run loop to stop after the current batch."""
        self._running = False

    async def _process_batch(self) -> None:
        """Consume one batch from the event bus and push through the pipeline."""
        pairs = await self.event_bus.consume(batch_size=50, block_ms=500)
        if not pairs:
            return

        entry_ids = [eid for eid, _ in pairs]
        events = [ev for _, ev in pairs]
        self.stats.events_consumed += len(events)

        salient = await self.salience.filter(events)
        self.stats.events_filtered += len(salient)

        if not salient:
            await self.event_bus.ack(entry_ids)
            self.stats.batches_processed += 1
            return

        extracted = await self.extractor.extract(salient)

        async with self.session_factory() as session:
            async with session.begin():
                counts = await self.router.route(session, extracted)
                self.stats.memories_created += sum(counts.values())

        await self.event_bus.ack(entry_ids)
        self.stats.batches_processed += 1

        logger.info(
            "Batch %d: %d events → %d salient → %d memories",
            self.stats.batches_processed,
            len(events),
            len(salient),
            sum(counts.values()),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Release all resources."""
        await self.stop()
        if self.import_watcher is not None:
            await self.import_watcher.stop()
        if self._import_task is not None and not self._import_task.done():
            self._import_task.cancel()
            try:
                await self._import_task
            except asyncio.CancelledError:
                pass
        await self.event_bus.close()
        await self.provider.close()
        await self.engine.dispose()
        logger.info("memoryd shut down")
