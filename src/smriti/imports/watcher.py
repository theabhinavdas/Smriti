"""Import directory watcher: polls a directory for new files and ingests them.

Scans recursively, detects file format via parser registry, checks the
import tracker to skip already-processed files, and pushes SourceEvents
into the event bus for the standard ingestion pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from smriti.event_bus import EventBus
from smriti.imports.parsers.base import ImportParser
from smriti.imports.parsers.chatgpt import ChatGPTParser
from smriti.imports.parsers.markdown import MarkdownParser
from smriti.imports.parsers.plaintext import PlainTextParser
from smriti.imports.tracker import ImportTracker

logger = logging.getLogger(__name__)

IGNORED_NAMES = frozenset({
    ".DS_Store", "Thumbs.db", "__MACOSX", ".git", ".gitignore",
    ".smriti-imports", "desktop.ini",
})


class ImportWatcher:
    """Polls a directory for new files and ingests them via the standard pipeline."""

    def __init__(
        self,
        watch_dir: Path,
        event_bus: EventBus,
        session_factory: async_sessionmaker[AsyncSession],
        *,
        poll_interval: float = 10.0,
        parsers: list[ImportParser] | None = None,
    ) -> None:
        self._watch_dir = watch_dir
        self._event_bus = event_bus
        self._session_factory = session_factory
        self._poll_interval = poll_interval
        self._tracker = ImportTracker()
        self._parsers: list[ImportParser] = parsers or [
            ChatGPTParser(),
            MarkdownParser(),
            PlainTextParser(),
        ]
        self._running = False

    async def run(self, *, max_iterations: int | None = None) -> None:
        """Start the polling loop. Runs until stopped or max_iterations reached."""
        self._running = True
        self._watch_dir.mkdir(parents=True, exist_ok=True)
        logger.info("ImportWatcher started — watching %s", self._watch_dir)

        iteration = 0
        try:
            while self._running:
                if max_iterations is not None and iteration >= max_iterations:
                    break
                await self._scan()
                iteration += 1
                if self._running and (max_iterations is None or iteration < max_iterations):
                    await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            logger.info("ImportWatcher cancelled")
        finally:
            self._running = False

    async def stop(self) -> None:
        self._running = False

    async def _scan(self) -> None:
        """Walk the import directory and process any new files."""
        files = self._collect_files()
        if not files:
            return

        for file_path in files:
            try:
                await self._process_file(file_path)
            except Exception:
                logger.warning("Failed to process %s", file_path, exc_info=True)

    def _collect_files(self) -> list[Path]:
        """Recursively find importable files, filtering out noise."""
        if not self._watch_dir.exists():
            return []

        files: list[Path] = []
        for path in self._watch_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.name in IGNORED_NAMES:
                continue
            if any(part in IGNORED_NAMES for part in path.parts):
                continue
            files.append(path)

        return sorted(files)

    async def _process_file(self, path: Path) -> None:
        """Check dedup, parse, and publish events for a single file."""
        file_hash = self._tracker.hash_file(path)
        file_size = path.stat().st_size

        async with self._session_factory() as session:
            async with session.begin():
                if await self._tracker.is_processed(session, file_hash):
                    return

                parser = self._find_parser(path)
                if parser is None:
                    logger.debug("No parser found for %s, skipping", path.name)
                    return

                try:
                    events = parser.parse(path)
                except Exception as exc:
                    await self._tracker.record(
                        session,
                        file_path=str(path),
                        file_hash=file_hash,
                        file_size=file_size,
                        fmt=parser.name,
                        status="failed",
                        error=str(exc),
                    )
                    logger.warning("Parser %s failed on %s: %s", parser.name, path, exc)
                    return

                for event in events:
                    await self._event_bus.publish(event)

                await self._tracker.record(
                    session,
                    file_path=str(path),
                    file_hash=file_hash,
                    file_size=file_size,
                    fmt=parser.name,
                    events_generated=len(events),
                )

                logger.info(
                    "Imported %s (%s): %d events", path.name, parser.name, len(events)
                )

    def _find_parser(self, path: Path) -> ImportParser | None:
        """Return the first parser that can handle this file."""
        for parser in self._parsers:
            try:
                if parser.can_parse(path):
                    return parser
            except Exception:
                continue
        return None
