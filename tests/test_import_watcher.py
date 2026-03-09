"""Tests for the ImportWatcher (directory scanning + dedup + event publishing)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from smriti.imports.watcher import ImportWatcher, IGNORED_NAMES
from smriti.models.events import SourceEvent

pytestmark = pytest.mark.asyncio


class _FakeTxn:
    async def __aenter__(self):
        return None

    async def __aexit__(self, *args):
        pass


class _FakeSessionFactory:
    """Mimics async_sessionmaker: calling it returns an async context manager."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def __call__(self):
        return self

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *args):
        pass


def _make_watcher(
    tmp_path: Path, db_session: AsyncSession, *, published: list[SourceEvent]
) -> ImportWatcher:
    """Build a watcher with a mock event bus and a real DB session."""
    mock_bus = AsyncMock()

    async def capture_publish(event: SourceEvent) -> str:
        published.append(event)
        return "fake-entry-id"

    mock_bus.publish = capture_publish

    db_session.begin = lambda: _FakeTxn()

    return ImportWatcher(
        watch_dir=tmp_path / "imports",
        event_bus=mock_bus,
        session_factory=_FakeSessionFactory(db_session),  # type: ignore[arg-type]
        poll_interval=0,
    )


class TestImportWatcher:
    async def test_processes_txt_file(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        published: list[SourceEvent] = []
        watcher = _make_watcher(tmp_path, db_session, published=published)
        import_dir = tmp_path / "imports"
        import_dir.mkdir()
        (import_dir / "notes.txt").write_text("Important note about testing.")

        await watcher.run(max_iterations=1)
        assert len(published) == 1
        assert "Important note" in published[0].raw_content

    async def test_processes_markdown_file(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        published: list[SourceEvent] = []
        watcher = _make_watcher(tmp_path, db_session, published=published)
        import_dir = tmp_path / "imports"
        import_dir.mkdir()
        (import_dir / "design.md").write_text("# Overview\n\nDesign doc content.")

        await watcher.run(max_iterations=1)
        assert len(published) == 1
        assert published[0].source == "obsidian"

    async def test_processes_chatgpt_export(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        published: list[SourceEvent] = []
        watcher = _make_watcher(tmp_path, db_session, published=published)
        import_dir = tmp_path / "imports"
        import_dir.mkdir()

        conv = {
            "id": "c1", "title": "Test", "create_time": 1700000000,
            "mapping": {
                "root": {"id": "root", "parent": None, "message": None},
                "m1": {
                    "id": "m1", "parent": "root",
                    "message": {
                        "author": {"role": "user"},
                        "content": {"content_type": "text", "parts": ["Hello"]},
                        "metadata": {},
                    },
                },
                "m2": {
                    "id": "m2", "parent": "m1",
                    "message": {
                        "author": {"role": "assistant"},
                        "content": {"content_type": "text", "parts": ["Hi!"]},
                        "metadata": {"model_slug": "gpt-4"},
                    },
                },
            },
        }
        (import_dir / "conversations.json").write_text(json.dumps([conv]))

        await watcher.run(max_iterations=1)
        assert len(published) == 1
        assert published[0].source == "chatgpt"

    async def test_dedup_skips_already_processed(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        published: list[SourceEvent] = []
        watcher = _make_watcher(tmp_path, db_session, published=published)
        import_dir = tmp_path / "imports"
        import_dir.mkdir()
        (import_dir / "notes.txt").write_text("Same content.")

        await watcher.run(max_iterations=1)
        assert len(published) == 1

        await watcher.run(max_iterations=1)
        assert len(published) == 1  # no new events — dedup kicked in

    async def test_ignores_dotfiles(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        published: list[SourceEvent] = []
        watcher = _make_watcher(tmp_path, db_session, published=published)
        import_dir = tmp_path / "imports"
        import_dir.mkdir()
        (import_dir / ".hidden.txt").write_text("secret")
        (import_dir / ".DS_Store").write_text("junk")

        await watcher.run(max_iterations=1)
        assert len(published) == 0

    async def test_ignores_unsupported_formats(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        published: list[SourceEvent] = []
        watcher = _make_watcher(tmp_path, db_session, published=published)
        import_dir = tmp_path / "imports"
        import_dir.mkdir()
        (import_dir / "image.png").write_bytes(b"\x89PNG\r\n")

        await watcher.run(max_iterations=1)
        assert len(published) == 0

    async def test_recursive_scan(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        published: list[SourceEvent] = []
        watcher = _make_watcher(tmp_path, db_session, published=published)
        import_dir = tmp_path / "imports"
        sub = import_dir / "vault" / "daily"
        sub.mkdir(parents=True)
        (sub / "2026-03-09.md").write_text("# Daily Note\n\nDid some work.")

        await watcher.run(max_iterations=1)
        assert len(published) == 1
        assert published[0].source == "obsidian"

    async def test_creates_watch_dir_if_missing(
        self, tmp_path: Path, db_session: AsyncSession
    ) -> None:
        published: list[SourceEvent] = []
        watcher = _make_watcher(tmp_path, db_session, published=published)
        assert not (tmp_path / "imports").exists()

        await watcher.run(max_iterations=1)
        assert (tmp_path / "imports").exists()
