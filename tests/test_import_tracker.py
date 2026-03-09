"""Tests for the import tracker (ProcessedImportsTable + ImportTracker repository)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from smriti.imports.tracker import ImportTracker

class TestHashFile:
    def test_consistent_hash(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        tracker = ImportTracker()
        h1 = tracker.hash_file(f)
        h2 = tracker.hash_file(f)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex digest

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("alpha")
        b.write_text("beta")
        tracker = ImportTracker()
        assert tracker.hash_file(a) != tracker.hash_file(b)


pytestmark = pytest.mark.asyncio


class TestImportTracker:
    async def test_record_and_is_processed(self, db_session: AsyncSession) -> None:
        tracker = ImportTracker()
        assert not await tracker.is_processed(db_session, "abc123")

        await tracker.record(
            db_session,
            file_path="/imports/chat.json",
            file_hash="abc123",
            file_size=1024,
            fmt="chatgpt",
            events_generated=5,
        )
        assert await tracker.is_processed(db_session, "abc123")

    async def test_failed_record_not_considered_processed(
        self, db_session: AsyncSession
    ) -> None:
        tracker = ImportTracker()
        await tracker.record(
            db_session,
            file_path="/imports/bad.txt",
            file_hash="fail_hash",
            file_size=100,
            status="failed",
            error="parse error",
        )
        assert not await tracker.is_processed(db_session, "fail_hash")

    async def test_mark_failed(self, db_session: AsyncSession) -> None:
        tracker = ImportTracker()
        await tracker.record(
            db_session,
            file_path="/imports/notes.md",
            file_hash="hash_ok",
            file_size=512,
            status="completed",
        )
        assert await tracker.is_processed(db_session, "hash_ok")

        await tracker.mark_failed(db_session, "hash_ok", "oops")
        assert not await tracker.is_processed(db_session, "hash_ok")

        row = await tracker.get_by_hash(db_session, "hash_ok")
        assert row is not None
        assert row.status == "failed"
        assert row.error == "oops"

    async def test_list_imports(self, db_session: AsyncSession) -> None:
        tracker = ImportTracker()
        for i in range(3):
            await tracker.record(
                db_session,
                file_path=f"/imports/file{i}.txt",
                file_hash=f"hash_{i}",
                file_size=100 + i,
            )
        await tracker.record(
            db_session,
            file_path="/imports/bad.txt",
            file_hash="hash_bad",
            file_size=50,
            status="failed",
            error="bad",
        )

        all_imports = await tracker.list_imports(db_session)
        assert len(all_imports) == 4

        completed = await tracker.list_imports(db_session, status="completed")
        assert len(completed) == 3

        failed = await tracker.list_imports(db_session, status="failed")
        assert len(failed) == 1
        assert failed[0].error == "bad"

    async def test_get_by_hash(self, db_session: AsyncSession) -> None:
        tracker = ImportTracker()
        await tracker.record(
            db_session,
            file_path="/imports/notes.md",
            file_hash="unique_hash",
            file_size=256,
            fmt="markdown",
        )
        row = await tracker.get_by_hash(db_session, "unique_hash")
        assert row is not None
        assert row.file_path == "/imports/notes.md"
        assert row.format == "markdown"

        missing = await tracker.get_by_hash(db_session, "nonexistent")
        assert missing is None

    async def test_delete_by_hash_allows_reprocessing(
        self, db_session: AsyncSession
    ) -> None:
        tracker = ImportTracker()
        await tracker.record(
            db_session,
            file_path="/imports/redo.txt",
            file_hash="redo_hash",
            file_size=200,
        )
        assert await tracker.is_processed(db_session, "redo_hash")

        deleted = await tracker.delete_by_hash(db_session, "redo_hash")
        assert deleted is True
        assert not await tracker.is_processed(db_session, "redo_hash")

        not_found = await tracker.delete_by_hash(db_session, "redo_hash")
        assert not_found is False
