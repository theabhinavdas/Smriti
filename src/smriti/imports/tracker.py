"""Tracks which import files have been processed to avoid re-ingestion.

Provides a Postgres-backed table and repository for recording file hashes,
status, and metadata for every file dropped into the import directory.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Integer, Text, Uuid, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column

from smriti.db.tables import Base


class ProcessedImportsTable(Base):
    __tablename__ = "processed_imports"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    file_hash: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    format: Mapped[str | None] = mapped_column(Text, nullable=True)
    events_generated: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(Text, default="completed")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class ImportTracker:
    """Repository for processed import records."""

    @staticmethod
    def hash_file(path: Path) -> str:
        """Compute SHA-256 hex digest of a file's contents."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    async def is_processed(self, session: AsyncSession, file_hash: str) -> bool:
        """Return True if a file with this hash has already been successfully processed."""
        stmt = select(ProcessedImportsTable.id).where(
            ProcessedImportsTable.file_hash == file_hash,
            ProcessedImportsTable.status == "completed",
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none() is not None

    async def record(
        self,
        session: AsyncSession,
        *,
        file_path: str,
        file_hash: str,
        file_size: int,
        fmt: str | None = None,
        events_generated: int = 0,
        status: str = "completed",
        error: str | None = None,
    ) -> ProcessedImportsTable:
        """Insert a processed-import record."""
        row = ProcessedImportsTable(
            id=uuid.uuid4(),
            file_path=file_path,
            file_hash=file_hash,
            file_size=file_size,
            format=fmt,
            events_generated=events_generated,
            status=status,
            error=error,
        )
        session.add(row)
        await session.flush()
        return row

    async def mark_failed(
        self, session: AsyncSession, file_hash: str, error: str
    ) -> None:
        """Update an existing record to failed status."""
        stmt = select(ProcessedImportsTable).where(
            ProcessedImportsTable.file_hash == file_hash
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if row is not None:
            row.status = "failed"
            row.error = error

    async def list_imports(
        self,
        session: AsyncSession,
        *,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ProcessedImportsTable]:
        """List processed imports, optionally filtered by status."""
        stmt = (
            select(ProcessedImportsTable)
            .order_by(ProcessedImportsTable.processed_at.desc())
            .limit(limit)
            .offset(offset)
        )
        if status is not None:
            stmt = stmt.where(ProcessedImportsTable.status == status)
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_by_hash(
        self, session: AsyncSession, file_hash: str
    ) -> ProcessedImportsTable | None:
        """Look up a single import record by file hash."""
        stmt = select(ProcessedImportsTable).where(
            ProcessedImportsTable.file_hash == file_hash
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def delete_by_hash(self, session: AsyncSession, file_hash: str) -> bool:
        """Delete a record by hash so the file can be re-processed. Returns True if deleted."""
        row = await self.get_by_hash(session, file_hash)
        if row is None:
            return False
        await session.delete(row)
        await session.flush()
        return True
