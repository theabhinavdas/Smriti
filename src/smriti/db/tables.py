"""SQLAlchemy table definitions for the unified memory store.

Two tables cover all four memory tiers:
- `memories`: stores Tier 2 (working), Tier 3 (episodic), and Tier 4 (semantic) records
- `edges`: stores graph relationships (Tier 4 knowledge graph + cross-tier links)

Tier 1 (buffer) is in-memory only and never touches the database.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Float,
    Index,
    Integer,
    Text,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MemoriesTable(Base):
    __tablename__ = "memories"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    tier: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(1536), nullable=True)
    facts = mapped_column(JSONB, nullable=True)
    metadata_ = mapped_column("metadata", JSONB, nullable=True)
    importance: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    access_count: Mapped[int] = mapped_column(Integer, default=0)

    __table_args__ = (
        Index("idx_memories_time", "created_at"),
        Index(
            "idx_memories_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class EdgesTable(Base):
    __tablename__ = "edges"

    source_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True
    )
    target_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True
    )
    relation: Mapped[str] = mapped_column(Text, primary_key=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    metadata_ = mapped_column("metadata", JSONB, nullable=True)

    __table_args__ = (
        Index("idx_edges_source", "source_id"),
        Index("idx_edges_target", "target_id"),
        Index("idx_edges_relation", "relation"),
    )
