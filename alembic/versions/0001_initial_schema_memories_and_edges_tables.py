"""initial schema: memories and edges tables

Revision ID: 0001
Revises:
Create Date: 2026-03-09 19:55:45.817239

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "memories",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("tier", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=True),
        sa.Column("facts", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.Column("importance", sa.Float(), server_default="0.5"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("accessed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("access_count", sa.Integer(), server_default="0"),
    )

    op.create_index("idx_memories_tier", "memories", ["tier"])
    op.create_index("idx_memories_time", "memories", ["created_at"])
    op.create_index(
        "idx_memories_embedding",
        "memories",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )

    op.create_table(
        "edges",
        sa.Column("source_id", sa.Uuid(), nullable=False),
        sa.Column("target_id", sa.Uuid(), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("weight", sa.Float(), server_default="1.0"),
        sa.Column("metadata", sa.dialects.postgresql.JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("source_id", "target_id", "relation"),
    )

    op.create_index("idx_edges_source", "edges", ["source_id"])
    op.create_index("idx_edges_target", "edges", ["target_id"])
    op.create_index("idx_edges_relation", "edges", ["relation"])


def downgrade() -> None:
    op.drop_table("edges")
    op.drop_table("memories")
    op.execute("DROP EXTENSION IF EXISTS vector")
