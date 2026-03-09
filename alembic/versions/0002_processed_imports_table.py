"""processed_imports table for import deduplication tracking

Revision ID: 0002
Revises: 0001
Create Date: 2026-03-09 22:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, Sequence[str], None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "processed_imports",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_hash", sa.Text(), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("format", sa.Text(), nullable=True),
        sa.Column("events_generated", sa.Integer(), server_default="0"),
        sa.Column("status", sa.Text(), server_default="'completed'"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("file_hash", name="uq_processed_imports_hash"),
    )


def downgrade() -> None:
    op.drop_table("processed_imports")
