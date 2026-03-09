"""Shared test fixtures.

Provides an async session connected to the local Docker Postgres instance.
Tests run inside a transaction that is rolled back after each test,
so the database stays clean without needing to recreate tables.
"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from smriti.config import PostgresConfig
from smriti.db.tables import Base

_pg = PostgresConfig()


@pytest.fixture
async def db_session():
    """Yield a session wrapped in a transaction that is rolled back after the test."""
    engine = create_async_engine(_pg.dsn, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with engine.connect() as conn:
        txn = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        yield session
        await session.close()
        await txn.rollback()

    await engine.dispose()
