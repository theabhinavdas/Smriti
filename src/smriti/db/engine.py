"""Async database engine and session factory."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from smriti.config import PostgresConfig


def create_engine(config: PostgresConfig | None = None) -> AsyncEngine:
    """Create an async SQLAlchemy engine from config."""
    if config is None:
        config = PostgresConfig()
    return create_async_engine(
        config.dsn,
        pool_size=config.pool_max,
        pool_pre_ping=True,
        echo=False,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create a session factory bound to the given engine."""
    return async_sessionmaker(engine, expire_on_commit=False)
