"""Alembic environment configuration.

Reads the database URL from SMRITI_PG_* env vars (via our config module)
when available, falling back to alembic.ini. Registers our SQLAlchemy
metadata so `alembic revision --autogenerate` can diff the schema.
"""

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context
from smriti.db.tables import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

try:
    from smriti.config import PostgresConfig

    pg = PostgresConfig()
    sync_url = (
        f"postgresql://{pg.user}:{pg.password}@{pg.host}:{pg.port}/{pg.database}"
    )
    config.set_main_option("sqlalchemy.url", sync_url)
except Exception:
    pass


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
