"""Application configuration loaded from environment and config files."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class PostgresConfig(BaseSettings):
    model_config = {"env_prefix": "SMRITI_PG_", "env_file": ".env", "extra": "ignore"}

    host: str = "127.0.0.1"
    port: int = 5432
    database: str = "smriti"
    user: str = "smriti"
    password: str = "smriti"
    pool_min: int = 2
    pool_max: int = 10

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class ValkeyConfig(BaseSettings):
    model_config = {"env_prefix": "SMRITI_VALKEY_", "env_file": ".env", "extra": "ignore"}

    host: str = "127.0.0.1"
    port: int = 6379
    password: str = ""
    event_stream: str = "smriti:events"
    consumer_group: str = "ingestion"
    cache_ttl_seconds: int = 3600
    working_memory_ttl_seconds: int = 86400


class ModelConfig(BaseSettings):
    model_config = {"env_prefix": "SMRITI_MODEL_", "env_file": ".env", "extra": "ignore"}

    provider: str = "openrouter"
    api_key: str = ""
    embedding_model: str = "openai/text-embedding-3-small"
    extraction_model: str = "anthropic/claude-3.5-haiku"
    consolidation_model: str = "anthropic/claude-3.5-haiku"
    max_daily_spend_usd: float = 5.00
    max_tokens_per_extraction: int = 1000


class DaemonConfig(BaseSettings):
    model_config = {"env_prefix": "SMRITI_DAEMON_", "env_file": ".env", "extra": "ignore"}

    host: str = "0.0.0.0"
    port: int = 9898
    consolidation_interval_minutes: int = 30


class Settings(BaseSettings):
    postgres: PostgresConfig = PostgresConfig()
    valkey: ValkeyConfig = ValkeyConfig()
    models: ModelConfig = ModelConfig()
    daemon: DaemonConfig = DaemonConfig()
    log_level: str = "info"


def load_settings() -> Settings:
    return Settings()
