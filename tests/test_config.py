"""Tests for configuration loading."""

from smriti.config import PostgresConfig, Settings, ValkeyConfig, load_settings


def test_default_postgres_config() -> None:
    cfg = PostgresConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 5432
    assert cfg.database == "smriti"
    assert "asyncpg" in cfg.dsn


def test_default_valkey_config() -> None:
    cfg = ValkeyConfig()
    assert cfg.host == "127.0.0.1"
    assert cfg.port == 6379
    assert cfg.event_stream == "smriti:events"


def test_load_settings_returns_settings() -> None:
    settings = load_settings()
    assert isinstance(settings, Settings)
    assert settings.log_level == "info"
