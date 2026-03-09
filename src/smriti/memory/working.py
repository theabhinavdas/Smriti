"""Tier 2: Working Memory Store -- session-scoped facts persisted to Valkey.

Read on every LLM call, updated incrementally after each turn.
On session end the caller should persist to Postgres as an episodic memory.
"""

from __future__ import annotations

import valkey.asyncio as valkey_async

from smriti.config import ValkeyConfig
from smriti.models.memory import WorkingMemory

_KEY_PREFIX = "smriti:working:"


class WorkingMemoryStore:
    """Valkey-backed store for session working memory."""

    def __init__(self, client: valkey_async.Valkey, ttl_seconds: int = 86400) -> None:
        self._client = client
        self._ttl = ttl_seconds

    @classmethod
    async def from_config(cls, config: ValkeyConfig | None = None) -> WorkingMemoryStore:
        if config is None:
            config = ValkeyConfig()
        client = valkey_async.Valkey(
            host=config.host,
            port=config.port,
            password=config.password or None,
            decode_responses=True,
        )
        return cls(client=client, ttl_seconds=config.working_memory_ttl_seconds)

    def _key(self, session_id: str) -> str:
        return f"{_KEY_PREFIX}{session_id}"

    async def save(self, wm: WorkingMemory) -> None:
        """Serialize and store working memory with TTL."""
        key = self._key(wm.session_id)
        await self._client.set(key, wm.model_dump_json(), ex=self._ttl)

    async def load(self, session_id: str) -> WorkingMemory | None:
        """Load working memory for a session. Returns None if expired or absent."""
        raw = await self._client.get(self._key(session_id))
        if raw is None:
            return None
        return WorkingMemory.model_validate_json(raw)

    async def delete(self, session_id: str) -> None:
        """Remove working memory for a session (e.g. after consolidation)."""
        await self._client.delete(self._key(session_id))

    async def exists(self, session_id: str) -> bool:
        return bool(await self._client.exists(self._key(session_id)))

    async def close(self) -> None:
        await self._client.aclose()
