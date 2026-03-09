"""Valkey Streams-based event bus for collector-to-ingestion communication.

Collectors publish SourceEvents via XADD. The ingestion pipeline consumes
them via XREADGROUP with exactly-once semantics (consumer groups + XACK).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import valkey.asyncio as valkey_async

from smriti.config import ValkeyConfig
from smriti.models.events import SourceEvent

logger = logging.getLogger(__name__)

_EVENT_PAYLOAD_KEY = "payload"


@dataclass
class EventBus:
    """Async wrapper around a Valkey Stream for SourceEvent pub/sub."""

    client: valkey_async.Valkey
    stream_key: str = "smriti:events"
    group_name: str = "ingestion"
    consumer_name: str = "worker-1"
    _group_ensured: bool = field(default=False, repr=False)

    @classmethod
    async def from_config(cls, config: ValkeyConfig | None = None) -> EventBus:
        if config is None:
            config = ValkeyConfig()
        client = valkey_async.Valkey(
            host=config.host,
            port=config.port,
            password=config.password or None,
            decode_responses=True,
        )
        bus = cls(
            client=client,
            stream_key=config.event_stream,
            group_name=config.consumer_group,
        )
        return bus

    async def ensure_group(self) -> None:
        """Create the consumer group if it doesn't exist yet."""
        if self._group_ensured:
            return
        try:
            await self.client.xgroup_create(
                self.stream_key, self.group_name, id="0", mkstream=True
            )
        except valkey_async.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._group_ensured = True

    async def publish(self, event: SourceEvent) -> str:
        """Serialize and XADD an event. Returns the stream entry ID."""
        payload = event.model_dump_json()
        entry_id: str = await self.client.xadd(
            self.stream_key, {_EVENT_PAYLOAD_KEY: payload}
        )
        return entry_id

    async def consume(self, batch_size: int = 100, block_ms: int = 1000) -> list[tuple[str, SourceEvent]]:
        """XREADGROUP: pull unacknowledged events. Returns (entry_id, event) pairs."""
        await self.ensure_group()
        response = await self.client.xreadgroup(
            self.group_name,
            self.consumer_name,
            {self.stream_key: ">"},
            count=batch_size,
            block=block_ms,
        )
        results: list[tuple[str, SourceEvent]] = []
        if not response:
            return results
        for _stream_name, entries in response:
            for entry_id, fields in entries:
                try:
                    event = SourceEvent.model_validate_json(fields[_EVENT_PAYLOAD_KEY])
                    results.append((entry_id, event))
                except Exception:
                    logger.warning("Skipping malformed event %s", entry_id, exc_info=True)
        return results

    async def ack(self, entry_ids: list[str]) -> int:
        """XACK: mark events as processed. Returns number acknowledged."""
        if not entry_ids:
            return 0
        return await self.client.xack(self.stream_key, self.group_name, *entry_ids)

    async def stream_length(self) -> int:
        """XLEN: number of entries in the stream."""
        return await self.client.xlen(self.stream_key)

    async def close(self) -> None:
        await self.client.aclose()
