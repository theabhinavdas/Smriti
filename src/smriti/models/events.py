"""Event models produced by collectors and consumed by the ingestion pipeline."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class ActivityContext(BaseModel):
    """Captures what the user was doing when an event occurred."""

    project: str | None = None
    working_directory: str | None = None
    active_window: str | None = None
    session_id: str = ""
    duration_seconds: float | None = None


class SourceEvent(BaseModel):
    """Normalized event emitted by any collector.

    Every raw observation (terminal command, page visit, file edit, etc.)
    is converted into this common schema before entering the ingestion pipeline.
    """

    id: UUID = Field(default_factory=uuid4)
    source: str
    event_type: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_content: str
    content_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    activity_context: ActivityContext = Field(default_factory=ActivityContext)

    def model_post_init(self, __context: Any) -> None:
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.raw_content.encode()).hexdigest()[:16]
