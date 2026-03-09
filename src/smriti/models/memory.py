"""Memory tier models spanning buffer, working, episodic, and semantic layers."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MemoryTier(str, Enum):
    BUFFER = "buffer"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


# ---------------------------------------------------------------------------
# Tier 1: Buffer Memory (immediate conversation turns)
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    """A single user or assistant message in the active conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0


# ---------------------------------------------------------------------------
# Tier 2: Working Memory (session-level extracted facts)
# ---------------------------------------------------------------------------


class EntityInfo(BaseModel):
    """An entity mentioned in the current session with contextual metadata."""

    entity_type: str  # "person", "project", "component", "tool", ...
    status: str = ""
    properties: dict[str, Any] = Field(default_factory=dict)


class Goal(BaseModel):
    """Something the user is trying to accomplish in this session."""

    description: str
    status: str = "active"  # "active", "completed", "abandoned"


class Decision(BaseModel):
    """A choice made during the session, with rationale."""

    choice: str
    reason: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WorkingMemory(BaseModel):
    """Session-scoped extracted context that survives buffer overflow.

    Updated incrementally after each turn. Persisted to Valkey during
    the session, then consolidated into an episodic memory at session end.
    """

    session_id: str
    summary: str = ""
    entities: dict[str, EntityInfo] = Field(default_factory=dict)
    active_goals: list[Goal] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    token_count: int = 0


# ---------------------------------------------------------------------------
# Tier 3: Episodic Memory (cross-session compressed episodes)
# ---------------------------------------------------------------------------


class MemoryLink(BaseModel):
    """A directed edge from one memory to another."""

    target_id: UUID
    relation: str  # "related_to", "caused_by", "follows", ...
    weight: float = 1.0


class EpisodicMemory(BaseModel):
    """A compressed record of a past conversation or activity episode.

    Stored in Postgres with a vector embedding for similarity search,
    a timestamp for recency queries, and entity/topic tags for keyword search.
    """

    id: UUID = Field(default_factory=uuid4)
    conversation_id: str = ""
    source: str = ""
    summary: str
    key_facts: list[str] = Field(default_factory=list)
    embedding: list[float] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance: float = 0.5
    emotional_valence: float = 0.0
    links: list[MemoryLink] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Tier 4: Semantic Memory (long-term knowledge graph)
# ---------------------------------------------------------------------------


class SemanticNode(BaseModel):
    """A node in the user's long-term knowledge graph.

    Represents a distilled fact: a person, preference, skill, project, etc.
    """

    id: UUID = Field(default_factory=uuid4)
    label: str
    node_type: str  # "person", "preference", "skill", "project", "concept"
    properties: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    source_episodes: list[UUID] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SemanticEdge(BaseModel):
    """A typed, weighted relationship between two semantic nodes."""

    source_id: UUID
    target_id: UUID
    relation: str  # "prefers", "works_on", "knows", "skilled_in", ...
    weight: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)
