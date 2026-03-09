"""Tests for core data models."""

from uuid import UUID

from smriti.models import (
    ActivityContext,
    ConversationTurn,
    Decision,
    EpisodicMemory,
    Goal,
    MemoryLink,
    MemoryTier,
    SemanticEdge,
    SemanticNode,
    SourceEvent,
    WorkingMemory,
)


# ---------------------------------------------------------------------------
# SourceEvent
# ---------------------------------------------------------------------------


class TestSourceEvent:
    def test_auto_generates_id_and_hash(self) -> None:
        event = SourceEvent(source="terminal", event_type="command_executed", raw_content="ls -la")
        assert isinstance(event.id, UUID)
        assert len(event.content_hash) == 16

    def test_same_content_produces_same_hash(self) -> None:
        a = SourceEvent(source="terminal", event_type="cmd", raw_content="echo hello")
        b = SourceEvent(source="terminal", event_type="cmd", raw_content="echo hello")
        assert a.content_hash == b.content_hash

    def test_different_content_produces_different_hash(self) -> None:
        a = SourceEvent(source="terminal", event_type="cmd", raw_content="echo hello")
        b = SourceEvent(source="terminal", event_type="cmd", raw_content="echo world")
        assert a.content_hash != b.content_hash

    def test_explicit_hash_is_preserved(self) -> None:
        event = SourceEvent(
            source="browser",
            event_type="page_visited",
            raw_content="<html>",
            content_hash="custom",
        )
        assert event.content_hash == "custom"

    def test_default_activity_context(self) -> None:
        event = SourceEvent(source="cursor", event_type="file_edited", raw_content="diff")
        assert event.activity_context.session_id == ""
        assert event.activity_context.project is None

    def test_roundtrip_serialization(self) -> None:
        event = SourceEvent(
            source="terminal",
            event_type="command_executed",
            raw_content="git commit -m 'init'",
            metadata={"exit_code": 0, "cwd": "/home/user/project"},
            activity_context=ActivityContext(project="smriti", session_id="sess-1"),
        )
        data = event.model_dump()
        restored = SourceEvent.model_validate(data)
        assert restored.source == "terminal"
        assert restored.activity_context.project == "smriti"
        assert restored.content_hash == event.content_hash


# ---------------------------------------------------------------------------
# Memory Tier Enum
# ---------------------------------------------------------------------------


class TestMemoryTier:
    def test_values(self) -> None:
        assert MemoryTier.BUFFER.value == "buffer"
        assert MemoryTier.WORKING.value == "working"
        assert MemoryTier.EPISODIC.value == "episodic"
        assert MemoryTier.SEMANTIC.value == "semantic"

    def test_from_string(self) -> None:
        assert MemoryTier("episodic") is MemoryTier.EPISODIC


# ---------------------------------------------------------------------------
# Tier 1: Buffer
# ---------------------------------------------------------------------------


class TestConversationTurn:
    def test_defaults(self) -> None:
        turn = ConversationTurn(role="user", content="hello")
        assert turn.token_count == 0
        assert turn.timestamp is not None


# ---------------------------------------------------------------------------
# Tier 2: Working Memory
# ---------------------------------------------------------------------------


class TestWorkingMemory:
    def test_empty_session(self) -> None:
        wm = WorkingMemory(session_id="sess-1")
        assert wm.summary == ""
        assert wm.entities == {}
        assert wm.active_goals == []
        assert wm.decisions == []

    def test_with_entities_and_goals(self) -> None:
        from smriti.models import EntityInfo

        wm = WorkingMemory(
            session_id="sess-2",
            summary="Debugging auth service",
            entities={"auth_service": EntityInfo(entity_type="component", status="debugging")},
            active_goals=[Goal(description="fix CORS error")],
            decisions=[Decision(choice="use middleware", reason="simpler than proxy")],
        )
        assert "auth_service" in wm.entities
        assert wm.active_goals[0].status == "active"
        assert wm.decisions[0].choice == "use middleware"


# ---------------------------------------------------------------------------
# Tier 3: Episodic Memory
# ---------------------------------------------------------------------------


class TestEpisodicMemory:
    def test_defaults(self) -> None:
        ep = EpisodicMemory(summary="Debugged CORS issue in auth service")
        assert isinstance(ep.id, UUID)
        assert ep.importance == 0.5
        assert ep.emotional_valence == 0.0
        assert ep.access_count == 0
        assert ep.embedding == []

    def test_with_links(self) -> None:
        from uuid import uuid4

        target = uuid4()
        ep = EpisodicMemory(
            summary="Set up Docker Compose",
            key_facts=["postgres on port 5432", "valkey on port 6379"],
            links=[MemoryLink(target_id=target, relation="related_to")],
        )
        assert len(ep.links) == 1
        assert ep.links[0].target_id == target

    def test_roundtrip(self) -> None:
        ep = EpisodicMemory(
            summary="Read HNSW docs",
            source="browser",
            entities=["HNSW", "Pinecone"],
            topics=["vector-search"],
            importance=0.65,
        )
        restored = EpisodicMemory.model_validate(ep.model_dump())
        assert restored.summary == ep.summary
        assert restored.entities == ["HNSW", "Pinecone"]


# ---------------------------------------------------------------------------
# Tier 4: Semantic Memory
# ---------------------------------------------------------------------------


class TestSemanticNode:
    def test_defaults(self) -> None:
        node = SemanticNode(label="TypeScript", node_type="language")
        assert isinstance(node.id, UUID)
        assert node.confidence == 0.5
        assert node.source_episodes == []

    def test_with_properties(self) -> None:
        node = SemanticNode(
            label="prefers dark mode",
            node_type="preference",
            properties={"scope": "ui"},
            confidence=0.88,
        )
        assert node.properties["scope"] == "ui"


class TestSemanticEdge:
    def test_construction(self) -> None:
        from uuid import uuid4

        src, tgt = uuid4(), uuid4()
        edge = SemanticEdge(source_id=src, target_id=tgt, relation="prefers")
        assert edge.weight == 1.0
        assert edge.metadata == {}
