"""Tests for context assembly and tier-adaptive rendering."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from smriti.context import (
    ContextAssembler,
    render_buffer,
    render_episodes,
    render_semantic,
    render_working,
    token_count,
)
from smriti.db.tables import MemoriesTable
from smriti.models.memory import (
    ConversationTurn,
    Decision,
    EntityInfo,
    Goal,
    WorkingMemory,
)
from smriti.retrieval import RankedMemory


def _row(content: str, tier: str = "episodic", importance: float = 0.7, **kwargs):
    return MemoriesTable(
        id=uuid.uuid4(),
        tier=tier,
        content=content,
        importance=importance,
        created_at=kwargs.get("created_at", datetime.now(timezone.utc)),
        accessed_at=kwargs.get("accessed_at", datetime.now(timezone.utc)),
        access_count=0,
        metadata_=kwargs.get("metadata_", {"source": "cursor"}),
        facts=kwargs.get("facts"),
    )


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


class TestTokenCount:
    def test_empty_string(self) -> None:
        assert token_count("") == 0

    def test_short_string(self) -> None:
        count = token_count("Hello, world!")
        assert 2 <= count <= 5


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


class TestRenderBuffer:
    def test_renders_turns(self) -> None:
        turns = [
            ConversationTurn(role="user", content="What is CORS?"),
            ConversationTurn(role="assistant", content="CORS stands for..."),
        ]
        text = render_buffer(turns)
        assert "user: What is CORS?" in text
        assert "assistant: CORS stands for..." in text

    def test_empty(self) -> None:
        assert render_buffer([]) == ""


class TestRenderWorking:
    def test_full_working_memory(self) -> None:
        wm = WorkingMemory(
            session_id="s1",
            summary="Debugging auth service",
            active_goals=[Goal(description="fix CORS"), Goal(description="deploy")],
            entities={"auth_svc": EntityInfo(entity_type="component")},
            decisions=[Decision(choice="use middleware")],
        )
        text = render_working(wm)
        assert "<working_memory>" in text
        assert "session: Debugging auth service" in text
        assert "fix CORS" in text
        assert "auth_svc: component" in text
        assert "use middleware" in text
        assert "</working_memory>" in text

    def test_empty_working_memory(self) -> None:
        wm = WorkingMemory(session_id="s1")
        text = render_working(wm)
        assert "<working_memory>" in text
        assert "</working_memory>" in text


class TestRenderEpisodes:
    def test_renders_timestamped_prose(self) -> None:
        rows = [
            _row("Debugged CORS issue", metadata_={"source": "cursor"}),
            _row("Set up Docker Compose", metadata_={"source": "terminal"}),
        ]
        text = render_episodes(rows)
        assert "<episodes>" in text
        assert "cursor" in text
        assert "Debugged CORS" in text
        assert "</episodes>" in text


class TestRenderSemantic:
    def test_renders_triples(self) -> None:
        rows = [
            _row(
                "TypeScript",
                tier="semantic",
                facts={"node_type": "language", "confidence": 0.92},
            ),
        ]
        text = render_semantic(rows)
        assert "<user_knowledge>" in text
        assert "TypeScript (language, confidence: 0.92)" in text
        assert "</user_knowledge>" in text


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------


class TestContextAssembler:
    def test_basic_assembly(self) -> None:
        assembler = ContextAssembler()
        turns = [ConversationTurn(role="user", content="Hello")]
        result = assembler.assemble(
            system_prompt="You are a helpful assistant.",
            buffer_turns=turns,
            budget=8000,
        )
        assert "You are a helpful assistant." in result
        assert "user: Hello" in result

    def test_includes_working_memory(self) -> None:
        assembler = ContextAssembler()
        wm = WorkingMemory(session_id="s1", summary="Building memory system")
        result = assembler.assemble(working_mem=wm, budget=8000)
        assert "<working_memory>" in result
        assert "Building memory system" in result

    def test_includes_ranked_memories(self) -> None:
        assembler = ContextAssembler()
        memories = [
            RankedMemory(row=_row("Debugged CORS", tier="episodic"), final_score=0.9),
            RankedMemory(
                row=_row(
                    "Python", tier="semantic",
                    facts={"node_type": "skill", "confidence": 0.95},
                ),
                final_score=0.8,
            ),
        ]
        result = assembler.assemble(ranked_memories=memories, budget=8000)
        assert "<user_knowledge>" in result
        assert "<episodes>" in result
        assert "Python" in result
        assert "Debugged CORS" in result

    def test_respects_budget(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(
            system_prompt="Be helpful.",
            ranked_memories=[
                RankedMemory(row=_row(f"memory number {i} " * 20), final_score=0.5)
                for i in range(50)
            ],
            budget=100,
        )
        count = token_count(result)
        assert count < 150  # greedy fill should stop well before all 50
        assert "memory number 0" in result  # at least some included

    def test_empty_assembly(self) -> None:
        assembler = ContextAssembler()
        result = assembler.assemble(budget=8000)
        assert result == ""
