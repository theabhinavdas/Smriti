"""Tests for Tier 1: Buffer Memory (pure in-memory, no I/O)."""

from smriti.memory.buffer import BufferMemory
from smriti.models.memory import ConversationTurn


def _turn(content: str, tokens: int = 10) -> ConversationTurn:
    return ConversationTurn(role="user", content=content, token_count=tokens)


class TestBufferMemory:
    def test_add_and_get_all(self) -> None:
        buf = BufferMemory(max_turns=5)
        buf.add(_turn("hello"))
        buf.add(_turn("world"))
        turns = buf.get_all()
        assert len(turns) == 2
        assert turns[0].content == "hello"
        assert turns[1].content == "world"

    def test_eviction_returns_oldest(self) -> None:
        buf = BufferMemory(max_turns=2)
        buf.add(_turn("a"))
        buf.add(_turn("b"))
        evicted = buf.add(_turn("c"))
        assert evicted is not None
        assert evicted.content == "a"
        assert len(buf) == 2
        assert buf.get_all()[0].content == "b"

    def test_no_eviction_when_not_full(self) -> None:
        buf = BufferMemory(max_turns=10)
        evicted = buf.add(_turn("first"))
        assert evicted is None

    def test_total_tokens(self) -> None:
        buf = BufferMemory(max_turns=5)
        buf.add(_turn("a", tokens=100))
        buf.add(_turn("b", tokens=200))
        assert buf.total_tokens == 300

    def test_get_recent(self) -> None:
        buf = BufferMemory(max_turns=10)
        for i in range(5):
            buf.add(_turn(f"msg-{i}"))
        recent = buf.get_recent(2)
        assert len(recent) == 2
        assert recent[0].content == "msg-3"
        assert recent[1].content == "msg-4"

    def test_get_recent_more_than_available(self) -> None:
        buf = BufferMemory(max_turns=10)
        buf.add(_turn("only"))
        assert len(buf.get_recent(5)) == 1

    def test_clear(self) -> None:
        buf = BufferMemory(max_turns=5)
        buf.add(_turn("x"))
        buf.clear()
        assert len(buf) == 0
        assert buf.total_tokens == 0

    def test_max_turns_property(self) -> None:
        buf = BufferMemory(max_turns=42)
        assert buf.max_turns == 42
