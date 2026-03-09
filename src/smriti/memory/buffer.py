"""Tier 1: Buffer Memory -- a fixed-size ring buffer of recent conversation turns.

Always in-memory, O(1) append/evict, injected verbatim into LLM context.
Tier 1 never touches the database or Valkey.
"""

from __future__ import annotations

from collections import deque

from smriti.models.memory import ConversationTurn

DEFAULT_MAX_TURNS = 20


class BufferMemory:
    """Ring buffer holding the most recent conversation turns."""

    def __init__(self, max_turns: int = DEFAULT_MAX_TURNS) -> None:
        self._turns: deque[ConversationTurn] = deque(maxlen=max_turns)

    @property
    def max_turns(self) -> int:
        return self._turns.maxlen  # type: ignore[return-value]

    @property
    def total_tokens(self) -> int:
        return sum(t.token_count for t in self._turns)

    def add(self, turn: ConversationTurn) -> ConversationTurn | None:
        """Append a turn. Returns the evicted turn if the buffer was full, else None."""
        evicted: ConversationTurn | None = None
        if len(self._turns) == self._turns.maxlen:
            evicted = self._turns[0]
        self._turns.append(turn)
        return evicted

    def get_all(self) -> list[ConversationTurn]:
        """Return all turns in chronological order."""
        return list(self._turns)

    def get_recent(self, n: int) -> list[ConversationTurn]:
        """Return the last n turns."""
        turns = list(self._turns)
        return turns[-n:] if n < len(turns) else turns

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
