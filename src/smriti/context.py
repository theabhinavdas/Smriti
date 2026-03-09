"""Context assembly and tier-adaptive rendering.

The renderer converts memories from each tier into compact, LLM-friendly
formats. The assembler orchestrates the full context window construction
that runs before every LLM call.
"""

from __future__ import annotations

import tiktoken

from smriti.db.tables import MemoriesTable
from smriti.models.memory import ConversationTurn, WorkingMemory
from smriti.retrieval import RankedMemory

_encoder = tiktoken.get_encoding("cl100k_base")


def token_count(text: str) -> int:
    return len(_encoder.encode(text))


# ---------------------------------------------------------------------------
# Tier-adaptive renderers
# ---------------------------------------------------------------------------


def render_buffer(turns: list[ConversationTurn]) -> str:
    """Tier 1: verbatim conversation turns."""
    lines: list[str] = []
    for t in turns:
        lines.append(f"{t.role}: {t.content}")
    return "\n".join(lines)


def render_working(wm: WorkingMemory) -> str:
    """Tier 2: compact key-value block."""
    lines = ["<working_memory>"]
    if wm.summary:
        lines.append(f"session: {wm.summary}")
    if wm.active_goals:
        goals = ", ".join(g.description for g in wm.active_goals)
        lines.append(f"goals: [{goals}]")
    if wm.entities:
        ents = ", ".join(
            f"{name}: {info.entity_type}" for name, info in wm.entities.items()
        )
        lines.append(f"entities: {{{ents}}}")
    if wm.decisions:
        decs = ", ".join(d.choice for d in wm.decisions)
        lines.append(f"decisions: [{decs}]")
    lines.append("</working_memory>")
    return "\n".join(lines)


def render_episodes(rows: list[MemoriesTable]) -> str:
    """Tier 3: timestamped prose fragments."""
    lines = ["<episodes>"]
    sorted_rows = sorted(rows, key=lambda r: r.created_at, reverse=True)
    for row in sorted_rows:
        date = row.created_at.strftime("%b %d")
        source = (row.metadata_ or {}).get("source", "?")
        lines.append(f"[{date}, {source}, {row.importance:.2f}] {row.content}")
    lines.append("</episodes>")
    return "\n".join(lines)


def render_semantic(rows: list[MemoriesTable]) -> str:
    """Tier 4: compact triple notation for knowledge graph nodes."""
    lines = ["<user_knowledge>"]
    for row in rows:
        facts = row.facts or {}
        node_type = facts.get("node_type", "")
        confidence = facts.get("confidence", row.importance)
        props = f"({node_type}"
        if confidence:
            props += f", confidence: {confidence:.2f}"
        props += ")"
        lines.append(f"{row.content} {props}")
    lines.append("</user_knowledge>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Builds the full context window from all memory tiers."""

    def assemble(
        self,
        *,
        system_prompt: str = "",
        buffer_turns: list[ConversationTurn] | None = None,
        working_mem: WorkingMemory | None = None,
        ranked_memories: list[RankedMemory] | None = None,
        budget: int = 8000,
    ) -> str:
        """Assemble context within a token budget.

        Priority order (highest to lowest):
        1. System prompt
        2. Buffer (recent turns, verbatim)
        3. Working memory (session context)
        4. Retrieved memories (ranked, greedily filled)
        """
        sections: list[str] = []
        remaining = budget

        if system_prompt:
            sections.append(system_prompt)
            remaining -= token_count(system_prompt)

        if buffer_turns:
            buf_text = render_buffer(buffer_turns)
            sections.append(buf_text)
            remaining -= token_count(buf_text)

        if working_mem and working_mem.summary:
            wm_text = render_working(working_mem)
            sections.append(wm_text)
            remaining -= token_count(wm_text)

        if ranked_memories and remaining > 0:
            mem_text = self._fill_memories(ranked_memories, remaining)
            if mem_text:
                sections.append(mem_text)

        return "\n\n".join(sections)

    def _fill_memories(self, ranked: list[RankedMemory], budget: int) -> str:
        episodic_rows: list[MemoriesTable] = []
        semantic_rows: list[MemoriesTable] = []
        used = 0

        for rm in ranked:
            row = rm.row
            line_cost = token_count(row.content) + 15  # overhead for formatting
            if used + line_cost > budget:
                break
            if row.tier == "semantic":
                semantic_rows.append(row)
            else:
                episodic_rows.append(row)
            used += line_cost

        parts: list[str] = []
        if semantic_rows:
            parts.append(render_semantic(semantic_rows))
        if episodic_rows:
            parts.append(render_episodes(episodic_rows))
        return "\n".join(parts)
