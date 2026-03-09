"""Tier router: decides where each extracted memory is stored.

Routing rules:
- memory_type == "semantic" with high confidence -> Tier 4 (semantic graph)
- memory_type == "episodic"                      -> Tier 3 (episodic store)
- Active session context present                 -> also update Tier 2 (working memory)
"""

from __future__ import annotations

import logging
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from smriti.ingestion.extractor import ExtractedMemory
from smriti.memory.episodic import EpisodicStore
from smriti.memory.semantic import SemanticStore
from smriti.models.memory import EpisodicMemory, MemoryTier, SemanticNode

logger = logging.getLogger(__name__)

SEMANTIC_CONFIDENCE_THRESHOLD = 0.75


class TierRouter:
    def __init__(
        self,
        episodic_store: EpisodicStore | None = None,
        semantic_store: SemanticStore | None = None,
    ) -> None:
        self._episodic = episodic_store or EpisodicStore()
        self._semantic = semantic_store or SemanticStore()

    def classify(self, memory: ExtractedMemory) -> MemoryTier:
        """Determine which tier a memory belongs to."""
        if memory.memory_type == "semantic" and memory.importance >= SEMANTIC_CONFIDENCE_THRESHOLD:
            return MemoryTier.SEMANTIC
        return MemoryTier.EPISODIC

    async def route(
        self,
        session: AsyncSession,
        memories: list[ExtractedMemory],
    ) -> dict[MemoryTier, int]:
        """Persist each memory to its appropriate tier. Returns counts per tier."""
        counts: dict[MemoryTier, int] = {MemoryTier.EPISODIC: 0, MemoryTier.SEMANTIC: 0}

        for mem in memories:
            tier = self.classify(mem)
            try:
                if tier == MemoryTier.SEMANTIC:
                    await self._save_semantic(session, mem)
                else:
                    await self._save_episodic(session, mem)
                counts[tier] += 1
            except Exception:
                logger.warning("Failed to route memory: %s", mem.summary[:80], exc_info=True)

        return counts

    async def _save_episodic(self, session: AsyncSession, mem: ExtractedMemory) -> None:
        episode = EpisodicMemory(
            id=uuid4(),
            source=mem.source,
            summary=mem.summary,
            key_facts=mem.key_facts,
            embedding=mem.embedding,
            entities=mem.entities,
            topics=mem.topics,
            importance=mem.importance,
        )
        await self._episodic.save(session, episode)

    async def _save_semantic(self, session: AsyncSession, mem: ExtractedMemory) -> None:
        node = SemanticNode(
            id=uuid4(),
            label=mem.summary,
            node_type=self._infer_node_type(mem),
            properties={"key_facts": mem.key_facts, "topics": mem.topics},
            confidence=mem.importance,
        )
        await self._semantic.save_node(session, node)

    @staticmethod
    def _infer_node_type(mem: ExtractedMemory) -> str:
        summary_lower = mem.summary.lower()
        if any(kw in summary_lower for kw in ("prefer", "like", "dislike", "favor")):
            return "preference"
        if any(kw in summary_lower for kw in ("skill", "proficien", "experience")):
            return "skill"
        if any(kw in summary_lower for kw in ("project", "repo", "codebase")):
            return "project"
        return "concept"
