"""Memory consolidation: episodic → semantic promotion.

Mirrors human sleep-like memory processing:
  1. Extract atomic facts from recent episodes (LLM)
  2. Deduplicate against existing semantic nodes (embedding similarity)
  3. Merge into existing nodes or create new ones
  4. Decay confidence of unaccessed nodes
  5. Prune nodes below confidence threshold (forgetting)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from smriti.memory.episodic import EpisodicStore
from smriti.memory.semantic import SemanticStore
from smriti.models.memory import EpisodicMemory, SemanticNode
from smriti.provider import ModelProvider

logger = logging.getLogger(__name__)

DEDUP_SIMILARITY_THRESHOLD = 0.85
DECAY_FACTOR = 0.95
PRUNE_THRESHOLD = 0.1

CONSOLIDATION_PROMPT = """\
You are a knowledge extraction system. Given a batch of episodic memories
(user activity summaries), extract atomic, durable facts about the user.

For each fact, provide:
1. "label": A short noun-phrase label (e.g. "TypeScript preference")
2. "node_type": One of "preference", "skill", "project", "person", "concept", "tool"
3. "confidence": Float 0.0-1.0 (how certain is this fact?)
4. "properties": Dict of extra key-value pairs

Return a JSON array of objects. Only extract facts that are likely to remain
true over time (preferences, skills, relationships), not transient tasks.

Episodes:
{episodes}"""


@dataclass
class ConsolidationResult:
    """Summary of one consolidation run."""

    episodes_processed: int = 0
    facts_extracted: int = 0
    nodes_created: int = 0
    nodes_merged: int = 0
    nodes_decayed: int = 0
    nodes_pruned: int = 0


@dataclass
class ConsolidationWorker:
    """Periodically promotes episodic memories into the semantic graph."""

    provider: ModelProvider
    episodic_store: EpisodicStore = field(default_factory=EpisodicStore)
    semantic_store: SemanticStore = field(default_factory=SemanticStore)
    batch_size: int = 20

    async def run(self, session: AsyncSession) -> ConsolidationResult:
        """Execute one consolidation cycle."""
        result = ConsolidationResult()

        episodes = await self.episodic_store.list_recent(
            session, limit=self.batch_size
        )
        if episodes:
            result.episodes_processed = len(episodes)

            facts = await self._extract_facts(episodes)
            result.facts_extracted = len(facts)

            for fact in facts:
                merged = await self._dedup_and_merge(session, fact, episodes)
                if merged:
                    result.nodes_merged += 1
                else:
                    await self._create_node(session, fact, episodes)
                    result.nodes_created += 1

        result.nodes_decayed = await self._decay(session)
        result.nodes_pruned = await self._prune(session)

        logger.info(
            "Consolidation: %d episodes → %d facts, %d created, %d merged, "
            "%d decayed, %d pruned",
            result.episodes_processed,
            result.facts_extracted,
            result.nodes_created,
            result.nodes_merged,
            result.nodes_decayed,
            result.nodes_pruned,
        )
        return result

    # ------------------------------------------------------------------
    # Step 1: Extract atomic facts via LLM
    # ------------------------------------------------------------------

    async def _extract_facts(
        self, episodes: list[EpisodicMemory]
    ) -> list[dict]:
        episode_lines = []
        for ep in episodes:
            episode_lines.append(
                f"- [{ep.source}, importance={ep.importance:.2f}] {ep.summary}"
            )

        prompt = CONSOLIDATION_PROMPT.format(
            episodes="\n".join(episode_lines)
        )

        try:
            raw = await self.provider.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.0,
            )
            return json.loads(raw)
        except Exception:
            logger.warning("Consolidation LLM call failed", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Step 2+3: Deduplicate and merge or create
    # ------------------------------------------------------------------

    async def _dedup_and_merge(
        self,
        session: AsyncSession,
        fact: dict,
        source_episodes: list[EpisodicMemory],
    ) -> bool:
        """Check if fact matches an existing node. If so, merge and return True."""
        label = fact.get("label", "")
        if not label:
            return False

        try:
            embeddings = await self.provider.embed([label])
            query_embedding = embeddings[0]
        except Exception:
            logger.warning("Embedding for dedup failed", exc_info=True)
            return False

        matches = await self.semantic_store.search_nodes_by_vector(
            session, query_embedding, limit=3
        )

        for node, distance in matches:
            similarity = 1.0 - distance
            if similarity >= DEDUP_SIMILARITY_THRESHOLD:
                await self._merge_into_node(session, node, fact, source_episodes)
                return True

        return False

    async def _merge_into_node(
        self,
        session: AsyncSession,
        existing: SemanticNode,
        fact: dict,
        source_episodes: list[EpisodicMemory],
    ) -> None:
        """Boost confidence and update properties of an existing node."""
        new_confidence = min(
            1.0,
            existing.confidence + fact.get("confidence", 0.5) * 0.2,
        )
        new_props = {**existing.properties, **fact.get("properties", {})}
        episode_ids = list(existing.source_episodes) + [ep.id for ep in source_episodes]

        updated = SemanticNode(
            id=existing.id,
            label=existing.label,
            node_type=existing.node_type,
            properties=new_props,
            confidence=new_confidence,
            source_episodes=episode_ids,
            created_at=existing.created_at,
        )
        await self.semantic_store.delete_node(session, existing.id)
        await self.semantic_store.save_node(session, updated)

    async def _create_node(
        self,
        session: AsyncSession,
        fact: dict,
        source_episodes: list[EpisodicMemory],
    ) -> None:
        """Create a new semantic node from an extracted fact."""
        node = SemanticNode(
            id=uuid4(),
            label=fact.get("label", "unknown"),
            node_type=fact.get("node_type", "concept"),
            properties=fact.get("properties", {}),
            confidence=fact.get("confidence", 0.5),
            source_episodes=[ep.id for ep in source_episodes],
        )
        await self.semantic_store.save_node(session, node)

    # ------------------------------------------------------------------
    # Step 4: Decay unaccessed nodes
    # ------------------------------------------------------------------

    async def _decay(self, session: AsyncSession) -> int:
        """Reduce confidence of all semantic nodes by DECAY_FACTOR."""
        nodes = await self.semantic_store.list_nodes(session, limit=500)
        decayed = 0
        for node in nodes:
            new_conf = node.confidence * DECAY_FACTOR
            if new_conf != node.confidence:
                updated = SemanticNode(
                    id=node.id,
                    label=node.label,
                    node_type=node.node_type,
                    properties=node.properties,
                    confidence=new_conf,
                    source_episodes=node.source_episodes,
                    created_at=node.created_at,
                )
                await self.semantic_store.delete_node(session, node.id)
                await self.semantic_store.save_node(session, updated)
                decayed += 1
        return decayed

    # ------------------------------------------------------------------
    # Step 5: Prune low-confidence nodes
    # ------------------------------------------------------------------

    async def _prune(self, session: AsyncSession) -> int:
        """Remove semantic nodes below PRUNE_THRESHOLD."""
        nodes = await self.semantic_store.list_nodes(session, limit=500)
        pruned = 0
        for node in nodes:
            if node.confidence < PRUNE_THRESHOLD:
                await self.semantic_store.delete_node(session, node.id)
                pruned += 1
        return pruned
