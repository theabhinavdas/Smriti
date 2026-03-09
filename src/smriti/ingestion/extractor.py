"""Memory extractor: transforms salient events into structured memories via LLM.

Groups related events, extracts atomic facts, entities, topics, assigns
importance, and generates embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from smriti.ingestion.salience import ScoredEvent
from smriti.llm_utils import parse_llm_json
from smriti.models.memory import SourceMetadata
from smriti.provider import ModelProvider

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = """\
You are a memory extraction system. Given a batch of user activity events,
extract structured memories. For each distinct memory, provide:

1. "summary": A 1-2 sentence natural language summary
2. "key_facts": A list of atomic facts (short strings)
3. "entities": Named entities mentioned (people, projects, tools, etc.)
4. "topics": Topic tags
5. "importance": Float 0.0-1.0 (how important for long-term memory)
6. "memory_type": One of "episodic" (a task/episode), "semantic" (a durable fact/preference)

Return a JSON array of memory objects.

Events:
{events}"""


@dataclass
class ExtractedMemory:
    """A structured memory produced by the extractor."""

    summary: str
    key_facts: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    importance: float = 0.5
    memory_type: str = "episodic"  # "episodic" or "semantic"
    source: str = ""
    embedding: list[float] = field(default_factory=list)
    source_metadata: list[SourceMetadata] = field(default_factory=list)


class MemoryExtractor:
    def __init__(self, provider: ModelProvider) -> None:
        self._provider = provider

    async def extract(self, scored_events: list[ScoredEvent]) -> list[ExtractedMemory]:
        """Extract structured memories from a batch of salient events."""
        if not scored_events:
            return []

        event_lines = []
        sources: set[str] = set()
        all_source_meta: list[SourceMetadata] = []
        for se in scored_events:
            e = se.event
            sources.add(e.source)
            all_source_meta.append(SourceMetadata.from_event(e))
            line = f"[{e.source}/{e.event_type}, salience={se.score:.1f}] {e.raw_content[:300]}"
            event_lines.append(line)

        prompt = EXTRACTION_PROMPT.format(events="\n".join(event_lines))

        try:
            raw = await self._provider.complete(
                [{"role": "user", "content": prompt}],
                max_tokens=self._provider.config.max_tokens_per_extraction,
                temperature=0.0,
            )
            memories_data = parse_llm_json(raw)
        except Exception:
            logger.warning("Memory extraction LLM call failed", exc_info=True)
            return self._fallback_extract(scored_events)

        primary_source = next(iter(sources)) if len(sources) == 1 else "mixed"
        memories = self._parse_extracted(memories_data, primary_source, all_source_meta)

        summaries = [m.summary for m in memories]
        if summaries:
            try:
                embeddings = await self._provider.embed(summaries)
                for mem, emb in zip(memories, embeddings):
                    mem.embedding = emb
            except Exception:
                logger.warning("Embedding generation failed", exc_info=True)

        return memories

    @staticmethod
    def _parse_extracted(
        data: list[dict],
        source: str,
        source_metadata: list[SourceMetadata],
    ) -> list[ExtractedMemory]:
        results: list[ExtractedMemory] = []
        for item in data:
            results.append(
                ExtractedMemory(
                    summary=item.get("summary", ""),
                    key_facts=item.get("key_facts", []),
                    entities=item.get("entities", []),
                    topics=item.get("topics", []),
                    importance=float(item.get("importance", 0.5)),
                    memory_type=item.get("memory_type", "episodic"),
                    source=source,
                    source_metadata=list(source_metadata),
                )
            )
        return results

    @staticmethod
    def _fallback_extract(scored_events: list[ScoredEvent]) -> list[ExtractedMemory]:
        """If the LLM call fails, create one memory per event as a best-effort fallback."""
        return [
            ExtractedMemory(
                summary=se.event.raw_content[:500],
                importance=se.score,
                memory_type="episodic",
                source=se.event.source,
                source_metadata=[SourceMetadata.from_event(se.event)],
            )
            for se in scored_events
        ]
