"""Tests for memory consolidation worker.

All LLM and store operations are mocked -- no live infrastructure.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from smriti.consolidation import (
    DECAY_FACTOR,
    PRUNE_THRESHOLD,
    ConsolidationResult,
    ConsolidationWorker,
)
from smriti.models.memory import EpisodicMemory, SemanticNode


def _episode(summary: str = "Debugged CORS issue", **kwargs) -> EpisodicMemory:
    return EpisodicMemory(
        id=kwargs.get("id", uuid4()),
        source=kwargs.get("source", "cursor"),
        summary=summary,
        importance=kwargs.get("importance", 0.8),
    )


def _node(label: str = "TypeScript preference", confidence: float = 0.7) -> SemanticNode:
    return SemanticNode(
        id=uuid4(),
        label=label,
        node_type="preference",
        confidence=confidence,
    )


def _worker(
    *,
    complete_return: str = "[]",
    embed_return: list | None = None,
    episodes: list | None = None,
    nodes: list | None = None,
) -> tuple[ConsolidationWorker, AsyncMock]:
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value=complete_return)
    provider.embed = AsyncMock(return_value=embed_return or [[0.1] * 1536])

    episodic_store = AsyncMock()
    episodic_store.list_recent = AsyncMock(return_value=episodes or [])

    semantic_store = AsyncMock()
    semantic_store.list_nodes = AsyncMock(return_value=nodes or [])
    semantic_store.search_nodes_by_vector = AsyncMock(return_value=[])
    semantic_store.save_node = AsyncMock()
    semantic_store.delete_node = AsyncMock()

    worker = ConsolidationWorker(
        provider=provider,
        episodic_store=episodic_store,
        semantic_store=semantic_store,
    )
    session = AsyncMock()
    return worker, session


class TestConsolidationResult:
    def test_defaults(self) -> None:
        r = ConsolidationResult()
        assert r.episodes_processed == 0
        assert r.nodes_created == 0


class TestNoEpisodes:
    @pytest.mark.asyncio
    async def test_returns_early(self) -> None:
        worker, session = _worker()
        result = await worker.run(session)
        assert result.episodes_processed == 0
        assert result.facts_extracted == 0


class TestFactExtraction:
    @pytest.mark.asyncio
    async def test_extracts_and_creates_nodes(self) -> None:
        facts = [
            {"label": "Python skill", "node_type": "skill", "confidence": 0.9},
            {"label": "Prefers dark mode", "node_type": "preference", "confidence": 0.8},
        ]
        episodes = [_episode("Used Python extensively"), _episode("Set dark mode")]

        worker, session = _worker(
            complete_return=json.dumps(facts),
            episodes=episodes,
        )

        result = await worker.run(session)

        assert result.episodes_processed == 2
        assert result.facts_extracted == 2
        assert result.nodes_created == 2
        assert worker.semantic_store.save_node.call_count == 2

    @pytest.mark.asyncio
    async def test_llm_failure_returns_no_facts(self) -> None:
        worker, session = _worker(episodes=[_episode()])
        worker.provider.complete.side_effect = RuntimeError("LLM down")

        result = await worker.run(session)

        assert result.episodes_processed == 1
        assert result.facts_extracted == 0
        assert result.nodes_created == 0


class TestDeduplication:
    @pytest.mark.asyncio
    async def test_merges_similar_node(self) -> None:
        existing_node = _node("Python skill", confidence=0.6)
        facts = [{"label": "Python skill", "node_type": "skill", "confidence": 0.9}]
        episodes = [_episode("Built a Python CLI")]

        worker, session = _worker(
            complete_return=json.dumps(facts),
            episodes=episodes,
        )
        worker.semantic_store.search_nodes_by_vector = AsyncMock(
            return_value=[(existing_node, 0.05)]  # distance 0.05 → similarity 0.95
        )

        result = await worker.run(session)

        assert result.nodes_merged == 1
        assert result.nodes_created == 0
        worker.semantic_store.delete_node.assert_called()
        worker.semantic_store.save_node.assert_called()

    @pytest.mark.asyncio
    async def test_creates_when_no_match(self) -> None:
        facts = [{"label": "Rust interest", "node_type": "skill", "confidence": 0.5}]

        worker, session = _worker(
            complete_return=json.dumps(facts),
            episodes=[_episode("Started learning Rust")],
        )
        worker.semantic_store.search_nodes_by_vector = AsyncMock(
            return_value=[(_node("Python skill"), 0.8)]  # distance 0.8 → similarity 0.2
        )

        result = await worker.run(session)

        assert result.nodes_created == 1
        assert result.nodes_merged == 0


class TestDecay:
    @pytest.mark.asyncio
    async def test_decays_all_nodes(self) -> None:
        nodes = [_node("Fact A", confidence=0.8), _node("Fact B", confidence=0.5)]
        worker, session = _worker(nodes=nodes)

        result = await worker.run(session)

        assert result.nodes_decayed == 2
        calls = worker.semantic_store.save_node.call_args_list
        for call in calls:
            saved_node = call.args[1]
            assert saved_node.confidence < 0.8


class TestPrune:
    @pytest.mark.asyncio
    async def test_prunes_low_confidence(self) -> None:
        nodes = [
            _node("Strong fact", confidence=0.9),
            _node("Weak fact", confidence=0.05),
        ]
        worker, session = _worker(nodes=nodes)

        result = await worker.run(session)

        assert result.nodes_pruned >= 1

    @pytest.mark.asyncio
    async def test_no_prune_above_threshold(self) -> None:
        nodes = [_node("Good fact", confidence=0.8)]
        worker, session = _worker(nodes=nodes)

        result = await worker.run(session)

        assert result.nodes_pruned == 0
