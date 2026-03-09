"""Tests for the ingestion pipeline: salience filter, extractor, and tier router."""

from __future__ import annotations

import json

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from smriti.config import ModelConfig
from smriti.ingestion.extractor import ExtractedMemory, MemoryExtractor
from smriti.ingestion.router import TierRouter
from smriti.ingestion.salience import SalienceFilter, ScoredEvent
from smriti.models.events import SourceEvent
from smriti.models.memory import MemoryTier
from smriti.provider import ModelProvider

def _event(source: str, event_type: str, content: str, **meta) -> SourceEvent:
    return SourceEvent(
        source=source, event_type=event_type, raw_content=content, metadata=meta
    )


# ---------------------------------------------------------------------------
# SalienceFilter: heuristic scoring
# ---------------------------------------------------------------------------


class TestSalienceHeuristic:
    def test_terminal_error_is_max_salience(self) -> None:
        f = SalienceFilter()
        score = f.heuristic_score(_event("terminal", "cmd", "npm test", exit_code=1))
        assert score == 1.0

    def test_terminal_noise_is_zero(self) -> None:
        f = SalienceFilter()
        for cmd in ("ls", "cd /tmp", "pwd", "clear"):
            assert f.heuristic_score(_event("terminal", "cmd", cmd)) == 0.0

    def test_terminal_git_commit_is_high(self) -> None:
        f = SalienceFilter()
        score = f.heuristic_score(_event("terminal", "cmd", "git commit -m 'init'"))
        assert score >= 0.8

    def test_terminal_install_is_high(self) -> None:
        f = SalienceFilter()
        score = f.heuristic_score(_event("terminal", "cmd", "pip install numpy"))
        assert score >= 0.7

    def test_browser_text_selected_is_high(self) -> None:
        f = SalienceFilter()
        score = f.heuristic_score(_event("browser", "text_selected", "important text"))
        assert score >= 0.8

    def test_browser_bounce_is_zero(self) -> None:
        f = SalienceFilter()
        score = f.heuristic_score(
            _event("browser", "page_visited", "http://example.com", dwell_seconds=2)
        )
        assert score == 0.0

    def test_cursor_file_created_is_high(self) -> None:
        f = SalienceFilter()
        score = f.heuristic_score(_event("cursor", "file_created", "new_module.py"))
        assert score >= 0.8

    def test_unknown_source_gets_default(self) -> None:
        f = SalienceFilter()
        score = f.heuristic_score(_event("calendar", "meeting", "standup"))
        assert score == 0.3


# ---------------------------------------------------------------------------
# SalienceFilter: combined pipeline
# ---------------------------------------------------------------------------


class TestSalienceFilterPipeline:
    async def test_drops_noise_keeps_salient(self) -> None:
        f = SalienceFilter()
        events = [
            _event("terminal", "cmd", "ls"),
            _event("terminal", "cmd", "git commit -m 'feat'"),
            _event("terminal", "cmd", "clear"),
        ]
        result = await f.filter(events)
        assert len(result) == 1
        assert result[0].event.raw_content == "git commit -m 'feat'"

    async def test_uncertain_events_kept_without_llm(self) -> None:
        """Without a provider, uncertain events (0.2-0.7) are passed to LLM scoring
        which is a no-op, so they remain with their heuristic scores."""
        f = SalienceFilter()
        events = [_event("terminal", "cmd", "make test")]
        result = await f.filter(events)
        assert len(result) == 1
        assert result[0].score == 0.3


# ---------------------------------------------------------------------------
# SalienceFilter: LLM scoring
# ---------------------------------------------------------------------------


class TestSalienceLLMScoring:
    async def test_llm_rescores_events(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "[8, 2]"}}]},
            )

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://openrouter.ai/api/v1",
        )
        provider = ModelProvider(config=ModelConfig(api_key="k"), http_client=client)
        f = SalienceFilter(provider=provider)

        events = [
            ScoredEvent(event=_event("terminal", "cmd", "make deploy"), score=0.3),
            ScoredEvent(event=_event("terminal", "cmd", "cat README"), score=0.3),
        ]
        result = await f.llm_score(events)
        assert result[0].score == 0.8
        assert result[1].score == 0.2


# ---------------------------------------------------------------------------
# MemoryExtractor
# ---------------------------------------------------------------------------


class TestMemoryExtractor:
    async def test_extract_parses_llm_response(self) -> None:
        llm_response = json.dumps([
            {
                "summary": "User set up Docker Compose for local dev",
                "key_facts": ["postgres on 5432", "valkey on 6379"],
                "entities": ["Docker", "Postgres"],
                "topics": ["devops"],
                "importance": 0.7,
                "memory_type": "episodic",
            }
        ])
        embed_response = {"data": [{"embedding": [0.1, 0.2], "index": 0}]}

        call_count = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            body = json.loads(request.content)
            if "input" in body:
                return httpx.Response(200, json=embed_response)
            return httpx.Response(
                200, json={"choices": [{"message": {"content": llm_response}}]}
            )

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://openrouter.ai/api/v1",
        )
        provider = ModelProvider(config=ModelConfig(api_key="k"), http_client=client)
        extractor = MemoryExtractor(provider)

        scored = [ScoredEvent(event=_event("terminal", "cmd", "docker compose up"), score=0.8)]
        result = await extractor.extract(scored)

        assert len(result) == 1
        assert result[0].summary == "User set up Docker Compose for local dev"
        assert result[0].entities == ["Docker", "Postgres"]
        assert result[0].embedding == [0.1, 0.2]

    async def test_fallback_on_llm_failure(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal error")

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://openrouter.ai/api/v1",
        )
        provider = ModelProvider(config=ModelConfig(api_key="k"), http_client=client)
        extractor = MemoryExtractor(provider)

        scored = [ScoredEvent(event=_event("terminal", "cmd", "npm install"), score=0.8)]
        result = await extractor.extract(scored)
        assert len(result) == 1
        assert "npm install" in result[0].summary

    async def test_empty_input(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={})

        client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url="https://openrouter.ai/api/v1",
        )
        provider = ModelProvider(config=ModelConfig(api_key="k"), http_client=client)
        assert await MemoryExtractor(provider).extract([]) == []


# ---------------------------------------------------------------------------
# TierRouter
# ---------------------------------------------------------------------------


class TestTierRouter:
    def test_classify_episodic(self) -> None:
        router = TierRouter()
        mem = ExtractedMemory(summary="Debugged CORS", memory_type="episodic", importance=0.7)
        assert router.classify(mem) == MemoryTier.EPISODIC

    def test_classify_semantic_high_confidence(self) -> None:
        router = TierRouter()
        mem = ExtractedMemory(
            summary="User prefers TypeScript", memory_type="semantic", importance=0.9
        )
        assert router.classify(mem) == MemoryTier.SEMANTIC

    def test_classify_semantic_low_confidence_falls_to_episodic(self) -> None:
        router = TierRouter()
        mem = ExtractedMemory(
            summary="Maybe prefers dark mode", memory_type="semantic", importance=0.5
        )
        assert router.classify(mem) == MemoryTier.EPISODIC

    async def test_route_persists_to_correct_stores(self, db_session: AsyncSession) -> None:
        router = TierRouter()
        memories = [
            ExtractedMemory(
                summary="Set up CI pipeline",
                key_facts=["GitHub Actions"],
                memory_type="episodic",
                importance=0.7,
                source="cursor",
            ),
            ExtractedMemory(
                summary="User prefers Python over Java",
                memory_type="semantic",
                importance=0.9,
                source="cursor",
            ),
        ]
        counts = await router.route(db_session, memories)
        assert counts[MemoryTier.EPISODIC] == 1
        assert counts[MemoryTier.SEMANTIC] == 1
