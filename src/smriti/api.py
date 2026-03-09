"""FastAPI HTTP API for memoryd.

Endpoints:
  GET  /v1/health  -- liveness + component status
  POST /v1/events  -- collectors push SourceEvents
  POST /v1/search  -- semantic memory search
  GET  /v1/stats   -- pipeline stats
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from smriti.config import Settings, load_settings
from smriti.daemon import Daemon
from smriti.models.events import SourceEvent
from smriti.retrieval import RetrievalEngine

_daemon: Daemon | None = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Startup: bootstrap the daemon. Shutdown: release resources."""
    global _daemon, _start_time
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else load_settings()
    _daemon = await Daemon.from_settings(settings)
    _start_time = time.monotonic()
    yield
    if _daemon:
        await _daemon.shutdown()
        _daemon = None


def create_app(settings: Settings | None = None) -> FastAPI:
    """Application factory."""
    app = FastAPI(title="Smriti", version="0.1.0", lifespan=lifespan)
    if settings:
        app.state.settings = settings
    app.include_router(_router)
    return app


def _get_daemon() -> Daemon:
    if _daemon is None:
        raise HTTPException(status_code=503, detail="Daemon not initialized")
    return _daemon


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

from fastapi import APIRouter  # noqa: E402

_router = APIRouter(prefix="/v1")


# -- Health ----------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float


@_router.get("/health")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )


# -- Events ----------------------------------------------------------------


class IngestRequest(BaseModel):
    events: list[SourceEvent] = Field(..., min_length=1, max_length=500)


class IngestResponse(BaseModel):
    accepted: int


@_router.post("/events")
async def ingest_events(req: IngestRequest) -> IngestResponse:
    daemon = _get_daemon()
    for event in req.events:
        await daemon.event_bus.publish(event)
    return IngestResponse(accepted=len(req.events))


# -- Search ----------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    tier: str | None = None


class SearchResult(BaseModel):
    content: str
    tier: str
    importance: float
    score: float
    created_at: str


class SearchResponse(BaseModel):
    results: list[SearchResult]


@_router.post("/search")
async def search_memories(req: SearchRequest) -> SearchResponse:
    daemon = _get_daemon()

    try:
        embeddings = await daemon.provider.embed([req.query])
        query_embedding = embeddings[0]
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Embedding failed: {exc}") from exc

    engine = RetrievalEngine()
    async with daemon.session_factory() as session:
        ranked = await engine.search(
            session, query_embedding, top_k=req.top_k, tier=req.tier
        )

    results = [
        SearchResult(
            content=rm.row.content,
            tier=rm.row.tier,
            importance=rm.row.importance,
            score=round(rm.final_score, 4),
            created_at=rm.row.created_at.isoformat(),
        )
        for rm in ranked
    ]
    return SearchResponse(results=results)


# -- Stats -----------------------------------------------------------------


class StatsResponse(BaseModel):
    batches_processed: int
    events_consumed: int
    events_filtered: int
    memories_created: int
    uptime_seconds: float


@_router.get("/stats")
async def stats() -> StatsResponse:
    daemon = _get_daemon()
    s = daemon.stats
    return StatsResponse(
        batches_processed=s.batches_processed,
        events_consumed=s.events_consumed,
        events_filtered=s.events_filtered,
        memories_created=s.memories_created,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
    )
