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
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from smriti.config import Settings, load_settings
from smriti.daemon import Daemon
from smriti.imports.tracker import ImportTracker
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
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=r"^(chrome-extension://.*|http://127\.0\.0\.1:\d+|http://localhost:\d+)$",
        allow_methods=["*"],
        allow_headers=["*"],
    )
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


# -- Imports ---------------------------------------------------------------


class ImportRecord(BaseModel):
    file_path: str
    file_hash: str
    file_size: int
    format: str | None
    events_generated: int
    status: str
    error: str | None
    processed_at: str


class ImportsListResponse(BaseModel):
    imports: list[ImportRecord]


class RetryResponse(BaseModel):
    deleted: bool
    message: str


@_router.get("/imports")
async def list_imports(
    status: str | None = None, limit: int = 50, offset: int = 0
) -> ImportsListResponse:
    daemon = _get_daemon()
    tracker = ImportTracker()
    async with daemon.session_factory() as session:
        rows = await tracker.list_imports(session, status=status, limit=limit, offset=offset)

    records = [
        ImportRecord(
            file_path=r.file_path,
            file_hash=r.file_hash,
            file_size=r.file_size,
            format=r.format,
            events_generated=r.events_generated,
            status=r.status,
            error=r.error,
            processed_at=r.processed_at.isoformat() if r.processed_at else "",
        )
        for r in rows
    ]
    return ImportsListResponse(imports=records)


@_router.post("/imports/retry")
async def retry_import(file_hash: str) -> RetryResponse:
    """Delete the tracking record for a file so it gets re-processed on next scan."""
    daemon = _get_daemon()
    tracker = ImportTracker()
    async with daemon.session_factory() as session:
        async with session.begin():
            deleted = await tracker.delete_by_hash(session, file_hash)

    if deleted:
        return RetryResponse(deleted=True, message="Record deleted; file will be re-imported on next scan.")
    return RetryResponse(deleted=False, message="No import record found for this hash.")


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
