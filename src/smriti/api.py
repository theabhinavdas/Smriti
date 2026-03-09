"""FastAPI HTTP API for memoryd.

Endpoints:
  GET    /v1/health              -- liveness + component status
  POST   /v1/events              -- collectors push SourceEvents
  POST   /v1/search              -- semantic memory search
  GET    /v1/memories            -- browse/list memories with filters
  GET    /v1/memories/counts     -- memory counts by tier
  DELETE /v1/memories/{id}       -- delete a memory and its edges
  GET    /v1/graph               -- semantic knowledge graph (nodes + edges)
  GET    /v1/stats               -- pipeline stats
  GET    /                       -- memory browser UI
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID

import valkey.exceptions
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from smriti.config import Settings, load_settings
from smriti.daemon import Daemon
from smriti.db.repository import EdgeRepository, MemoryRepository
from smriti.imports.tracker import ImportTracker
from smriti.models.events import SourceEvent
from smriti.retrieval import RetrievalEngine

_STATIC_DIR = Path(__file__).parent / "static"

_daemon: Daemon | None = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN201
    """Startup: bootstrap the daemon and start the consume loop. Shutdown: release resources."""
    global _daemon, _start_time
    settings: Settings = app.state.settings if hasattr(app.state, "settings") else load_settings()
    _daemon = await Daemon.from_settings(settings)
    _start_time = time.monotonic()
    run_task = asyncio.create_task(_daemon.run())
    yield
    # Signal the daemon loop to stop, then cancel the task so it
    # unblocks from xreadgroup. Valkey's retry logic may convert the
    # CancelledError into its own ConnectionError, so we catch both.
    await _daemon.stop()
    run_task.cancel()
    try:
        await run_task
    except (asyncio.CancelledError, valkey.exceptions.ConnectionError):
        pass
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

    @app.get("/", include_in_schema=False)
    async def ui() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html", media_type="text/html")

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


# -- Memories (browse) ----------------------------------------------------


class MemoryItem(BaseModel):
    id: str
    tier: str
    content: str
    facts: Any | None = None
    metadata: Any | None = None
    importance: float
    created_at: str
    accessed_at: str
    access_count: int


class MemoriesListResponse(BaseModel):
    memories: list[MemoryItem]


@_router.get("/memories")
async def list_memories(
    tier: str | None = None,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> MemoriesListResponse:
    daemon = _get_daemon()
    repo = MemoryRepository()
    async with daemon.session_factory() as session:
        if q:
            rows = await repo.search_text(
                session, q, tier=tier, limit=limit, offset=offset
            )
        else:
            rows = await repo.list_all(
                session, tier=tier, limit=limit, offset=offset
            )

    return MemoriesListResponse(
        memories=[
            MemoryItem(
                id=str(r.id),
                tier=r.tier,
                content=r.content,
                facts=r.facts,
                metadata=r.metadata_,
                importance=r.importance,
                created_at=r.created_at.isoformat(),
                accessed_at=r.accessed_at.isoformat(),
                access_count=r.access_count,
            )
            for r in rows
        ]
    )


class CountsResponse(BaseModel):
    counts: dict[str, int]
    total: int


@_router.get("/memories/counts")
async def memory_counts() -> CountsResponse:
    daemon = _get_daemon()
    repo = MemoryRepository()
    async with daemon.session_factory() as session:
        counts = await repo.count_by_tier(session)
    return CountsResponse(counts=counts, total=sum(counts.values()))


# -- Delete memory ---------------------------------------------------------


class DeleteMemoryResponse(BaseModel):
    deleted: bool
    id: str


@_router.delete("/memories/{memory_id}")
async def delete_memory(memory_id: UUID) -> DeleteMemoryResponse:
    """Delete a memory and any associated graph edges."""
    daemon = _get_daemon()
    mem_repo = MemoryRepository()
    edge_repo = EdgeRepository()

    async with daemon.session_factory() as session:
        async with session.begin():
            existing = await mem_repo.get_by_id(session, memory_id)
            if existing is None:
                raise HTTPException(status_code=404, detail="Memory not found")

            await edge_repo.delete_edges_for(session, memory_id)
            await mem_repo.delete_by_id(session, memory_id)

    return DeleteMemoryResponse(deleted=True, id=str(memory_id))


# -- Graph -----------------------------------------------------------------


class GraphNode(BaseModel):
    id: str
    label: str
    tier: str
    importance: float
    metadata: Any | None = None


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


@_router.get("/graph")
async def knowledge_graph(limit: int = 200) -> GraphResponse:
    """Return semantic nodes and edges for graph visualization."""
    daemon = _get_daemon()
    mem_repo = MemoryRepository()
    edge_repo = EdgeRepository()
    async with daemon.session_factory() as session:
        nodes_raw = await mem_repo.list_all(
            session, tier="semantic", limit=limit
        )
        edges_raw = await edge_repo.get_all_edges(session, limit=limit * 3)

    node_ids = {str(n.id) for n in nodes_raw}
    nodes = [
        GraphNode(
            id=str(n.id),
            label=_extract_label(n),
            tier=n.tier,
            importance=n.importance,
            metadata=n.metadata_,
        )
        for n in nodes_raw
    ]
    edges = [
        GraphEdge(
            source=str(e.source_id),
            target=str(e.target_id),
            relation=e.relation,
            weight=e.weight,
        )
        for e in edges_raw
        if str(e.source_id) in node_ids and str(e.target_id) in node_ids
    ]
    return GraphResponse(nodes=nodes, edges=edges)


def _extract_label(row: Any) -> str:
    """Pull a human-readable label from metadata, falling back to content prefix."""
    if row.metadata_:
        for key in ("label", "title", "name"):
            if key in row.metadata_:
                return str(row.metadata_[key])
    return row.content[:60]
