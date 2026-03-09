# Smriti

> *smriti* (स्मृति) — Sanskrit for "memory", "remembrance"

Persistent, cross-session memory for LLMs. Smriti observes your activity (terminal, browser, IDE), extracts what matters, and assembles relevant context every time an LLM needs it.

Instead of starting every conversation from scratch, your LLM remembers what you worked on, what you prefer, and what you know.

## How it works

Smriti uses a four-tier memory architecture inspired by human cognition:

| Tier | Name | What it stores | Backed by |
|------|------|---------------|-----------|
| 1 | **Buffer** | Recent conversation turns (verbatim) | In-memory ring buffer |
| 2 | **Working** | Active session context (goals, entities, decisions) | Valkey with TTL |
| 3 | **Episodic** | Compressed past sessions with embeddings | Postgres + pgvector |
| 4 | **Semantic** | Long-term knowledge graph (skills, preferences, relationships) | Postgres + pgvector |

Memories flow upward over time: raw events are filtered for salience, extracted into episodes, and periodically consolidated into durable semantic facts -- much like how human memory consolidates during sleep.

## Quick start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose v2
- An [OpenRouter](https://openrouter.ai/) API key (provides access to all LLM and embedding models through a single key)

### 1. Clone and install

```bash
git clone https://github.com/your-org/smriti.git
cd smriti
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set your OpenRouter API key:
#   SMRITI_MODEL_API_KEY=sk-or-...
```

### 3. Start infrastructure

```bash
docker compose up -d
```

This starts Postgres (with pgvector) and Valkey locally. Both bind to localhost only.

### 4. Run migrations

```bash
alembic upgrade head
```

### 5. Start the daemon

```bash
smriti serve
```

The daemon starts on `http://127.0.0.1:9898` and begins consuming events, running the ingestion pipeline, and serving the API.

### 6. Use the CLI

```bash
# Check system health
smriti status

# Search your memories
smriti search "CORS debugging"

# Search with filters
smriti search -k 5 --tier semantic "Python projects"
```

## Production deployment

Deploy the full stack (daemon + Postgres + Valkey) with a single command:

```bash
SMRITI_MODEL_API_KEY=sk-or-... docker compose -f docker-compose.prod.yml up -d
```

This builds the daemon image, runs migrations on startup, and starts all three services. Postgres and Valkey are internal-only (no published ports). The daemon binds to `127.0.0.1:9898`.

## API

The daemon exposes a local HTTP API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/v1/health` | Liveness check with uptime |
| `POST` | `/v1/events` | Push events (for collectors) |
| `POST` | `/v1/search` | Semantic memory search |
| `GET` | `/v1/stats` | Pipeline statistics |

### Push an event

```bash
curl -X POST http://localhost:9898/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "events": [{
      "source": "terminal",
      "event_type": "command",
      "raw_content": "docker compose up -d"
    }]
  }'
```

### Search memories

```bash
curl -X POST http://localhost:9898/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "CORS debugging", "top_k": 5}'
```

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SMRITI_MODEL_API_KEY` | (required) | OpenRouter API key |
| `SMRITI_MODEL_EMBEDDING_MODEL` | `openai/text-embedding-3-small` | Embedding model |
| `SMRITI_MODEL_EXTRACTION_MODEL` | `anthropic/claude-3.5-haiku` | LLM for extraction |
| `SMRITI_PG_HOST` | `127.0.0.1` | Postgres host |
| `SMRITI_PG_PORT` | `5432` | Postgres port |
| `SMRITI_PG_PASSWORD` | `smriti` | Postgres password |
| `SMRITI_VALKEY_HOST` | `127.0.0.1` | Valkey host |
| `SMRITI_DAEMON_PORT` | `9898` | API port |
| `SMRITI_DAEMON_CONSOLIDATION_INTERVAL_MINUTES` | `30` | Consolidation frequency |

## Development

```bash
# Run the full test suite (needs Docker services running)
pytest

# Run only unit tests (no infrastructure needed)
pytest tests/test_models.py tests/test_config.py tests/test_buffer.py \
       tests/test_provider.py tests/test_ingestion.py tests/test_daemon.py \
       tests/test_context.py tests/test_cli.py tests/test_consolidation.py

# Run integration tests (needs Postgres + Valkey)
pytest tests/test_db.py tests/test_event_bus.py tests/test_working_memory.py \
       tests/test_episodic.py tests/test_semantic.py tests/test_retrieval.py

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Project structure

```
src/smriti/
├── models/          # Pydantic data models (events, memory tiers)
├── db/              # SQLAlchemy tables, engine, repositories
├── memory/          # Tier implementations (buffer, working, episodic, semantic)
├── ingestion/       # Pipeline: salience filter → extractor → tier router
├── config.py        # Environment-based configuration
├── event_bus.py     # Valkey Streams pub/sub
├── provider.py      # OpenRouter LLM/embedding client
├── retrieval.py     # Hybrid search + multi-signal ranking
├── context.py       # Tier-adaptive rendering + token budget assembly
├── consolidation.py # Episodic → semantic promotion (sleep-like)
├── daemon.py        # Pipeline orchestrator (consume loop)
├── api.py           # FastAPI HTTP endpoints
└── cli.py           # Click CLI (serve, status, search)
```

## License

MIT
