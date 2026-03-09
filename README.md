# Smriti

> *smriti* (स्मृति) — Sanskrit for "memory", "remembrance"

A multi-tier memory system for LLMs inspired by human cognitive memory. Smriti gives language models persistent, cross-session memory by observing user activity, extracting salient information, and assembling relevant context on every query.

## Architecture

- **Tier 1 — Buffer**: Ring buffer of recent conversation turns
- **Tier 2 — Working Memory**: Session-level key-value facts (lives in Valkey)
- **Tier 3 — Episodic**: Compressed past conversations (Postgres + pgvector)
- **Tier 4 — Semantic**: Long-term knowledge graph of distilled facts

## Requirements

- Python 3.11+
- Docker and Docker Compose v2

## Quick Start

```bash
# Start infrastructure
docker compose up -d

# Install in development mode
python -m pip install -e ".[dev]"

# Run tests
pytest

# Launch the CLI
smriti status
```

## License

MIT
