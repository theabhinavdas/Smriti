#!/usr/bin/env bash
set -euo pipefail

echo "[smriti] Running database migrations..."
alembic upgrade head

echo "[smriti] Starting memoryd on ${SMRITI_DAEMON_HOST:-0.0.0.0}:${SMRITI_DAEMON_PORT:-9898}"
exec smriti serve \
    --host "${SMRITI_DAEMON_HOST:-0.0.0.0}" \
    --port "${SMRITI_DAEMON_PORT:-9898}"
