# Stage 1: build wheels
FROM python:3.11-slim AS builder

WORKDIR /build

RUN pip install --no-cache-dir hatchling

COPY pyproject.toml README.md ./
COPY src/ src/
COPY alembic.ini ./
COPY alembic/ alembic/

RUN pip wheel --no-deps --wheel-dir /wheels .

# Stage 2: runtime
FROM python:3.11-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash smriti

WORKDIR /app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels

COPY alembic.ini ./
COPY alembic/ alembic/
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER smriti

EXPOSE 9898

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:9898/v1/health || exit 1

ENTRYPOINT ["/entrypoint.sh"]
