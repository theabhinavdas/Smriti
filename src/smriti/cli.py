"""Smriti CLI -- talks to the memoryd HTTP API."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import click
import httpx


DEFAULT_DAEMON_URL = "http://127.0.0.1:9898"


def _base_url(ctx: click.Context) -> str:
    return ctx.obj.get("daemon_url", DEFAULT_DAEMON_URL)


def _request(method: str, url: str, **kwargs: Any) -> httpx.Response:
    """Synchronous HTTP helper that handles connection errors."""
    try:
        return httpx.request(method, url, timeout=10.0, **kwargs)
    except httpx.ConnectError:
        click.echo("Error: cannot connect to memoryd. Is it running?", err=True)
        click.echo(f"  Tried: {url}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Top-level group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="smriti")
@click.option(
    "--url",
    envvar="SMRITI_DAEMON_URL",
    default=DEFAULT_DAEMON_URL,
    help="Daemon base URL.",
)
@click.pass_context
def main(ctx: click.Context, url: str) -> None:
    """Smriti -- memory for LLMs."""
    ctx.ensure_object(dict)
    ctx.obj["daemon_url"] = url


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@main.command()
@click.option("--host", default="127.0.0.1", help="Bind address.")
@click.option("--port", default=9898, type=int, help="Bind port.")
def serve(host: str, port: int) -> None:
    """Start the memoryd daemon."""
    import uvicorn

    from smriti.api import create_app
    from smriti.config import DaemonConfig, load_settings

    settings = load_settings()
    settings.daemon = DaemonConfig(host=host, port=port)
    app = create_app(settings=settings)

    click.echo(f"Starting memoryd on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level=settings.log_level)


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


def _format_uptime(seconds: float) -> str:
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    parts.append(f"{minutes}m")
    return " ".join(parts)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show system health and pipeline stats."""
    base = _base_url(ctx)

    health_resp = _request("GET", f"{base}/v1/health")
    if health_resp.status_code != 200:
        click.echo(f"memoryd: unhealthy (HTTP {health_resp.status_code})")
        sys.exit(1)

    health = health_resp.json()
    uptime = _format_uptime(health["uptime_seconds"])

    stats_resp = _request("GET", f"{base}/v1/stats")
    stats = stats_resp.json() if stats_resp.status_code == 200 else {}

    click.echo(f"memoryd: healthy (uptime {uptime})")
    if stats:
        click.echo(f"events consumed: {stats['events_consumed']}")
        click.echo(f"events filtered: {stats['events_filtered']}")
        click.echo(f"memories created: {stats['memories_created']}")
        click.echo(f"batches processed: {stats['batches_processed']}")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@main.command()
@click.argument("query")
@click.option("-k", "--top-k", default=10, type=int, help="Max results.")
@click.option("--tier", default=None, help="Filter by tier (episodic, semantic).")
@click.pass_context
def search(ctx: click.Context, query: str, top_k: int, tier: str | None) -> None:
    """Search memories by semantic similarity."""
    base = _base_url(ctx)
    payload: dict[str, Any] = {"query": query, "top_k": top_k}
    if tier:
        payload["tier"] = tier

    resp = _request("POST", f"{base}/v1/search", json=payload)
    if resp.status_code != 200:
        click.echo(f"Search failed: {resp.text}", err=True)
        sys.exit(1)

    results = resp.json()["results"]
    if not results:
        click.echo("No memories found.")
        return

    for r in results:
        created = r["created_at"][:10]
        click.echo(
            f"[{created}, {r['tier']}, {r['importance']:.2f}] "
            f"{r['content']}"
        )
