"""Smriti CLI entrypoint."""

import click


@click.group()
@click.version_option(package_name="smriti")
def main() -> None:
    """Smriti -- memory for LLMs."""


@main.command()
def status() -> None:
    """Show system health and stats."""
    click.echo("smriti is not yet running.")
