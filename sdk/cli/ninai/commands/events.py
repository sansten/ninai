"""Event commands."""

from __future__ import annotations

import json

import typer

from ..client import CliClient

events_app = typer.Typer(help="Event stream operations")


@events_app.command("list")
def list_events(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-l", help="Number of events to fetch"),
) -> None:
    client = CliClient(ctx.obj["config"])
    try:
        data = client.get("/events", params={"limit": limit})
        typer.echo(json.dumps(data, indent=2))
    finally:
        client.close()
