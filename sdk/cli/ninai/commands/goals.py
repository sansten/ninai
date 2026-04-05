"""Goals commands."""

from __future__ import annotations

import json

import typer

from ..client import CliClient

goals_app = typer.Typer(help="Goal management")


@goals_app.command("list")
def list_goals(
    ctx: typer.Context,
    status: str | None = typer.Option(None, "--status", help="Optional status filter"),
) -> None:
    client = CliClient(ctx.obj["config"])
    try:
        params = {"status": status} if status else None
        data = client.get("/goals", params=params)
        typer.echo(json.dumps(data, indent=2))
    finally:
        client.close()
