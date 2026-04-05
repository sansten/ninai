"""Memory commands."""

from __future__ import annotations

import json

import typer

from ..client import CliClient

memory_app = typer.Typer(help="Memory operations")


@memory_app.command("search")
def memory_search(
    ctx: typer.Context,
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(10, "--limit", "-l", help="Result limit"),
) -> None:
    client = CliClient(ctx.obj["config"])
    try:
        data = client.get("/memory/search", params={"query": query, "k": limit})
        typer.echo(json.dumps(data, indent=2))
    finally:
        client.close()
