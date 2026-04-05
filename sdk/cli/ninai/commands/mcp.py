"""MCP commands."""

from __future__ import annotations

import json

import typer

from ..client import CliClient

mcp_app = typer.Typer(help="MCP integration commands")


@mcp_app.command("status")
def mcp_status(ctx: typer.Context) -> None:
    client = CliClient(ctx.obj["config"])
    try:
        data = client.get("/mcp/status")
        typer.echo(json.dumps(data, indent=2))
    finally:
        client.close()
