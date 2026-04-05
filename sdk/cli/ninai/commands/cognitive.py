"""Cognitive commands."""

from __future__ import annotations

import json

import typer

from ..client import CliClient

cognitive_app = typer.Typer(help="Cognitive engine operations")


@cognitive_app.command("decide")
def cognitive_decide(
    ctx: typer.Context,
    prompt: str = typer.Argument(..., help="Prompt for decision engine"),
) -> None:
    client = CliClient(ctx.obj["config"])
    try:
        data = client.post("/cognitive/decide", payload={"prompt": prompt})
        typer.echo(json.dumps(data, indent=2))
    finally:
        client.close()
