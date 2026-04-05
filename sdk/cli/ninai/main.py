"""Ninai CLI entrypoint."""

from __future__ import annotations

import typer

from .commands.cognitive import cognitive_app
from .commands.config_cmd import config_app
from .commands.events import events_app
from .commands.goals import goals_app
from .commands.mcp import mcp_app
from .commands.memory import memory_app
from .config import load_config

app = typer.Typer(help="Ninai Cognitive OS CLI")

app.add_typer(memory_app, name="memory")
app.add_typer(cognitive_app, name="cognitive")
app.add_typer(goals_app, name="goals")
app.add_typer(events_app, name="events")
app.add_typer(mcp_app, name="mcp")
app.add_typer(config_app, name="config")


@app.callback()
def app_callback(ctx: typer.Context) -> None:
    # Share config across command handlers through Typer context object.
    ctx.obj = {"config": load_config()}


if __name__ == "__main__":
    app()
