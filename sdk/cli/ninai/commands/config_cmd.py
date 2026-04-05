"""Configuration commands."""

from __future__ import annotations

from pathlib import Path

import typer

from ..config import load_config, write_env_template

config_app = typer.Typer(help="CLI configuration helpers")


@config_app.command("show")
def show_config() -> None:
    config = load_config()
    redacted = "***" if config.api_key else ""
    typer.echo(f"base_url={config.base_url}")
    typer.echo(f"org_id={config.org_id}")
    typer.echo(f"api_key={redacted}")


@config_app.command("init")
def init_config(
    output: Path = typer.Option(Path(".env.ninai"), "--output", "-o", help="Output env file"),
) -> None:
    if output.exists():
        raise typer.BadParameter(f"{output} already exists")
    write_env_template(output)
    typer.echo(f"wrote {output}")
