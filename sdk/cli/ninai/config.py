"""Config loading for the Ninai CLI."""

from __future__ import annotations

import os
from pathlib import Path

from .client import CliConfig


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def load_config() -> CliConfig:
    return CliConfig(
        api_key=_env_or_default("NINAI_API_KEY", ""),
        org_id=_env_or_default("NINAI_ORG_ID", ""),
        base_url=_env_or_default("NINAI_BASE_URL", "http://localhost:8000/api/v1"),
    )


def write_env_template(path: Path) -> None:
    lines = [
        "NINAI_BASE_URL=http://localhost:8000/api/v1",
        "NINAI_API_KEY=",
        "NINAI_ORG_ID=",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
