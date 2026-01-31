from __future__ import annotations

from pathlib import Path


def load_prompt_text(*parts: str) -> str:
    """Load a prompt template from app/prompts.

    Prompts are versioned plaintext files to keep them editable without code changes.
    """

    base_dir = Path(__file__).resolve().parents[2] / "prompts"
    path = base_dir.joinpath(*parts)
    return path.read_text(encoding="utf-8")
