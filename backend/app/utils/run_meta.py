from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_get_git_sha(repo_root: str) -> Optional[str]:
    """Best-effort git SHA; returns None if git not available or not a repo."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception:
        return None

    if completed.returncode != 0:
        return None
    sha = (completed.stdout or "").strip()
    return sha or None


def collect_run_meta(*, base_url: str, dataset_path: Optional[str], resume_jsonl_path: Optional[str], repo_root: str) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "base_url": base_url,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
        "git_sha": try_get_git_sha(repo_root),
    }

    if dataset_path:
        try:
            meta["dataset_path"] = dataset_path
            meta["dataset_sha256"] = sha256_file(dataset_path)
        except Exception:
            meta["dataset_sha256"] = None

    if resume_jsonl_path:
        try:
            meta["resume_from_jsonl"] = resume_jsonl_path
            meta["resume_jsonl_sha256"] = sha256_file(resume_jsonl_path)
        except Exception:
            meta["resume_jsonl_sha256"] = None

    # Capture a few env vars that affect behavior, without leaking secrets.
    for k in ["EVAL_API_BASE_URL"]:
        if k in os.environ:
            meta[f"env_{k}"] = os.environ.get(k)

    return meta
