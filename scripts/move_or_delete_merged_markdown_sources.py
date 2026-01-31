"""Move (default) or delete merged markdown source files.

This uses the manifest created by scripts/compile_markdown_docs.py:
  DOCS_MERGED_SOURCES_MANIFEST.txt

By default it MOVES merged source .md files into an archive folder, so you can
recover them if needed.

Usage:
  python scripts/move_or_delete_merged_markdown_sources.py
  python scripts/move_or_delete_merged_markdown_sources.py --delete

Notes:
- The book/ folder is excluded from merging and is never touched.
- The consolidated docs and this manifest are never touched.
"""

from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "md" / "DOCS_MERGED_SOURCES_MANIFEST.txt"
SOURCES_ROOT = REPO_ROOT / "md" / "sources"
KEEP = {
    "DOCS_REQUIREMENTS_CONSOLIDATED.md",
    "DOCS_TESTING_CONSOLIDATED.md",
    "DOCS_MERGED_SOURCES_MANIFEST.txt",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Permanently delete source .md files instead of moving to an archive.",
    )
    args = parser.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(
            f"Manifest not found: {MANIFEST}. Run scripts/compile_markdown_docs.py first."
        )

    rel_paths = [p.strip() for p in MANIFEST.read_text(encoding="utf-8").splitlines() if p.strip()]

    # Stable local destination that is git-ignored.
    archive_root = REPO_ROOT / "md" / "sources"

    moved_or_deleted = 0
    skipped = 0

    for rel in rel_paths:
        # Safety: never touch anything under book/.
        if rel.startswith("book/"):
            skipped += 1
            continue

        # Safety: never touch the outputs.
        if Path(rel).name in KEEP:
            skipped += 1
            continue

        src = SOURCES_ROOT / rel
        if not src.exists():
            skipped += 1
            continue

        # Extra safety: only act on .md.
        if src.suffix.lower() != ".md":
            skipped += 1
            continue

        if args.delete:
            src.unlink()
        else:
            # Already in the destination.
            moved_or_deleted += 0
            skipped += 1
            continue

        moved_or_deleted += 1

    action = "deleted" if args.delete else f"no-op (sources already in {archive_root})"
    print(f"Processed source .md files: {moved_or_deleted} ({action})")
    print(f"Skipped: {skipped}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
