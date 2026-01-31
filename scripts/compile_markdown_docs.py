"""Compile all .md docs into two consolidated markdown files.

Goal:
- Keep original markdown content as-is (no rewriting), only wrap per-file with separators.
- Deduplicate exact duplicate files (by normalized content hash).
- Split into two outputs:
  - requirements: everything that is not clearly testing/perf/e2e oriented
  - testing: test/e2e/perf/benchmark/how-to-run guides

Usage (repo root):
  python scripts/compile_markdown_docs.py

Outputs:
  - DOCS_REQUIREMENTS_CONSOLIDATED.md
  - DOCS_TESTING_CONSOLIDATED.md

This is intentionally heuristic; adjust CLASSIFIERS if you want different bucketing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = REPO_ROOT / "md"
OUTPUT_REQUIREMENTS = OUTPUT_DIR / "DOCS_REQUIREMENTS_CONSOLIDATED.md"
OUTPUT_TESTING = OUTPUT_DIR / "DOCS_TESTING_CONSOLIDATED.md"
OUTPUT_MANIFEST = OUTPUT_DIR / "DOCS_MERGED_SOURCES_MANIFEST.txt"

# Directories we never want to traverse.
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
    # Keep the book as-is; do not merge it.
    "book",
    # Local folders for moved docs.
    "md",
    "md_sources",
    "docs_archive",
}

# Filenames to skip (including our generated outputs).
SKIP_FILE_NAMES = {
    OUTPUT_REQUIREMENTS.name,
    OUTPUT_TESTING.name,
    OUTPUT_MANIFEST.name,
}


TESTING_PATH_MARKERS = (
    "/tests/",
    "backend/tests/",
    "frontend/e2e",
    "/e2e",
    "e2e_",
    "e2e-",
    "performance-tests",
    "benchmark",
    "test_results",
    "test_",
    "_test",
)

TESTING_CONTENT_KEYWORDS = (
    # Strong, specific signals (avoid matching broad roadmap/feature docs).
    "pytest",
    "python -m pytest",
    "run_postgres_tests",
    "run_perf_tests",
    "e2e",
    "playbook",
    "performance-tests",
    "benchmark",
    "load test",
)


@dataclass(frozen=True)
class DocFile:
    rel_path: str
    abs_path: Path
    content: str
    content_hash: str


def _iter_md_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.md"):
        # Skip logic should be relative to the chosen scan root.
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        if path.name in SKIP_FILE_NAMES:
            continue
        yield path


def _read_text(path: Path) -> str:
    # Be tolerant of odd encodings in historical docs.
    return path.read_text(encoding="utf-8", errors="replace")


def _normalize_for_hash(text: str) -> str:
    # Normalize line endings and trailing whitespace so duplicates compare stable.
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return "\n".join(lines).strip() + "\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_testing_doc(rel_path: str, content: str) -> bool:
    rel_lower = rel_path.lower()

    # Path-based detection (more precise than raw 'test' substring).
    if any(marker in rel_lower for marker in TESTING_PATH_MARKERS):
        return True

    # Filename-based detection.
    name = Path(rel_lower).name
    if name.startswith("test_") or name.endswith("_test.md"):
        return True
    if any(token in name for token in ("e2e", "perf", "benchmark")):
        return True

    # Lightweight content scan: first ~3000 chars.
    chunk = content[:3000].lower()
    return any(k in chunk for k in TESTING_CONTENT_KEYWORDS)


def _render_header(title: str, bucket_description: str, source_count: int, unique_count: int) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    return (
        f"# {title}\n\n"
        f"Generated: {now} (UTC)\n\n"
        f"{bucket_description}\n\n"
        f"- Source .md files discovered: {source_count}\n"
        f"- Unique file contents included: {unique_count}\n\n"
        "---\n\n"
        "## Table of Contents\n\n"
    )


def _anchor_for(text: str) -> str:
    safe = text.strip().lower().replace(" ", "-")
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in safe)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "doc"


def _extract_title(content: str) -> str | None:
    # Prefer the first markdown heading as the doc title.
    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip() or None
        # Some docs start with a bold title.
        if s.startswith("**") and s.endswith("**") and len(s) > 4:
            return s.strip("*").strip() or None
    return None


def _render_index(items: list[DocFile], titles_by_rel: dict[str, str], anchors_by_rel: dict[str, str]) -> str:
    lines: list[str] = []
    for doc in items:
        title = titles_by_rel.get(doc.rel_path, "Untitled")
        anchor = anchors_by_rel.get(doc.rel_path, "doc")
        lines.append(f"- [{title}](#{anchor})")
    return "\n".join(lines) + "\n\n---\n\n"


def _render_doc(doc: DocFile, title: str, anchor: str) -> str:
    # Wrap each doc in a <details> block to keep the consolidated file navigable.
    # Do NOT mention source paths/filenames in visible text (so sources can be deleted).
    return (
        f"<a id=\"{anchor}\"></a>\n\n"
        f"## {title}\n\n"
        "<details>\n"
        "<summary>Show content</summary>\n\n"
        f"{doc.content.rstrip()}\n\n"
        "</details>\n\n"
        "---\n\n"
    )


def _render_duplicate(title: str, anchor: str, original_title: str, original_anchor: str) -> str:
    return (
        f"<a id=\"{anchor}\"></a>\n\n"
        f"## {title}\n\n"
        f"Duplicate of [{original_title}](#{original_anchor}). Content omitted.\n\n"
        "---\n\n"
    )


def compile_docs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # If sources have been moved into md/sources, compile from there.
    sources_root = OUTPUT_DIR / "sources"
    scan_root = sources_root if sources_root.exists() else REPO_ROOT

    md_paths = sorted(_iter_md_files(scan_root), key=lambda p: str(p).lower())

    docs: list[DocFile] = []
    for path in md_paths:
        rel_path = path.relative_to(scan_root).as_posix()
        content = _read_text(path)
        normalized = _normalize_for_hash(content)
        docs.append(
            DocFile(
                rel_path=rel_path,
                abs_path=path,
                content=content,
                content_hash=_sha256(normalized),
            )
        )

    # Determine duplicates across the whole repo by content hash.
    first_by_hash: dict[str, DocFile] = {}
    duplicates: dict[str, str] = {}  # rel_path -> original_rel_path
    unique_docs: list[DocFile] = []

    for doc in docs:
        if doc.content_hash in first_by_hash:
            duplicates[doc.rel_path] = first_by_hash[doc.content_hash].rel_path
            continue
        first_by_hash[doc.content_hash] = doc
        unique_docs.append(doc)

    # Bucket docs (including duplicates, so “all files” appear in the index).
    requirements_items: list[DocFile] = []
    testing_items: list[DocFile] = []

    for doc in docs:
        if _is_testing_doc(doc.rel_path, doc.content):
            testing_items.append(doc)
        else:
            requirements_items.append(doc)

    # Build titles/anchors per bucket (titles are derived from in-doc headings).
    def build_titles(items: list[DocFile]) -> tuple[dict[str, str], dict[str, str]]:
        titles_by_rel: dict[str, str] = {}
        anchors_by_rel: dict[str, str] = {}
        used_titles: dict[str, int] = {}
        used_anchors: dict[str, int] = {}

        for i, doc in enumerate(items, start=1):
            title = _extract_title(doc.content) or f"Document {i}"
            # Ensure uniqueness within the file.
            count = used_titles.get(title, 0)
            used_titles[title] = count + 1
            if count:
                title = f"{title} ({count + 1})"

            anchor = _anchor_for(title)
            ac = used_anchors.get(anchor, 0)
            used_anchors[anchor] = ac + 1
            if ac:
                anchor = f"{anchor}-{ac + 1}"

            titles_by_rel[doc.rel_path] = title
            anchors_by_rel[doc.rel_path] = anchor

        return titles_by_rel, anchors_by_rel

    # Render outputs.
    def write_output(out_path: Path, title: str, description: str, items: list[DocFile]) -> None:
        # Unique count is across included unique contents in this bucket.
        bucket_unique_hashes = {d.content_hash for d in items if d.rel_path not in duplicates}
        header = _render_header(title, description, source_count=len(items), unique_count=len(bucket_unique_hashes))

        titles_by_rel, anchors_by_rel = build_titles(items)
        index = _render_index(items, titles_by_rel, anchors_by_rel)

        parts: list[str] = [header, index]
        for doc in items:
            if doc.rel_path in duplicates:
                original_rel = duplicates[doc.rel_path]
                parts.append(
                    _render_duplicate(
                        title=titles_by_rel[doc.rel_path],
                        anchor=anchors_by_rel[doc.rel_path],
                        original_title=titles_by_rel.get(original_rel, "Original"),
                        original_anchor=anchors_by_rel.get(original_rel, "original"),
                    )
                )
            else:
                parts.append(
                    _render_doc(
                        doc,
                        title=titles_by_rel[doc.rel_path],
                        anchor=anchors_by_rel[doc.rel_path],
                    )
                )

        out_path.write_text("".join(parts), encoding="utf-8")

    write_output(
        OUTPUT_REQUIREMENTS,
        title="Ninai Consolidated Requirements (All Docs)",
        description=(
            "This file consolidates project documentation that is *not primarily testing-oriented*.\n"
            "Per-document content is included verbatim (wrapped only for navigation).\n"
            "The visible output does not reference the original source filenames/paths so the sources can be deleted.\n"
            "(The `book/` folder is excluded from merging.)"
        ),
        items=requirements_items,
    )

    write_output(
        OUTPUT_TESTING,
        title="Ninai Consolidated Testing & Validation (All Test Docs)",
        description=(
            "This file consolidates documentation related to testing, E2E, performance, benchmarks, and playbooks.\n"
            "Per-document content is included verbatim (wrapped only for navigation).\n"
            "The visible output does not reference the original source filenames/paths so the sources can be deleted.\n"
            "(The `book/` folder is excluded from merging.)"
        ),
        items=testing_items,
    )

    # Emit a manifest of every merged source .md (relative paths), so it can be deleted/moved later.
    manifest_paths = [d.rel_path for d in docs]
    OUTPUT_MANIFEST.write_text("\n".join(manifest_paths) + "\n", encoding="utf-8")

    print(f"Discovered .md files: {len(docs)}")
    print(f"Unique contents: {len(unique_docs)}")
    print(f"Duplicates: {len(duplicates)}")
    print(f"Wrote: {OUTPUT_REQUIREMENTS}")
    print(f"Wrote: {OUTPUT_TESTING}")
    print(f"Wrote: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    compile_docs()
