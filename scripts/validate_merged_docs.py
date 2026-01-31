"""Validate that merged markdown outputs contain all source markdown contents.

Checks every path in DOCS_MERGED_SOURCES_MANIFEST.txt and verifies that the
source file's content appears verbatim inside either DOCS_REQUIREMENTS_CONSOLIDATED.md
or DOCS_TESTING_CONSOLIDATED.md.

Run this BEFORE moving/deleting sources.

Usage:
  python scripts/validate_merged_docs.py
"""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "md" / "DOCS_MERGED_SOURCES_MANIFEST.txt"
REQ_OUT = REPO_ROOT / "md" / "DOCS_REQUIREMENTS_CONSOLIDATED.md"
TEST_OUT = REPO_ROOT / "md" / "DOCS_TESTING_CONSOLIDATED.md"
SOURCES_ROOT = REPO_ROOT / "md" / "sources"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _norm(text: str) -> str:
    # Normalize newlines to compare reliably across Windows/Linux.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def main() -> int:
    if not MANIFEST.exists():
        raise SystemExit(f"Missing manifest: {MANIFEST}. Run scripts/compile_markdown_docs.py first.")
    if not REQ_OUT.exists() or not TEST_OUT.exists():
        raise SystemExit("Missing consolidated outputs. Run scripts/compile_markdown_docs.py first.")

    req_text = _norm(_read_text(REQ_OUT))
    test_text = _norm(_read_text(TEST_OUT))

    rel_paths = [p.strip() for p in _read_text(MANIFEST).splitlines() if p.strip()]

    missing: list[str] = []
    found_in_req = 0
    found_in_test = 0

    for rel in rel_paths:
        src = SOURCES_ROOT / rel
        if not src.exists():
            missing.append(f"{rel} (source missing on disk)")
            continue

        src_text = _norm(_read_text(src)).rstrip()
        if not src_text:
            # Empty docs are fine; they won't materially affect merged outputs.
            continue

        if src_text in req_text:
            found_in_req += 1
            continue
        if src_text in test_text:
            found_in_test += 1
            continue

        missing.append(rel)

    total = len(rel_paths)
    ok = total - len(missing)

    print(f"Sources in manifest: {total}")
    print(f"Found in requirements: {found_in_req}")
    print(f"Found in testing: {found_in_test}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nMissing files:")
        for rel in missing:
            print(f"- {rel}")
        return 1

    print("\nOK: All manifest sources are present in one of the consolidated outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
