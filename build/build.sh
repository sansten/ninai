#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
BOOK_DIR="$ROOT_DIR/book"
OUT_DIR="$ROOT_DIR/build/out"
mkdir -p "$OUT_DIR"

# Validate chapters
python "$BOOK_DIR/tools/critic_validate.py" || echo "Validator reported issues (placeholders may remain)."

# Pre-render mermaid diagrams for PDF (creates build/tmp/chapters_pdf)
python "$BOOK_DIR/tools/render_mermaid.py"

# Build HTML (mermaid rendered client-side)
pandoc \
  --standalone \
  --toc \
  --toc-depth=2 \
  --metadata-file "$ROOT_DIR/pandoc/pandoc.yaml" \
  -o "$OUT_DIR/book.html" \
  "$BOOK_DIR/index.md" \
  $(printf "%s " "$BOOK_DIR"/chapters/ch*.md)

# Build PDF (basic, diagrams may need mermaid-cli pre-render)
pandoc \
  --pdf-engine=wkhtmltopdf \
  --metadata-file "$ROOT_DIR/pandoc/pandoc.yaml" \
  -o "$OUT_DIR/book.pdf" \
  "$BOOK_DIR/index.md" \
  $(printf "%s " "$ROOT_DIR/build/tmp/chapters_pdf"/ch*.md)

echo "Built: $OUT_DIR/book.html and $OUT_DIR/book.pdf"
