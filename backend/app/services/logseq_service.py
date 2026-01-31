"""Logseq integration service.

Implements:
- Logseq-friendly Markdown export
- Simple graph representation (nodes/edges) for visualization
- Admin-only write-to-disk helper (API enforces role)

Security notes:
- Any DB reads must occur via existing services that enforce permission
  checks and respect PostgreSQL RLS.
- This module keeps the Markdown/graph rendering pure so it can be unit
  tested without a live database.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class ExportableMemory:
    """Minimal shape required for export/graph rendering."""

    id: str
    title: Optional[str]
    content_preview: str
    created_at: Optional[datetime]
    scope: Optional[str]
    classification: Optional[str]
    tags: list[str]
    entities: dict[str, Any]


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).isoformat()
    return dt.astimezone(timezone.utc).isoformat()


def _safe_label(value: Optional[str], fallback: str) -> str:
    v = (value or "").strip()
    return v if v else fallback


def render_logseq_markdown(
    memories: Iterable[ExportableMemory],
    *,
    title: str = "NinaivOS Export",
) -> tuple[str, int]:
    """Render a Logseq-friendly Markdown document.

    Format is intentionally simple and compatible with Logseq's block model.

    Returns:
        (markdown, item_count)
    """

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    exported_at = datetime.now(timezone.utc).isoformat()
    lines.append(f"- exported_at:: {exported_at}")

    count = 0
    for m in memories:
        count += 1
        heading = _safe_label(m.title, f"Memory {m.id}")
        lines.append("")
        lines.append(f"- ## {heading}")
        lines.append(f"  id:: {m.id}")
        if m.created_at is not None:
            lines.append(f"  created_at:: {_iso(m.created_at)}")
        if m.scope:
            lines.append(f"  scope:: {m.scope}")
        if m.classification:
            lines.append(f"  classification:: {m.classification}")
        if m.tags:
            lines.append(f"  tags:: {', '.join(m.tags)}")
        if m.entities:
            # Keep this compact: key=value; value can be list/str
            entity_parts: list[str] = []
            for k, v in m.entities.items():
                if v is None:
                    continue
                if isinstance(v, list):
                    vv = ", ".join(str(x) for x in v if x is not None)
                else:
                    vv = str(v)
                vv = vv.strip()
                if vv:
                    entity_parts.append(f"{k}={vv}")
            if entity_parts:
                lines.append(f"  entities:: {'; '.join(entity_parts)}")

        preview = (m.content_preview or "").strip()
        if preview:
            lines.append(f"  - {preview}")

    lines.append("")
    return "\n".join(lines), count


def render_logseq_memory_page(
    memory: ExportableMemory,
    *,
    organization_id: Optional[str] = None,
    outgoing_memory_ids: Optional[Iterable[str]] = None,
    backlink_memory_ids: Optional[Iterable[str]] = None,
) -> str:
    """Render a single Logseq page (one memory == one page).

    This is the Zettelkasten-style export mode: atomic notes with explicit links.
    """

    title = _safe_label(memory.title, f"Memory {memory.id}")
    tags = [str(t).strip() for t in (memory.tags or []) if str(t).strip()]
    out_ids = [str(x).strip() for x in (outgoing_memory_ids or []) if str(x).strip()]
    back_ids = [str(x).strip() for x in (backlink_memory_ids or []) if str(x).strip()]

    lines: list[str] = []
    # YAML frontmatter (as per LOGSEQ_INTEGRATION_SPEC.md)
    lines.append("---")
    lines.append(f"id: {memory.id}")
    if organization_id:
        lines.append(f"org_id: {organization_id}")
    if memory.scope:
        lines.append(f"scope: {memory.scope}")
    if memory.classification:
        lines.append(f"classification: {memory.classification}")
    if memory.created_at is not None:
        lines.append(f"created_at: {_iso(memory.created_at)}")
    if tags:
        # JSON is valid YAML and avoids escaping issues.
        lines.append(f"tags: {json.dumps(tags, ensure_ascii=False)}")
    lines.append("---")
    lines.append("")

    # Logseq content as blocks
    lines.append(f"- # {title}")

    preview = (memory.content_preview or "").strip()
    if preview:
        lines.append("  - Summary")
        lines.append(f"    - {preview}")

    if tags:
        lines.append("  - Tags")
        for t in tags:
            lines.append(f"    - [[{t}]]")

    if memory.entities:
        lines.append("  - Entities")
        for k, v in memory.entities.items():
            if v is None:
                continue
            values = v if isinstance(v, list) else [v]
            clean = [str(x).strip() for x in values if x is not None and str(x).strip()]
            if not clean:
                continue
            lines.append(f"    - {k}:: {', '.join(clean)}")

    if out_ids:
        lines.append("  - Related")
        for target_id in sorted(set(out_ids)):
            lines.append(f"    - [[mem_{target_id}]]")

    if back_ids:
        lines.append("  - Backlinks")
        for source_id in sorted(set(back_ids)):
            lines.append(f"    - [[mem_{source_id}]]")

    lines.append("")
    return "\n".join(lines)


def build_logseq_graph(memories: Iterable[ExportableMemory]) -> dict[str, list[dict[str, Any]]]:
    """Build a simple graph: memory nodes connected to tag and entity nodes."""

    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def upsert_node(node_id: str, node_type: str, label: str, data: Optional[dict[str, Any]] = None):
        if node_id in nodes:
            return
        nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "label": label,
            "data": data or {},
        }

    for m in memories:
        mem_node_id = f"memory:{m.id}"
        upsert_node(mem_node_id, "memory", _safe_label(m.title, m.id), {
            "memory_id": m.id,
            "scope": m.scope,
            "classification": m.classification,
        })

        for tag in (m.tags or []):
            t = str(tag).strip()
            if not t:
                continue
            tag_id = f"tag:{t.lower()}"
            upsert_node(tag_id, "tag", t)
            edges.append({"source": mem_node_id, "target": tag_id, "type": "has_tag", "data": {}})

        for key, value in (m.entities or {}).items():
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                v = str(item).strip()
                if not v:
                    continue
                ent_id = f"entity:{key}:{v.lower()}"
                upsert_node(ent_id, "entity", v, {"entity_type": key})
                edges.append({"source": mem_node_id, "target": ent_id, "type": "mentions", "data": {"entity_type": key}})

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
    }


def write_markdown_export(
    markdown: str,
    *,
    export_dir: Path,
    filename: str,
) -> tuple[str, int]:
    """Write an export file to disk.

    Returns:
        (relative_path, bytes_written)
    """

    export_dir.mkdir(parents=True, exist_ok=True)

    # Basic Windows-safe filename sanitation
    safe = "".join(ch for ch in filename if ch not in '<>:"/\\|?*')
    safe = safe.strip().rstrip(".")
    if not safe:
        safe = "logseq_export.md"
    if not safe.lower().endswith(".md"):
        safe += ".md"

    out_path = export_dir / safe
    data = markdown.encode("utf-8")
    out_path.write_bytes(data)

    return str(out_path), len(data)


def write_logseq_vault_pages(
    *,
    export_dir: Path,
    organization_id: str,
    memories: Iterable[ExportableMemory],
    outgoing_links: dict[str, set[str]] | None = None,
    backlinks: dict[str, set[str]] | None = None,
    stamp: Optional[str] = None,
) -> tuple[str, int]:
    """Write a Logseq vault structure to disk.

    Returns:
        (run_meta_absolute_path, total_bytes_written)

    Notes:
    - Writes per-memory pages to {export_dir}/pages/mem_<id>.md
    - Writes a run metadata marker to {export_dir}/export_runs/export_run_<stamp>.json
    - Does not delete old pages; repeated exports overwrite pages for memories exported.
    """

    export_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = export_dir / "pages"
    journals_dir = export_dir / "journals"
    assets_dir = export_dir / "assets"
    runs_dir = export_dir / "export_runs"
    pages_dir.mkdir(parents=True, exist_ok=True)
    journals_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = 0
    exported_ids: list[str] = []

    for m in memories:
        exported_ids.append(m.id)
        filename = f"mem_{m.id}.md"
        out_path = pages_dir / filename
        page_md = render_logseq_memory_page(
            m,
            organization_id=organization_id,
            outgoing_memory_ids=sorted((outgoing_links or {}).get(m.id, set())),
            backlink_memory_ids=sorted((backlinks or {}).get(m.id, set())),
        )
        data = page_md.encode("utf-8")
        out_path.write_bytes(data)
        total_bytes += len(data)

    # Minimal README to help humans open the vault.
    readme = (
        "# NinaivOS Logseq Vault\n\n"
        "Open this folder in Logseq. Pages live under `pages/`.\n"
        "This vault is export-only; Postgres/Qdrant remain the source of truth.\n"
    )
    readme_path = export_dir / "README.md"
    total_bytes += len(readme.encode("utf-8"))
    readme_path.write_text(readme, encoding="utf-8")

    export_meta = {
        "organization_id": organization_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "item_count": len(exported_ids),
        "mode": "vault_pages",
    }
    meta_path = export_dir / "export_meta.json"
    meta_bytes = json.dumps(export_meta, ensure_ascii=False, indent=2).encode("utf-8")
    meta_path.write_bytes(meta_bytes)
    total_bytes += len(meta_bytes)

    stamp = stamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_meta = {
        **export_meta,
        "export_run": stamp,
        "exported_memory_ids": exported_ids,
    }
    run_path = runs_dir / f"export_run_{stamp}.json"
    run_bytes = json.dumps(run_meta, ensure_ascii=False, indent=2).encode("utf-8")
    run_path.write_bytes(run_bytes)
    total_bytes += len(run_bytes)

    return str(run_path), int(total_bytes)
