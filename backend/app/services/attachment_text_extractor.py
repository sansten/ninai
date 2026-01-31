"""Attachment text extraction for indexing.

MVP+: extract from text formats, PDFs, DOCX, and images (OCR).

All extraction is best-effort; if optional dependencies or system binaries
aren't available, we return empty text rather than failing the upload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import shutil


def _decode_text(data: bytes) -> str:
    return data.decode("utf-8", errors="ignore")


def extract_text_for_indexing(
    *,
    content_type: Optional[str],
    filename: str,
    data: bytes,
    max_chars: int,
) -> str:
    ct = (content_type or "").lower().split(";")[0].strip()
    name = (filename or "").lower()

    # Plain text-ish
    if ct.startswith("text/") or name.endswith((".txt", ".md", ".csv", ".log")):
        return _decode_text(data)[:max_chars]

    if ct in {"application/json", "application/xml"} or name.endswith((".json", ".xml")):
        return _decode_text(data)[:max_chars]

    # PDF (best-effort)
    if ct == "application/pdf" or name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return ""

        # pypdf can read from bytes via a BytesIO handle.
        import io

        try:
            reader = PdfReader(io.BytesIO(data))
            parts: list[str] = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                if txt:
                    parts.append(txt)
                if sum(len(p) for p in parts) >= max_chars:
                    break
            return "\n".join(parts)[:max_chars]
        except Exception:
            return ""

    # Not supported (images, office docs, etc.)
    return ""


def extract_text_for_indexing_from_file(
    *,
    content_type: Optional[str],
    filename: str,
    file_path: Path,
    max_chars: int,
    max_bytes: int,
    ocr_service_url: Optional[str] = None,
    ocr_timeout_seconds: float = 5.0,
) -> str:
    ct = (content_type or "").lower().split(";")[0].strip()
    name = (filename or "").lower()

    try:
        size = file_path.stat().st_size
        if size > max_bytes:
            return ""
    except Exception:
        return ""

    # Plain text-ish
    if ct.startswith("text/") or name.endswith((".txt", ".md", ".csv", ".log")):
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception:
            return ""

    if ct in {"application/json", "application/xml"} or name.endswith((".json", ".xml")):
        try:
            return file_path.read_text(encoding="utf-8", errors="ignore")[:max_chars]
        except Exception:
            return ""

    # PDF (best-effort)
    if ct == "application/pdf" or name.endswith(".pdf"):
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return ""

        try:
            reader = PdfReader(str(file_path))
            parts: list[str] = []
            for page in reader.pages:
                txt = page.extract_text() or ""
                if txt:
                    parts.append(txt)
                if sum(len(p) for p in parts) >= max_chars:
                    break
            return "\n".join(parts)[:max_chars]
        except Exception:
            return ""

    # DOCX (best-effort)
    if (
        ct
        in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/msword",
        }
        or name.endswith(".docx")
    ):
        try:
            import docx  # type: ignore
        except Exception:
            return ""

        try:
            d = docx.Document(str(file_path))
            text = "\n".join(p.text for p in d.paragraphs if p.text)
            return text[:max_chars]
        except Exception:
            return ""

    # Images (OCR best-effort)
    if ct.startswith("image/") or name.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")):
        # Prefer local tesseract if present; otherwise try the optional OCR sidecar.
        if shutil.which("tesseract"):
            try:
                from PIL import Image  # type: ignore
            except Exception:
                return ""

            try:
                import pytesseract  # type: ignore
            except Exception:
                return ""

            try:
                with Image.open(str(file_path)) as img:
                    txt = pytesseract.image_to_string(img) or ""
                    return txt[:max_chars]
            except Exception:
                return ""

        if ocr_service_url:
            try:
                import httpx

                with httpx.Client(timeout=ocr_timeout_seconds) as client:
                    with open(file_path, "rb") as f:
                        files = {
                            "file": (
                                filename,
                                f,
                                content_type or "application/octet-stream",
                            )
                        }
                        resp = client.post(f"{ocr_service_url.rstrip('/')}/ocr", files=files)
                        resp.raise_for_status()
                        data = resp.json()
                        txt = (data.get("text") or "")
                        return txt[:max_chars]
            except Exception:
                return ""

        return ""

    return ""
