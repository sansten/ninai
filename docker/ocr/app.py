from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ocr")
def ocr():
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400

    upload = request.files["file"]
    filename = upload.filename or "upload"
    suffix = Path(filename).suffix or ".bin"

    lang = os.getenv("TESSERACT_LANG", "eng")
    psm = os.getenv("TESSERACT_PSM", "3")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            upload.save(tmp_path)

        cmd = [
            "tesseract",
            tmp_path,
            "stdout",
            "-l",
            lang,
            "--psm",
            str(psm),
        ]
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        if proc.returncode != 0:
            return jsonify({"text": "", "error": (proc.stderr or "")[:1000]}), 500

        return jsonify({"text": proc.stdout or ""})
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
