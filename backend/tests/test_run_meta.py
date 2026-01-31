from __future__ import annotations

from app.utils.run_meta import sha256_file


def test_sha256_file(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("abc", encoding="utf-8")
    # Known SHA256 for 'abc'
    assert sha256_file(str(p)) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
