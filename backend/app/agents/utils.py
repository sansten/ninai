from __future__ import annotations

import hashlib
from typing import Iterable


CLASSIFICATION_ORDER = {
    "public": 0,
    "internal": 1,
    "confidential": 2,
    "restricted": 3,
}


def compute_inputs_hash(parts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def max_classification(a: str | None, b: str | None) -> str:
    aa = (a or "internal").lower()
    bb = (b or "internal").lower()
    if CLASSIFICATION_ORDER.get(aa, 1) >= CLASSIFICATION_ORDER.get(bb, 1):
        return aa
    return bb
