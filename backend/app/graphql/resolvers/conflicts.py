"""Conflict resolver helpers."""

from __future__ import annotations

import strawberry


@strawberry.type
class GqlConflict:
    conflict_type: str
    severity: str
    resolution_hint: str


def build_conflicts_from_metadata(extra_metadata: dict) -> list[GqlConflict]:
    """Map memory metadata conflict hints into GraphQL conflict objects."""
    conflicts = extra_metadata.get("conflicts") or []
    if isinstance(conflicts, list) and conflicts:
        mapped: list[GqlConflict] = []
        for item in conflicts:
            if not isinstance(item, dict):
                continue
            mapped.append(
                GqlConflict(
                    conflict_type=str(item.get("conflict_type") or "unknown"),
                    severity=str(item.get("severity") or "medium"),
                    resolution_hint=str(item.get("resolution_hint") or ""),
                )
            )
        if mapped:
            return mapped

    count = int(extra_metadata.get("conflict_count") or 0)
    if count <= 0:
        return []

    return [
        GqlConflict(
            conflict_type="metadata.conflict_count",
            severity="medium" if count < 3 else "high",
            resolution_hint=f"{count} possible conflict signal(s) detected",
        )
    ]
