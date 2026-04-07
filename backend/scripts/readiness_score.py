from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.readiness_score import score_checklist


def main() -> int:
    parser = argparse.ArgumentParser(description="Score CognitiveOS readiness checklist")
    parser.add_argument(
        "--file",
        default=None,
        help="Path to the readiness checklist markdown file",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when go criteria are not met",
    )
    args = parser.parse_args()

    if args.file:
        path = Path(args.file)
    else:
        candidates = [
            BACKEND_ROOT / "docs" / "CognitiveOS_Readiness_Checklist.md",
            BACKEND_ROOT.parent.parent.parent / "REQUIREMENTS_MASTER.md",
        ]
        path = next((p for p in candidates if p.exists()), candidates[0])

    if not path.exists():
        print(json.dumps({"error": f"file not found: {path}"}))
        return 2

    score = score_checklist(path.read_text(encoding="utf-8"))
    print(json.dumps(score.to_dict(), indent=2))

    if args.strict and not score.go:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
