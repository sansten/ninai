"""E1: Decision quality benchmark for CognitiveGatewayService.decide()."""
from __future__ import annotations

from typing import Any

from app.services.cognitive_gateway_service import (
    CognitiveGatewayCapabilities,
    CognitiveGatewayService,
)

# Curated task set with inputs that produce deterministic heuristic decisions.
#
# Decision rules in _heuristic_decide():
#   - anomaly_detected=True AND anomaly_score >= 0.9  → "escalate"
#   - anomaly_detected=True AND anomaly_score >= 0.7  → "investigate"
#   - content has "critical"|"urgent"|"outage"         → "escalate"
#   - content has "warning"|"caution"                  → "monitor"
#   - default                                          → "acknowledge"
#
DECISION_TASKS: list[tuple[str, dict, str]] = [
    # high-score anomalies → escalate
    ("auth failure spike in prod", {"anomaly_detected": True, "anomaly_score": 0.95}, "escalate"),
    ("payment gateway returning 500s", {"anomaly_detected": True, "anomaly_score": 0.92}, "escalate"),
    # mid-score anomalies → investigate
    ("database connection timeout after deploy", {"anomaly_detected": True, "anomaly_score": 0.80}, "investigate"),
    ("memory usage climbing on api nodes", {"anomaly_detected": True, "anomaly_score": 0.75}, "investigate"),
    # content keywords → escalate
    ("full outage on billing service", {"anomaly_detected": False, "anomaly_score": 0.30}, "escalate"),
    ("critical error in payment module", {"anomaly_detected": False, "anomaly_score": 0.25}, "escalate"),
    ("urgent: certificate expiry in 1 hour", {"anomaly_detected": False, "anomaly_score": 0.20}, "escalate"),
    # content keywords → monitor
    ("warning: disk usage at 85% on prod", {"anomaly_detected": False, "anomaly_score": 0.15}, "monitor"),
    ("caution: cache miss rate increasing slowly", {"anomaly_detected": False, "anomaly_score": 0.10}, "monitor"),
    # default → acknowledge
    ("routine daily report completed", {"anomaly_detected": False, "anomaly_score": 0.05}, "acknowledge"),
    ("weekly newsletter sent successfully", {"anomaly_detected": False, "anomaly_score": 0.02}, "acknowledge"),
    ("user updated their profile photo", {"anomaly_detected": False, "anomaly_score": 0.01}, "acknowledge"),
]

QUALITY_FLOOR = 0.75  # 75% correct decisions required


async def run(*, mode: str, strategy: str, **kwargs: Any) -> dict[str, Any]:
    caps = CognitiveGatewayCapabilities.full()
    svc = CognitiveGatewayService(capabilities=caps)

    correct = 0
    results = []
    for content, enrichment, expected in DECISION_TASKS:
        result = await svc.decide(content=content, enrichment=enrichment)
        got = result.decision
        match = got == expected
        if match:
            correct += 1
        results.append(
            {
                "content": content[:40],
                "expected": expected,
                "got": got,
                "match": match,
            }
        )

    accuracy = correct / len(DECISION_TASKS)
    return {
        "benchmark": "decide_quality",
        "mode": mode,
        "strategy": strategy,
        "accuracy": round(accuracy, 4),
        "quality_floor": QUALITY_FLOOR,
        "passed": accuracy >= QUALITY_FLOOR,
        "task_count": len(DECISION_TASKS),
        "correct": correct,
        "results": results,
        "status": "ok" if accuracy >= QUALITY_FLOOR else "below_floor",
    }
