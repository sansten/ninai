from __future__ import annotations

import re
from dataclasses import dataclass


GATE_HEADER_RE = re.compile(r"^##\s+Gate\s+.+\((P0|P1)\)\s*$")
STATUS_RE = re.compile(r"^[-*]\s+Status:\s+\[(?P<pass>[ xX])\]\s+Pass\s+\[(?P<partial>[ xX])\]\s+Partial\s+\[(?P<fail>[ xX])\]\s+Fail\s*$")


@dataclass
class GateScore:
    priority: str
    total: int = 0
    passed: int = 0


@dataclass
class ReadinessScore:
    p0_total: int
    p0_passed: int
    p1_total: int
    p1_passed: int

    @property
    def p0_all_pass(self) -> bool:
        return self.p0_total > 0 and self.p0_passed == self.p0_total

    @property
    def p1_pass_rate(self) -> float:
        if self.p1_total == 0:
            return 0.0
        return self.p1_passed / self.p1_total

    @property
    def go(self) -> bool:
        return self.p0_all_pass and self.p1_pass_rate >= 0.80

    def to_dict(self) -> dict:
        return {
            "p0_total": self.p0_total,
            "p0_passed": self.p0_passed,
            "p0_all_pass": self.p0_all_pass,
            "p1_total": self.p1_total,
            "p1_passed": self.p1_passed,
            "p1_pass_rate": round(self.p1_pass_rate, 4),
            "p1_pass_threshold_met": self.p1_pass_rate >= 0.80,
            "go": self.go,
        }


def _is_checked(val: str) -> bool:
    return val.strip().lower() == "x"


def score_checklist(markdown: str) -> ReadinessScore:
    p0 = GateScore(priority="P0")
    p1 = GateScore(priority="P1")
    current_priority: str | None = None

    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        gate_match = GATE_HEADER_RE.match(line)
        if gate_match:
            current_priority = gate_match.group(1)
            continue

        status_match = STATUS_RE.match(line)
        if not status_match or current_priority is None:
            continue

        is_pass = _is_checked(status_match.group("pass"))
        if current_priority == "P0":
            p0.total += 1
            if is_pass:
                p0.passed += 1
        elif current_priority == "P1":
            p1.total += 1
            if is_pass:
                p1.passed += 1

    return ReadinessScore(
        p0_total=p0.total,
        p0_passed=p0.passed,
        p1_total=p1.total,
        p1_passed=p1.passed,
    )
