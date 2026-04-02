"""Adversarial robustness checks for inbound content and confidence batches."""

from __future__ import annotations

import re
import statistics
from typing import Any


class AdversarialRobustnessMonitor:
    _INJECTION_PATTERNS = [
        r"ignore previous instructions",
        r"disregard (all|the) (above|previous)",
        r"you are now",
        r"system prompt",
        r"forget everything",
        r"new instruction",
        r"act as",
    ]

    def _detect_prompt_injection(self, content: str) -> dict[str, Any] | None:
        text = content or ""
        for pattern in self._INJECTION_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return {
                    "type": "prompt_injection",
                    "severity": "high",
                    "matched": pattern,
                }
        return None

    @staticmethod
    def _detect_score_manipulation(metadata: dict[str, Any]) -> dict[str, Any] | None:
        meta = metadata or {}
        for key in ("credibility_score", "confidence"):
            if key not in meta:
                continue
            value = meta.get(key)
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue

            if value_f < 0.0 or value_f > 1.0:
                return {
                    "type": "score_manipulation",
                    "severity": "medium",
                    "value": value_f,
                }

            if key == "credibility_score" and value_f > 0.99:
                return {
                    "type": "score_manipulation",
                    "severity": "medium",
                    "value": value_f,
                }
        return None

    @staticmethod
    def _has_control_char(content: str) -> bool:
        for char in content:
            if ord(char) < 32 and char not in ("\n", "\r", "\t"):
                return True
        return False

    def _detect_encoding_attack(self, content: str) -> dict[str, Any] | None:
        text = content or ""
        if "\x00" in text:
            return {"type": "encoding_attack", "severity": "high"}

        if self._has_control_char(text):
            return {"type": "encoding_attack", "severity": "high"}

        if text.count("\u202e") >= 2:
            return {"type": "encoding_attack", "severity": "high"}

        return None

    def check_content(self, *, content: str, metadata: dict) -> list[dict]:
        findings: list[dict] = []

        prompt_finding = self._detect_prompt_injection(content)
        if prompt_finding:
            findings.append(prompt_finding)

        score_finding = self._detect_score_manipulation(metadata)
        if score_finding:
            findings.append(score_finding)

        encoding_finding = self._detect_encoding_attack(content)
        if encoding_finding:
            findings.append(encoding_finding)

        return findings

    @staticmethod
    def check_confidence_batch(*, confidence_values: list[float]) -> list[dict]:
        values = [float(v) for v in (confidence_values or [])]
        if len(values) <= 5:
            return []

        stddev = statistics.pstdev(values)
        if stddev < 0.01:
            return [{"type": "uniform_confidence_anomaly", "severity": "medium"}]
        return []

    @staticmethod
    def is_safe(*, findings: list[dict]) -> bool:
        return not any((item or {}).get("severity") == "high" for item in (findings or []))

    @staticmethod
    def risk_summary(*, findings: list[dict]) -> str:
        if not findings:
            return "clean"

        finding_types: list[str] = []
        for finding in findings:
            finding_type = str((finding or {}).get("type", "")).strip()
            if finding_type:
                finding_types.append(finding_type)

        return ", ".join(finding_types) if finding_types else "clean"
