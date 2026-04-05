"""Debate Ensemble Agent (Feature 24.5).

Runs a three-position debate for high-stakes decisions and synthesizes
the final recommendation with dissenting opinions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.base import BaseAgent
from app.agents.types import AgentContext, AgentResult
from app.services.multi_agent_voting_engine import MultiAgentVotingEngine


class DebateEnsembleAgent(BaseAgent):
    name = "DebateEnsembleAgent"
    version = "v1"

    def _debater_ballots(
        self,
        *,
        decision: str,
        confidence: float,
        enrichment: dict[str, Any],
    ) -> list[dict[str, Any]]:
        anomaly = bool(enrichment.get("anomaly_detected"))
        anomaly_score = float(enrichment.get("anomaly_score") or 0.0)

        # Three perspectives use slightly different priors.
        safety_verdict = anomaly or anomaly_score >= 0.7
        safety_position = "escalate" if safety_verdict else "investigate"

        operations_verdict = decision in {"escalate", "investigate"}
        operations_position = "investigate" if operations_verdict else "monitor"

        efficiency_verdict = decision == "monitor"
        efficiency_position = "monitor" if efficiency_verdict else "investigate"

        return [
            {
                "agent_name": "anomaly_detection_agent",
                "role": "safety",
                "position": safety_position,
                "argument": "Prioritize risk containment under anomaly evidence.",
                "verdict": safety_verdict,
                "confidence": min(1.0, max(0.45, confidence + 0.05)),
            },
            {
                "agent_name": "causal_reasoning_agent",
                "role": "operations",
                "position": operations_position,
                "argument": "Favor diagnostic progression before full escalation when possible.",
                "verdict": operations_verdict,
                "confidence": min(1.0, max(0.4, confidence)),
            },
            {
                "agent_name": "uncertainty_reporting_agent",
                "role": "efficiency",
                "position": efficiency_position,
                "argument": "Minimize disruption unless evidence justifies stronger intervention.",
                "verdict": efficiency_verdict,
                "confidence": min(1.0, max(0.35, 1.0 - confidence + 0.2)),
            },
        ]

    def generate_transcript(
        self,
        *,
        content: str,
        decision: str,
        confidence: float,
        enrichment: dict[str, Any],
    ) -> dict[str, Any]:
        ballots = self._debater_ballots(
            decision=decision,
            confidence=confidence,
            enrichment=enrichment,
        )

        vote_summary = MultiAgentVotingEngine().vote(ballots=ballots)
        moderator_decision = "investigate" if vote_summary["consensus"] else "monitor"
        if decision == "escalate" and vote_summary["consensus"]:
            moderator_decision = "escalate"

        transcript: list[dict[str, Any]] = []
        for idx, ballot in enumerate(ballots, start=1):
            transcript.append(
                {
                    "round": 1,
                    "speaker": f"debater_{idx}",
                    "role": ballot["role"],
                    "position": ballot["position"],
                    "argument": ballot["argument"],
                    "confidence": round(float(ballot["confidence"]), 4),
                }
            )

        transcript.append(
            {
                "round": 2,
                "speaker": "moderator",
                "position": moderator_decision,
                "argument": "Synthesized recommendation from weighted debate voting.",
                "dissenting_opinions": vote_summary.get("dissenting_agents") or [],
                "agreement_rate": vote_summary.get("agreement_rate", 0.0),
            }
        )

        return {
            "debate_transcript": transcript,
            "moderator_decision": moderator_decision,
            "dissenting_opinions": vote_summary.get("dissenting_agents") or [],
        }

    async def run(self, memory_id: str, context: AgentContext) -> AgentResult:
        started = datetime.now(timezone.utc)
        runtime = context.get("runtime") if isinstance(context, dict) else {}
        inputs = context.get("debate") if isinstance(context, dict) else {}
        inputs = inputs if isinstance(inputs, dict) else {}

        result = self.generate_transcript(
            content=str(inputs.get("content") or ""),
            decision=str(inputs.get("decision") or "investigate"),
            confidence=float(inputs.get("confidence") or 0.7),
            enrichment=dict(inputs.get("enrichment") or {}),
        )

        finished = datetime.now(timezone.utc)
        return AgentResult(
            agent_name=self.name,
            agent_version=self.version,
            memory_id=memory_id,
            status="success",
            confidence=0.82,
            outputs=result,
            warnings=[],
            errors=[],
            started_at=started,
            finished_at=finished,
            trace_id=str((runtime or {}).get("job_id")) if (runtime or {}).get("job_id") else None,
        )
