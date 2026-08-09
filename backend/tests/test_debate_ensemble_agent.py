from __future__ import annotations

import pytest

from app.agents.debate_ensemble_agent import DebateEnsembleAgent


class TestDebateEnsembleAgent:
    def test_generate_transcript_returns_moderator_and_dissent(self):
        agent = DebateEnsembleAgent()
        result = agent.generate_transcript(
            content="critical auth outage because token service is failing",
            decision="escalate",
            confidence=0.82,
            enrichment={"anomaly_detected": True, "anomaly_score": 0.9},
        )

        transcript = result["debate_transcript"]
        assert len(transcript) >= 4
        assert transcript[-1]["speaker"] == "moderator"
        assert "dissenting_opinions" in transcript[-1]
        assert result["moderator_decision"] in {"escalate", "investigate", "monitor"}

    @pytest.mark.asyncio
    async def test_run_outputs_transcript(self):
        agent = DebateEnsembleAgent()
        result = await agent.run(
            memory_id="m-debate",
            context={
                "debate": {
                    "content": "incident with elevated anomaly signals",
                    "decision": "investigate",
                    "confidence": 0.75,
                    "enrichment": {"anomaly_detected": True, "anomaly_score": 0.8},
                }
            },
        )

        assert result.status == "success"
        assert "debate_transcript" in result.outputs
        assert result.outputs["debate_transcript"][-1]["speaker"] == "moderator"

    @pytest.mark.asyncio
    async def test_run_falls_back_to_bus_enrichment_when_no_debate_key(self):
        """Regression: the only real caller is OrchestrationBusAgent, which
        never populates context["debate"] — it supplies the standard
        context["memory"] location every other agent reads from. Previously
        the absence of context["debate"] silently fabricated a debate from
        content="" and enrichment={}, ignoring the real anomaly signal."""
        agent = DebateEnsembleAgent()
        result = await agent.run(
            memory_id="m-debate-2",
            context={
                "memory": {
                    "content": "critical auth outage, anomaly detected",
                    "enrichment": {"anomaly_detected": True, "anomaly_score": 0.95},
                },
                "runtime": {},
            },
        )

        assert result.status == "success"
        # A real anomaly signal should push the safety debater to escalate,
        # which the fabricated content="" / enrichment={} path never could.
        transcript = result.outputs["debate_transcript"]
        safety_ballot = next(t for t in transcript if t.get("role") == "safety")
        assert safety_ballot["position"] == "escalate"

    @pytest.mark.asyncio
    async def test_run_prefers_explicit_debate_key_over_memory_fallback(self):
        """Backward compatibility: an explicit context["debate"] (if a
        future caller supplies one) still takes precedence over the
        context["memory"] fallback."""
        agent = DebateEnsembleAgent()
        result = await agent.run(
            memory_id="m-debate-3",
            context={
                "debate": {
                    "content": "explicit debate content",
                    "enrichment": {"anomaly_detected": False, "anomaly_score": 0.0},
                },
                "memory": {
                    "content": "should be ignored",
                    "enrichment": {"anomaly_detected": True, "anomaly_score": 0.99},
                },
            },
        )
        transcript = result.outputs["debate_transcript"]
        safety_ballot = next(t for t in transcript if t.get("role") == "safety")
        assert safety_ballot["position"] == "investigate"
