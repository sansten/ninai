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
