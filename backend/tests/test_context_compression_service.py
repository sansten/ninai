from __future__ import annotations

from app.services.context_compression_service import ContextCompressionService


class TestContextCompressionService:
    def test_no_memories_returns_neutral_metrics(self):
        svc = ContextCompressionService()
        result = svc.compress(memories=[], token_budget=120)
        assert result.memories == []
        assert result.compression_ratio == 1.0

    def test_compress_reduces_tokens_when_over_budget(self):
        svc = ContextCompressionService()
        memories = [
            {
                "id": "m1",
                "content": " ".join(["auth"] * 120),
                "credibility_score": 0.9,
            },
            {
                "id": "m2",
                "content": " ".join(["incident"] * 120),
                "credibility_score": 0.4,
            },
        ]
        result = svc.compress(memories=memories, token_budget=80)
        assert result.compression_ratio < 1.0
        assert any(m.get("_compressed") for m in result.memories)

    def test_preserves_entity_sentence_when_possible(self):
        svc = ContextCompressionService()
        memories = [
            {
                "id": "m1",
                "content": "noise noise noise. customer_id C123 failed due to auth timeout. more filler words "
                + " ".join(["x"] * 80),
                "entities": {"customer_id": "C123"},
                "credibility_score": 0.8,
            }
        ]
        result = svc.compress(memories=memories, token_budget=40)
        assert "C123" in result.memories[0]["content"]
