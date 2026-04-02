from __future__ import annotations

from app.services.cognitive_offload_scheduler import CognitiveOffloadScheduler


class TestDecide:
    def test_decision_type_high_importance_kept(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="decision",
            importance_score=0.8,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "keep"

    def test_log_mid_importance_unreferenced_compress(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="log",
            importance_score=0.3,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "compress"

    def test_documentation_low_importance_offload(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="documentation",
            importance_score=0.1,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "offload"

    def test_near_duplicate_discard(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="goal",
            importance_score=0.95,
            reference_count=10,
            is_near_duplicate=True,
        )
        assert result == "discard"

    def test_debug_trace_low_importance_discard(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="debug_trace",
            importance_score=0.05,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "discard"

    def test_test_low_importance_discard(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="test",
            importance_score=0.01,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "discard"

    def test_reference_count_two_keeps_even_low_importance(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="documentation",
            importance_score=0.02,
            reference_count=2,
            is_near_duplicate=False,
        )
        assert result == "keep"

    def test_importance_over_point_six_fallback_keep(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="unknown",
            importance_score=0.7,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "keep"

    def test_unknown_type_mid_importance_fallback_compress(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="unknown",
            importance_score=0.4,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "compress"

    def test_unknown_type_low_importance_fallback_offload(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="unknown",
            importance_score=0.05,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "offload"

    def test_uppercase_content_type_supported(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="DECISION",
            importance_score=0.8,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "keep"

    def test_reference_count_negative_treated_zero(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="log",
            importance_score=0.3,
            reference_count=-5,
            is_near_duplicate=False,
        )
        assert result == "compress"

    def test_importance_clamped_above_one(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="decision",
            importance_score=2.0,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "keep"

    def test_importance_clamped_below_zero(self):
        svc = CognitiveOffloadScheduler()
        result = svc.decide(
            content_type="documentation",
            importance_score=-1.0,
            reference_count=0,
            is_near_duplicate=False,
        )
        assert result == "offload"


class TestCompressContent:
    def test_compress_truncates_at_sentence_boundary(self):
        svc = CognitiveOffloadScheduler()
        content = "First sentence. Second sentence with extra words. Third sentence."
        result = svc.compress_content(content, max_chars=35)
        assert result.startswith("First sentence.")

    def test_compress_appends_marker(self):
        svc = CognitiveOffloadScheduler()
        content = "Sentence one. Sentence two. Sentence three."
        result = svc.compress_content(content, max_chars=20)
        assert result.endswith("[compressed]")

    def test_short_content_not_modified(self):
        svc = CognitiveOffloadScheduler()
        content = "Short line."
        assert svc.compress_content(content, max_chars=200) == "Short line."

    def test_compress_without_punctuation_uses_hard_cut(self):
        svc = CognitiveOffloadScheduler()
        content = "abcdefghijklmnopqrstuvwxyz"
        result = svc.compress_content(content, max_chars=10)
        assert result == "abcdefghij [compressed]"

    def test_compress_handles_none_content(self):
        svc = CognitiveOffloadScheduler()
        assert svc.compress_content(None, max_chars=10) == ""

    def test_compress_min_limit_one(self):
        svc = CognitiveOffloadScheduler()
        result = svc.compress_content("abcdef", max_chars=0)
        assert result == "a [compressed]"


class TestOffloadPointer:
    def test_offload_pointer_shape(self):
        svc = CognitiveOffloadScheduler()
        result = svc.offload_pointer(content="docs page", source_url="https://example.com/docs")
        assert result == {
            "type": "pointer",
            "summary": "docs page",
            "source_url": "https://example.com/docs",
            "retrievable": True,
        }

    def test_offload_pointer_summary_capped_to_hundred(self):
        svc = CognitiveOffloadScheduler()
        content = "x" * 150
        result = svc.offload_pointer(content=content, source_url=None)
        assert len(result["summary"]) == 100

    def test_offload_pointer_trims_content(self):
        svc = CognitiveOffloadScheduler()
        result = svc.offload_pointer(content="  hello world  ", source_url=None)
        assert result["summary"] == "hello world"


class TestBatchDecide:
    def test_batch_decide_populates_all_buckets(self):
        svc = CognitiveOffloadScheduler()
        memories = [
            {"id": "k", "content_type": "decision", "importance_score": 0.9, "reference_count": 0, "is_near_duplicate": False},
            {"id": "c", "content_type": "log", "importance_score": 0.3, "reference_count": 0, "is_near_duplicate": False},
            {"id": "o", "content_type": "documentation", "importance_score": 0.1, "reference_count": 0, "is_near_duplicate": False},
            {"id": "d", "content_type": "debug_trace", "importance_score": 0.01, "reference_count": 0, "is_near_duplicate": False},
        ]
        buckets = svc.batch_decide(memories=memories)
        assert len(buckets["keep"]) == 1
        assert len(buckets["compress"]) == 1
        assert len(buckets["offload"]) == 1
        assert len(buckets["discard"]) == 1

    def test_batch_decide_empty_input(self):
        svc = CognitiveOffloadScheduler()
        buckets = svc.batch_decide(memories=[])
        assert buckets == {"keep": [], "compress": [], "offload": [], "discard": []}

    def test_batch_decide_missing_fields_defaults(self):
        svc = CognitiveOffloadScheduler()
        buckets = svc.batch_decide(memories=[{"id": "x"}])
        assert len(buckets["offload"]) == 1

    def test_batch_decide_near_duplicate_wins_over_ref_count(self):
        svc = CognitiveOffloadScheduler()
        buckets = svc.batch_decide(
            memories=[
                {
                    "id": "x",
                    "content_type": "goal",
                    "importance_score": 0.9,
                    "reference_count": 10,
                    "is_near_duplicate": True,
                }
            ]
        )
        assert len(buckets["discard"]) == 1

    def test_batch_decide_preserves_input_identity_fields(self):
        svc = CognitiveOffloadScheduler()
        memory = {
            "id": "abc",
            "content_type": "decision",
            "importance_score": 0.8,
            "reference_count": 0,
            "is_near_duplicate": False,
            "custom": "value",
        }
        buckets = svc.batch_decide(memories=[memory])
        assert buckets["keep"][0]["id"] == "abc"
        assert buckets["keep"][0]["custom"] == "value"
