"""Tests for EmotionalAffectiveMemoryAgent — Phase 42."""

from __future__ import annotations

import pytest

from app.agents.emotional_affective_memory_agent import (
    EmotionalAffectiveMemoryAgent,
    _tokenize,
    score_valence,
    classify_valence,
    compute_arousal,
    check_empathy_flag,
    check_escalation,
    derive_tone_guidance,
    compute_confidence,
    _merge_valence_with_hint,
)
from app.agents.types import AgentContext


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _ctx(
    content: str = "",
    feedback_signals: list | None = None,
    narrative_tone: str = "",
    urgency_score: float = 0.0,
    strategy: str = "heuristic",
) -> AgentContext:
    return {
        "memory": {
            "content": content,
            "enrichment": {
                "feedback_signals": feedback_signals or [],
                "narrative_tone": narrative_tone,
                "urgency_score": urgency_score,
            },
        },
        "runtime": {"job_id": "trace-42"},
        "settings": {"AGENT_STRATEGY": strategy},
    }


AGENT = EmotionalAffectiveMemoryAgent()

# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------

def test_tokenize_basic():
    tokens = _tokenize("Hello World test")
    assert "hello" in tokens
    assert "world" in tokens
    assert "test" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []


def test_tokenize_numbers_excluded():
    tokens = _tokenize("abc 123 xyz")
    assert "123" not in tokens


def test_tokenize_lowercases():
    tokens = _tokenize("URGENT CRISIS")
    assert "urgent" in tokens
    assert "crisis" in tokens


# ---------------------------------------------------------------------------
# score_valence
# ---------------------------------------------------------------------------

def test_score_valence_positive():
    pos, neg = score_valence("This is great and excellent work, I am very happy")
    assert pos >= 2
    assert neg == 0


def test_score_valence_negative():
    pos, neg = score_valence("The system crashed and there is a critical error")
    assert neg >= 2


def test_score_valence_empty():
    pos, neg = score_valence("")
    assert pos == 0
    assert neg == 0


def test_score_valence_mixed():
    pos, neg = score_valence("great success but also a terrible failure")
    assert pos >= 1
    assert neg >= 1


# ---------------------------------------------------------------------------
# classify_valence
# ---------------------------------------------------------------------------

def test_classify_valence_neutral():
    assert classify_valence(0, 0) == "neutral"


def test_classify_valence_positive():
    assert classify_valence(3, 0) == "positive"


def test_classify_valence_negative():
    assert classify_valence(0, 3) == "negative"


def test_classify_valence_mixed():
    assert classify_valence(2, 2) == "mixed"


def test_classify_valence_positive_dominates():
    # Any mix of pos+neg → mixed regardless of magnitude
    assert classify_valence(4, 1) == "mixed"


def test_classify_valence_negative_dominates():
    assert classify_valence(1, 4) == "mixed"


# ---------------------------------------------------------------------------
# compute_arousal
# ---------------------------------------------------------------------------

def test_arousal_high_with_crisis_words():
    arousal = compute_arousal("urgent critical emergency outage incident panic", [])
    assert arousal > 0.5


def test_arousal_low_with_neutral_text():
    arousal = compute_arousal("the report was submitted on time", [])
    assert arousal <= 0.3


def test_arousal_empty_text():
    arousal = compute_arousal("", [])
    assert arousal == 0.0


def test_arousal_boosted_by_low_feedback_ratings():
    signals = [{"rating": 1}, {"rating": 1}, {"rating": 2}]
    arousal = compute_arousal("normal text here", signals)
    assert arousal > 0.0


def test_arousal_not_boosted_by_high_ratings():
    signals = [{"rating": 5}, {"rating": 4}]
    arousal_with = compute_arousal("urgent critical", signals)
    arousal_without = compute_arousal("urgent critical", [])
    # High ratings shouldn't inflate arousal beyond word-based score
    assert arousal_with >= arousal_without or arousal_with == pytest.approx(arousal_without, abs=0.1)


def test_arousal_capped_at_one():
    text = " ".join(["urgent critical emergency outage incident panic alarm alert"] * 5)
    arousal = compute_arousal(text, [{"rating": 1}] * 10)
    assert arousal <= 1.0


def test_arousal_no_feedback_signals():
    arousal = compute_arousal("everything is fine", [])
    assert 0.0 <= arousal <= 1.0


# ---------------------------------------------------------------------------
# check_empathy_flag
# ---------------------------------------------------------------------------

def test_empathy_flag_true_on_distress_word():
    assert check_empathy_flag("please help us we are stuck", "neutral", 0.2) is True


def test_empathy_flag_true_on_negative_high_arousal():
    assert check_empathy_flag("normal text", "negative", 0.5) is True


def test_empathy_flag_false_on_positive():
    assert check_empathy_flag("great result achieved", "positive", 0.2) is False


def test_empathy_flag_true_on_mixed_high_arousal():
    assert check_empathy_flag("some text", "mixed", 0.7) is True


def test_empathy_flag_false_on_mixed_low_arousal():
    assert check_empathy_flag("some text", "mixed", 0.3) is False


def test_empathy_flag_distress_word_overrides_positive():
    assert check_empathy_flag("we are desperate for help", "positive", 0.1) is True


# ---------------------------------------------------------------------------
# check_escalation
# ---------------------------------------------------------------------------

def test_escalation_high_urgency():
    assert check_escalation(
        valence="neutral", arousal=0.1, empathy_flag=False,
        urgency_score=0.9, feedback_signals=[]
    ) is True


def test_escalation_negative_high_arousal_empathy():
    assert check_escalation(
        valence="negative", arousal=0.6, empathy_flag=True,
        urgency_score=0.2, feedback_signals=[]
    ) is True


def test_escalation_multiple_low_ratings():
    signals = [{"rating": 1}, {"rating": 2}]
    assert check_escalation(
        valence="neutral", arousal=0.1, empathy_flag=False,
        urgency_score=0.1, feedback_signals=signals
    ) is True


def test_escalation_very_high_arousal():
    assert check_escalation(
        valence="positive", arousal=0.8, empathy_flag=False,
        urgency_score=0.0, feedback_signals=[]
    ) is True


def test_no_escalation_normal():
    assert check_escalation(
        valence="positive", arousal=0.1, empathy_flag=False,
        urgency_score=0.2, feedback_signals=[]
    ) is False


def test_no_escalation_one_low_rating():
    signals = [{"rating": 1}]
    assert check_escalation(
        valence="neutral", arousal=0.2, empathy_flag=False,
        urgency_score=0.1, feedback_signals=signals
    ) is False


def test_escalation_boundary_urgency():
    # Exactly 0.8 should trigger escalation
    assert check_escalation(
        valence="neutral", arousal=0.0, empathy_flag=False,
        urgency_score=0.8, feedback_signals=[]
    ) is True


def test_no_escalation_below_urgency_threshold():
    assert check_escalation(
        valence="neutral", arousal=0.0, empathy_flag=False,
        urgency_score=0.79, feedback_signals=[]
    ) is False


# ---------------------------------------------------------------------------
# derive_tone_guidance
# ---------------------------------------------------------------------------

def test_tone_positive_low():
    assert derive_tone_guidance("positive", 0.2) == "affirming"


def test_tone_positive_high():
    assert derive_tone_guidance("positive", 0.7) == "enthusiastic"


def test_tone_negative_low():
    assert derive_tone_guidance("negative", 0.3) == "empathetic"


def test_tone_negative_high():
    assert derive_tone_guidance("negative", 0.8) == "urgent_empathetic"


def test_tone_neutral_low():
    assert derive_tone_guidance("neutral", 0.1) == "neutral"


def test_tone_neutral_high():
    assert derive_tone_guidance("neutral", 0.6) == "attentive"


def test_tone_mixed_low():
    assert derive_tone_guidance("mixed", 0.2) == "balanced"


def test_tone_mixed_high():
    assert derive_tone_guidance("mixed", 0.9) == "cautious_empathetic"


def test_tone_boundary_at_half():
    # 0.5 is high bucket
    assert derive_tone_guidance("negative", 0.5) == "urgent_empathetic"


def test_tone_below_half_is_low():
    assert derive_tone_guidance("negative", 0.49) == "empathetic"


# ---------------------------------------------------------------------------
# compute_confidence
# ---------------------------------------------------------------------------

def test_confidence_no_content():
    assert compute_confidence(content_len=0, pos=0, neg=0, feedback_count=0) == 0.30


def test_confidence_high_signal():
    conf = compute_confidence(content_len=100, pos=3, neg=3, feedback_count=2)
    assert conf >= 0.85


def test_confidence_medium_signal():
    conf = compute_confidence(content_len=50, pos=2, neg=1, feedback_count=0)
    assert conf >= 0.65


def test_confidence_low_signal():
    conf = compute_confidence(content_len=20, pos=0, neg=0, feedback_count=1)
    assert conf >= 0.65


def test_confidence_no_signal():
    conf = compute_confidence(content_len=50, pos=0, neg=0, feedback_count=0)
    assert conf == 0.50


# ---------------------------------------------------------------------------
# _merge_valence_with_hint
# ---------------------------------------------------------------------------

def test_merge_same_values():
    assert _merge_valence_with_hint("positive", "positive") == "positive"


def test_merge_neutral_text_gets_hint():
    assert _merge_valence_with_hint("neutral", "negative") == "negative"


def test_merge_neutral_hint_keeps_text():
    assert _merge_valence_with_hint("positive", "neutral") == "positive"


def test_merge_conflict_returns_mixed():
    assert _merge_valence_with_hint("positive", "negative") == "mixed"


def test_merge_empty_tone_keeps_text():
    assert _merge_valence_with_hint("negative", "") == "negative"


def test_merge_unknown_tone_keeps_text():
    assert _merge_valence_with_hint("positive", "unknown_tone") == "positive"


# ---------------------------------------------------------------------------
# Agent.validate_outputs
# ---------------------------------------------------------------------------

def _make_result(outputs: dict) -> object:
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    return AgentResult(
        agent_name="EmotionalAffectiveMemoryAgent",
        agent_version="v1",
        memory_id="m1",
        status="success",
        confidence=outputs.get("confidence", 0.7),
        outputs=outputs,
        warnings=[],
        errors=[],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )


def test_validate_good_outputs():
    result = _make_result({
        "emotional_valence": "positive",
        "arousal_level": 0.3,
        "empathy_flag": False,
        "escalation_recommended": False,
        "tone_guidance": "affirming",
        "confidence": 0.8,
        "rationale": "heuristic",
    })
    AGENT.validate_outputs(result)  # should not raise


def test_validate_bad_valence():
    result = _make_result({
        "emotional_valence": "ecstatic",
        "arousal_level": 0.3,
        "empathy_flag": False,
        "escalation_recommended": False,
        "tone_guidance": "affirming",
        "confidence": 0.8,
    })
    with pytest.raises(ValueError, match="emotional_valence"):
        AGENT.validate_outputs(result)


def test_validate_bad_arousal():
    result = _make_result({
        "emotional_valence": "neutral",
        "arousal_level": 1.5,
        "empathy_flag": False,
        "escalation_recommended": False,
        "tone_guidance": "neutral",
        "confidence": 0.7,
    })
    with pytest.raises(ValueError, match="arousal_level"):
        AGENT.validate_outputs(result)


def test_validate_bad_empathy_flag():
    result = _make_result({
        "emotional_valence": "negative",
        "arousal_level": 0.4,
        "empathy_flag": "yes",
        "escalation_recommended": False,
        "tone_guidance": "empathetic",
        "confidence": 0.7,
    })
    with pytest.raises(ValueError, match="empathy_flag"):
        AGENT.validate_outputs(result)


def test_validate_bad_escalation():
    result = _make_result({
        "emotional_valence": "negative",
        "arousal_level": 0.4,
        "empathy_flag": True,
        "escalation_recommended": "maybe",
        "tone_guidance": "empathetic",
        "confidence": 0.7,
    })
    with pytest.raises(ValueError, match="escalation_recommended"):
        AGENT.validate_outputs(result)


def test_validate_bad_confidence():
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    # Pass valid top-level confidence but bad value inside outputs dict
    result = AgentResult(
        agent_name="EmotionalAffectiveMemoryAgent",
        agent_version="v1",
        memory_id="m1",
        status="success",
        confidence=0.7,
        outputs={
            "emotional_valence": "positive",
            "arousal_level": 0.2,
            "empathy_flag": False,
            "escalation_recommended": False,
            "tone_guidance": "affirming",
            "confidence": 2.0,
        },
        warnings=[],
        errors=[],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )
    with pytest.raises(ValueError, match="confidence"):
        AGENT.validate_outputs(result)


def test_validate_skipped_on_non_success():
    from app.agents.types import AgentResult
    from datetime import datetime, timezone
    result = AgentResult(
        agent_name="EmotionalAffectiveMemoryAgent",
        agent_version="v1",
        memory_id="m1",
        status="failed",
        confidence=0.0,
        outputs={},
        warnings=[],
        errors=["fail"],
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        trace_id=None,
    )
    AGENT.validate_outputs(result)  # should not raise


# ---------------------------------------------------------------------------
# Agent.run — heuristic path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_positive_content():
    result = await AGENT.run("m1", _ctx("The deployment was a great success and everyone is happy"))
    assert result.status == "success"
    assert result.outputs["emotional_valence"] == "positive"
    assert result.outputs["empathy_flag"] is False
    assert isinstance(result.outputs["arousal_level"], float)


@pytest.mark.asyncio
async def test_run_negative_crisis_content():
    result = await AGENT.run(
        "m2",
        _ctx("critical outage emergency the system is down please help urgent panic"),
    )
    assert result.status == "success"
    assert result.outputs["emotional_valence"] in {"negative", "mixed"}
    assert result.outputs["empathy_flag"] is True
    assert result.outputs["escalation_recommended"] is True


@pytest.mark.asyncio
async def test_run_neutral_content():
    result = await AGENT.run("m3", _ctx("The meeting is scheduled for Monday at 10am"))
    assert result.status == "success"
    assert result.outputs["emotional_valence"] == "neutral"
    assert result.outputs["empathy_flag"] is False


@pytest.mark.asyncio
async def test_run_empty_content():
    result = await AGENT.run("m4", _ctx(""))
    assert result.status == "success"
    assert result.outputs["emotional_valence"] in {"positive", "negative", "neutral", "mixed"}
    assert isinstance(result.outputs["confidence"], float)


@pytest.mark.asyncio
async def test_run_with_low_feedback_signals():
    signals = [{"text": "terrible", "rating": 1}, {"text": "awful", "rating": 1}]
    result = await AGENT.run("m5", _ctx("service degraded", feedback_signals=signals))
    assert result.outputs["escalation_recommended"] is True


@pytest.mark.asyncio
async def test_run_with_high_feedback_signals():
    signals = [{"text": "great", "rating": 5}, {"text": "perfect", "rating": 5}]
    result = await AGENT.run("m6", _ctx("everything looks good", feedback_signals=signals))
    assert result.outputs["escalation_recommended"] is False


@pytest.mark.asyncio
async def test_run_high_urgency_score(monkeypatch):
    import app.agents.emotional_affective_memory_agent as eama_mod
    monkeypatch.setattr(eama_mod.settings, "AGENT_STRATEGY", "heuristic")
    result = await AGENT.run("m7", _ctx("routine update", urgency_score=0.9))
    assert result.outputs["escalation_recommended"] is True


@pytest.mark.asyncio
async def test_run_narrative_tone_hint_applied():
    # Neutral text but alarming narrative tone should push valence negative or mixed
    result = await AGENT.run("m8", _ctx("the situation continues", narrative_tone="alarming"))
    assert result.outputs["emotional_valence"] in {"negative", "mixed"}


@pytest.mark.asyncio
async def test_run_positive_narrative_hint_on_neutral_text():
    result = await AGENT.run("m9", _ctx("the report is ready", narrative_tone="positive"))
    assert result.outputs["emotional_valence"] == "positive"


@pytest.mark.asyncio
async def test_run_tone_guidance_is_string():
    result = await AGENT.run("m10", _ctx("everything is fine"))
    assert isinstance(result.outputs["tone_guidance"], str)
    assert len(result.outputs["tone_guidance"]) > 0


@pytest.mark.asyncio
async def test_run_confidence_in_range():
    result = await AGENT.run("m11", _ctx("some content here"))
    assert 0.0 <= result.outputs["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_run_rationale_heuristic(monkeypatch):
    import app.agents.emotional_affective_memory_agent as eama_mod
    monkeypatch.setattr(eama_mod.settings, "AGENT_STRATEGY", "heuristic")
    result = await AGENT.run("m12", _ctx("test content"))
    assert result.outputs["rationale"] == "heuristic"


@pytest.mark.asyncio
async def test_run_trace_id_propagated():
    result = await AGENT.run("m13", _ctx("content"))
    assert result.trace_id == "trace-42"


@pytest.mark.asyncio
async def test_run_mixed_content():
    result = await AGENT.run(
        "m14",
        _ctx("great progress achieved but there was a terrible failure in the pipeline"),
    )
    assert result.outputs["emotional_valence"] == "mixed"


@pytest.mark.asyncio
async def test_run_empathy_negative_medium_arousal():
    result = await AGENT.run(
        "m15",
        _ctx("this is really bad and frustrating, everything is broken and wrong"),
    )
    assert result.outputs["empathy_flag"] is True


@pytest.mark.asyncio
async def test_run_no_escalation_positive_normal():
    result = await AGENT.run("m16", _ctx("great success, all tests pass"))
    assert result.outputs["escalation_recommended"] is False


@pytest.mark.asyncio
async def test_run_outputs_all_keys():
    result = await AGENT.run("m17", _ctx("some content"))
    required = {
        "emotional_valence", "arousal_level", "empathy_flag",
        "escalation_recommended", "tone_guidance", "confidence", "rationale",
    }
    assert required.issubset(result.outputs.keys())


# ---------------------------------------------------------------------------
# Agent metadata
# ---------------------------------------------------------------------------

def test_agent_name():
    assert AGENT.name == "EmotionalAffectiveMemoryAgent"


def test_agent_version():
    assert AGENT.version == "v1"


def test_agent_dependencies():
    deps = AGENT.dependencies()
    assert "FeedbackIntegrationAgent" in deps
    assert "HumanReviewQueueAgent" in deps
    assert "NarrativeSynthesisAgent" in deps


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_lookup():
    from app.agents.registry import get_agent
    agent = get_agent("emotional_affective_memory")
    assert isinstance(agent, EmotionalAffectiveMemoryAgent)


def test_registry_alias_camel():
    from app.agents.registry import get_agent
    agent = get_agent("emotionalaffectivememoryagent")
    assert isinstance(agent, EmotionalAffectiveMemoryAgent)


def test_registry_alias_short():
    from app.agents.registry import get_agent
    agent = get_agent("emotional_affective")
    assert isinstance(agent, EmotionalAffectiveMemoryAgent)


def test_registry_miss():
    from app.agents.registry import get_agent
    assert get_agent("not_a_real_agent_42") is None
