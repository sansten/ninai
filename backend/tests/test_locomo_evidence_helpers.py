from __future__ import annotations

import sys
from pathlib import Path


_NOTEBOOKS_DIR = Path(__file__).resolve().parents[2] / "notebooks"
if str(_NOTEBOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NOTEBOOKS_DIR))

from locomo_evidence import (  # noqa: E402
    build_evidence_block,
    build_evidence_state,
    classify_failure_layer,
)


def test_build_evidence_state_collects_temporal_and_fact_hints():
    hits = [
        {
            "content": "I started the new job last month and moved to Seattle.",
            "occurred_at": "2023-06-09T10:00:00Z",
            "entities": {"place": ["Seattle"]},
        },
        {
            "content": "The week before, I wrapped up my old apartment lease.",
            "occurred_at": "2023-06-09T10:05:00Z",
        },
    ]

    state = build_evidence_state(
        "When did she start the new job?",
        "temporal",
        hits,
    )

    assert state["question_terms"]
    assert state["session_dates"] == ["2023-06-09"]
    assert "last month" in state["relative_time_markers"]
    assert state["fact_candidates"]
    assert "Seattle" in state["top_entities"]


def test_build_evidence_block_surfaces_bridge_and_perspective_cues():
    hits = [
        {
            "content": "I met Maya at the pottery studio after work.",
            "occurred_at": "2023-05-01T09:00:00Z",
            "entities": {"people": ["Maya"], "activity": ["pottery"]},
        },
        {
            "content": "Maya later invited me to the gallery opening downtown.",
            "occurred_at": "2023-05-20T09:00:00Z",
            "entities": {"people": ["Maya"], "place": ["gallery"]},
        },
    ]

    state = build_evidence_state(
        "What might they have in common?",
        "multi_hop",
        hits,
    )
    state["first_person_turn_count"] = 2
    block = build_evidence_block(state)

    assert "Bridge entities: Maya" in block
    assert "Candidate facts:" in block
    assert "Perspective cues: 2 first-person turn(s)" in block


def test_classify_failure_layer_distinguishes_format_and_bridge_misses():
    format_only = classify_failure_layer({
        "category": "single_hop",
        "semantic_correct": 1,
        "rouge1_f1": 18.0,
        "retrieval_has_gold": True,
        "evidence_state": {"fact_candidates": [{"text": "Sweden", "date": ""}]},
        "generated_answer": "She is from Sweden.",
    })
    bridge_miss = classify_failure_layer({
        "category": "multi_hop",
        "semantic_correct": 0,
        "rouge1_f1": 0.0,
        "retrieval_has_gold": True,
        "evidence_state": {"bridge_entities": []},
        "generated_answer": "no mention",
    })

    assert format_only == "format_only_miss"
    assert bridge_miss == "bridge_miss"
