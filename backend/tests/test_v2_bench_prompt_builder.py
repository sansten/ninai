from __future__ import annotations

from app.v2.llm.prompt_builder import build_bench_prompt, _compute_query_analysis


def test_bench_prompt_includes_target_people_hint() -> None:
    prompt = build_bench_prompt(
        user_input="Answer in one concise phrase based on the conversation history: What is Caroline's identity?",
        graph_nodes=[],
        qdrant_chunks=[],
        session_utterances=[],
    )
    assert "TARGET PEOPLE: Caroline" in prompt


def test_bench_prompt_prioritizes_target_profile_before_other_people() -> None:
    prompt = build_bench_prompt(
        user_input="Answer in one concise phrase based on the conversation history: What is Caroline's identity?",
        graph_nodes=[
            {"entity_type": "person_profile", "subject": "Melanie", "content": "artist | runner"},
            {"entity_type": "person_profile", "subject": "Caroline", "content": "transgender woman | counselor"},
        ],
        qdrant_chunks=[],
        session_utterances=[],
    )
    assert prompt.index("[Caroline]:") < prompt.index("[Melanie]:")


def test_bench_prompt_prioritizes_question_matching_chunk_for_same_person() -> None:
    prompt = build_bench_prompt(
        user_input=(
            "Answer in one concise phrase based on the conversation history: "
            "What are Melanie's plans for the summer with respect to adoption?"
        ),
        graph_nodes=[],
        qdrant_chunks=[
            {"payload": {"speaker": "Melanie", "text": "Melanie plans to attend LGBTQ events this summer."}},
            {"payload": {"speaker": "Melanie", "text": "Melanie is researching adoption agencies this summer."}},
        ],
        session_utterances=[],
    )
    assert prompt.index("researching adoption agencies") < prompt.index("attend LGBTQ events")


# ---------------------------------------------------------------------------
# _compute_query_analysis unit tests
# ---------------------------------------------------------------------------

def _chunks(texts_and_dates: list[tuple[str, str]]) -> list[dict]:
    return [{"payload": {"text": t, "anchor_date": d}} for t, d in texts_and_dates]


def _nodes(texts_and_dates: list[tuple[str, str]]) -> list[dict]:
    return [{"content": t, "anchor_date": d} for t, d in texts_and_dates]


# ── Fix 3: temporal ordering ──────────────────────────────────────────────

def test_compute_temporal_ordering_sorts_by_anchor_date() -> None:
    chunks = _chunks([
        ("I bought the Dell XPS 13 last week.", "2023-03-20"),
        ("I just got a Samsung Galaxy S22.", "2023-02-15"),
    ])
    result = _compute_query_analysis(
        "Which device did I get first, the Samsung Galaxy S22 or the Dell XPS 13?",
        graph_nodes=[],
        qdrant_chunks=chunks,
    )
    assert "COMPUTED ANALYSIS" in result
    # S22 date (2023-02-15) should appear before XPS date (2023-03-20) in sorted list
    assert result.index("2023-02-15") < result.index("2023-03-20")


def test_compute_temporal_ordering_returns_empty_for_non_temporal_question() -> None:
    chunks = _chunks([("I love hiking.", "2023-01-01")])
    result = _compute_query_analysis("What is my favourite hobby?", graph_nodes=[], qdrant_chunks=chunks)
    assert "Chronological" not in result


# ── Fix 4: date arithmetic ────────────────────────────────────────────────

def test_compute_date_arith_from_anchor_dates() -> None:
    chunks = _chunks([
        ("I attended Sunday mass at St. Mary's Church.", "2023-01-02"),
        ("I went to the Ash Wednesday service.", "2023-02-01"),
    ])
    result = _compute_query_analysis(
        "How many days had passed between the Sunday mass and the Ash Wednesday service?",
        graph_nodes=[],
        qdrant_chunks=chunks,
    )
    assert "COMPUTED ANALYSIS" in result
    assert "30 days" in result


def test_compute_date_arith_from_prose_dates() -> None:
    # When anchor_dates are identical, fall back to prose month+day extraction
    chunks = _chunks([
        ("I attended the workshop on January 10th.", "2023-01-13"),
        ("I have a team meeting on January 17th.", "2023-01-13"),
    ])
    result = _compute_query_analysis(
        "How many days before the team meeting did I attend the workshop?",
        graph_nodes=[],
        qdrant_chunks=chunks,
    )
    assert "COMPUTED ANALYSIS" in result
    assert "7 days" in result


def test_compute_date_arith_not_triggered_for_unrelated_question() -> None:
    chunks = _chunks([("I went hiking yesterday.", "2023-05-01")])
    result = _compute_query_analysis("What is my favourite food?", graph_nodes=[], qdrant_chunks=chunks)
    assert "days" not in result.lower() or "COMPUTED" not in result


# ── Numeric aggregation excluded (money/duration sums removed) ────────────
# Money and duration sums are not injected because multi-session retrieval is
# almost always incomplete — a partial sum misleads the LLM into confidently
# reporting the wrong total. Rule 18 handles counting from full context.

def test_compute_money_sum_not_injected() -> None:
    # Sums from retrieved chunks should NOT appear — retrieval is often partial
    chunks = _chunks([
        ("I spent $40 on a new bike lock.", "2023-01-10"),
        ("I bought cycling shoes for $120.", "2023-02-01"),
        ("I got a helmet for $25.", "2023-03-05"),
    ])
    result = _compute_query_analysis(
        "How much total money have I spent on bike-related expenses?",
        graph_nodes=[],
        qdrant_chunks=chunks,
    )
    assert "Monetary amounts" not in result


def test_compute_duration_sum_not_injected() -> None:
    chunks = _chunks([
        ("I just got back from a 3-day solo camping trip to Big Sur.", "2023-04-05"),
        ("I spent 5 days camping in Yosemite this year.", "2023-06-15"),
    ])
    result = _compute_query_analysis(
        "How many days did I spend on camping trips in the United States this year?",
        graph_nodes=[],
        qdrant_chunks=chunks,
    )
    assert "Duration expressions" not in result


# ── Fix 7: comparative / average ─────────────────────────────────────────

def test_compute_average_age() -> None:
    chunks = _chunks([
        ("My parents are 60 years old and 65 years old.", "2023-01-01"),
        ("My grandparents are 85 years old and 90 years old.", "2023-01-01"),
        ("I am 30 years old.", "2023-01-01"),
    ])
    result = _compute_query_analysis(
        "What is the average age of me, my parents, and my grandparents?",
        graph_nodes=[],
        qdrant_chunks=chunks,
    )
    assert "COMPUTED ANALYSIS" in result
    assert "Average" in result


def test_compute_percentage_found() -> None:
    chunks = _chunks([("I got a 20% discount on the book.", "2023-03-01")])
    result = _compute_query_analysis(
        "What percentage discount did I get on the book?",
        graph_nodes=[],
        qdrant_chunks=chunks,
    )
    assert "COMPUTED ANALYSIS" in result
    assert "20%" in result


# ── Integration: COMPUTED ANALYSIS appears in full prompt ────────────────

def test_bench_prompt_no_computed_analysis_for_money_question() -> None:
    # Money sums are excluded — partial retrieval would give wrong totals
    chunks = _chunks([
        ("I spent $40 on a bike lock.", "2023-01-10"),
        ("I bought a helmet for $145.", "2023-02-01"),
    ])
    prompt = build_bench_prompt(
        user_input="How much total have I spent on bike expenses?",
        graph_nodes=[],
        qdrant_chunks=chunks,
        session_utterances=[],
    )
    assert "Monetary amounts" not in prompt


def test_bench_prompt_no_computed_analysis_for_simple_question() -> None:
    prompt = build_bench_prompt(
        user_input="What is Caroline's favourite hobby?",
        graph_nodes=[{"entity_type": "person_profile", "subject": "Caroline", "content": "likes hiking"}],
        qdrant_chunks=[],
        session_utterances=[],
    )
    # No data block should appear — the unique sentinel is the "(pre-computed" prefix
    # which only appears in the injected data header, not in rule 19.
    assert "(pre-computed from retrieved memory" not in prompt


def test_compute_returns_empty_when_no_retrieved_content() -> None:
    result = _compute_query_analysis(
        "How many days did I spend camping?",
        graph_nodes=[],
        qdrant_chunks=[],
    )
    assert result == ""
