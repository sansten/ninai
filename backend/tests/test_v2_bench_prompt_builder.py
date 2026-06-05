from __future__ import annotations

from app.v2.llm.prompt_builder import build_bench_prompt


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
