"""Graph-connectivity render boost (multi-hop bridge promotion).

Covers the _apply_graph_connectivity_boost pass and the prompt builder honouring the
precomputed payload['_graph_bonus'] in its render ranking.
"""

from app.v2.pipeline.cognitive_loop import _apply_graph_connectivity_boost
from app.v2.llm.prompt_builder import build_bench_prompt


def _chunk(cid, text, entity_ids=None, entity_names=None, subject=""):
    return {
        "id": cid,
        "score": 0.0,
        "payload": {
            "text": text,
            "subject": subject,
            "entity_ids": entity_ids or [],
            "entity_names": entity_names or [],
        },
    }


def test_boost_promotes_graph_connected_bridge():
    # Anchor chunk mentions entity 'counseling_cert'; a low-lexical bridge shares it.
    chunks = [
        _chunk("a", "Caroline got her counseling certification", entity_ids=["counseling_cert", "caroline"]),
        _chunk("b", "She is excited about the next chapter", entity_ids=["counseling_cert"]),
        _chunk("c", "Unrelated note about the weather", entity_ids=["weather"]),
    ]
    boosted = _apply_graph_connectivity_boost(chunks, anchor_top_k=1, bonus_weight=8)
    assert boosted >= 1
    # The bridge (b) shares the discriminative 'counseling_cert' entity with the anchor.
    assert chunks[1]["payload"].get("_graph_bonus", 0) > 0
    # The unrelated chunk gets nothing.
    assert chunks[2]["payload"].get("_graph_bonus", 0) == 0


def test_ubiquitous_entity_is_ignored():
    # 'caroline' appears in every chunk → no discriminative signal, so no boost.
    chunks = [
        _chunk(str(i), f"note {i}", entity_ids=["caroline"]) for i in range(10)
    ]
    boosted = _apply_graph_connectivity_boost(chunks, anchor_top_k=3, bonus_weight=8)
    assert boosted == 0
    assert all(c["payload"].get("_graph_bonus", 0) == 0 for c in chunks)


def test_boost_is_noop_without_entity_metadata():
    chunks = [_chunk("a", "text one"), _chunk("b", "text two")]
    assert _apply_graph_connectivity_boost(chunks) == 0
    assert all("_graph_bonus" not in c["payload"] for c in chunks)


def test_boost_handles_empty_input():
    assert _apply_graph_connectivity_boost([]) == 0


def test_prompt_builder_honours_graph_bonus():
    # Two regular chunks with equal (zero) lexical overlap to the question; the one with
    # a graph bonus must be rendered. Build with only one render slot's worth of signal
    # by making the bonus the tiebreaker.
    question = "What did the team decide?"
    low = _chunk("low", "alpha beta gamma delta", entity_ids=["x"])
    high = _chunk("high", "epsilon zeta eta theta", entity_ids=["x"])
    high["payload"]["_graph_bonus"] = 50

    prompt = build_bench_prompt(
        user_input=question,
        graph_nodes=[],
        qdrant_chunks=[low, high],
        session_utterances=[],
    )
    # The boosted chunk's text appears before the unboosted one in the rendered record.
    assert "epsilon zeta eta theta" in prompt
    assert prompt.index("epsilon zeta eta theta") < prompt.index("alpha beta gamma delta")
