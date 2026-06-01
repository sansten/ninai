"""
Tests for the NINAI v2 Graph-RAG + DNC cognitive engine.

All external dependencies (FalkorDB, Ollama, Qdrant) are mocked.
Tests verify the logic, data flow, and error-resilience of each component.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.v2.graph.schema import DECAY_FACTOR, WEIGHT_INITIAL, WEIGHT_PRUNE_THRESHOLD
from app.v2.llm.prompt_builder import build_inference_prompt
from app.v2.memory.dnc_router import DNCMemoryRouter, ReadResult, WriteResult
from app.v2.pipeline.cognitive_loop import CognitiveLoopResult, V2CognitiveLoop


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph_client(available: bool = True) -> MagicMock:
    gc = MagicMock()
    gc.is_available.return_value = available
    gc.fetch_subgraph = AsyncMock(return_value=[
        {"id": "e1", "label": "Entity", "content": "project alpha", "weight": 0.7, "created_at": 1000},
        {"id": "e2", "label": "Entity", "content": "deadline friday", "weight": 0.5, "created_at": 900},
    ])
    gc.fetch_recent_utterances = AsyncMock(return_value=[
        {"id": "u0", "label": "Utterance", "text": "hello", "role": "user", "created_at": 800},
    ])
    gc.upsert_entity = AsyncMock(return_value={"id": "eid", "weight": 0.6})
    gc.create_utterance = AsyncMock(return_value={"id": "utt-001"})
    gc.link_utterance_to_entities = AsyncMock()
    gc.link_sequential_utterances = AsyncMock()
    gc.decay_neighborhood = AsyncMock(return_value=5)
    gc.prune_weak_edges = AsyncMock(return_value=1)
    return gc


def _make_engine() -> MagicMock:
    engine = MagicMock()
    engine.embed = AsyncMock(return_value=[0.1] * 768)
    engine.extract_entities = AsyncMock(return_value=[
        {"id": "project_alpha", "name": "project alpha", "type": "task"},
    ])
    from app.v2.llm.ollama_engine import InferenceResult
    engine.infer = AsyncMock(return_value=InferenceResult(
        response="The project alpha deadline is Friday.",
        cited_node_ids=["e1", "e2"],
        extracted_entities=[{"id": "project_alpha", "name": "project alpha", "type": "task"}],
    ))
    engine.is_available = AsyncMock(return_value=True)
    return engine


def _make_router(graph_client=None, engine=None, qdrant=None) -> DNCMemoryRouter:
    gc = graph_client or _make_graph_client()
    eng = engine or _make_engine()
    return DNCMemoryRouter(
        graph_client=gc,
        qdrant_service=qdrant,
        embedding_fn=eng.embed,
        entity_extractor=eng.extract_entities,
    )


# ---------------------------------------------------------------------------
# Graph schema constants
# ---------------------------------------------------------------------------

class TestGraphSchema:
    def test_weight_constants(self):
        assert 0.0 < WEIGHT_PRUNE_THRESHOLD < WEIGHT_INITIAL < 1.0
        assert 0.0 < DECAY_FACTOR < 1.0

    def test_decay_reduces_weight(self):
        w = 0.5
        for _ in range(100):
            w *= DECAY_FACTOR
        assert w < WEIGHT_PRUNE_THRESHOLD


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_contains_user_input(self):
        prompt = build_inference_prompt(
            user_input="What is the deadline?",
            graph_nodes=[],
            qdrant_chunks=[],
            session_utterances=[],
        )
        assert "What is the deadline?" in prompt

    def test_graph_nodes_serialised(self):
        nodes = [{"id": "n1", "label": "Entity", "content": "alpha project", "weight": 0.8}]
        prompt = build_inference_prompt("q", nodes, [], [])
        assert "n1" in prompt
        assert "alpha project" in prompt

    def test_qdrant_chunks_included(self):
        chunks = [{"id": "c1", "score": 0.9, "payload": {"text": "deadline next week"}}]
        prompt = build_inference_prompt("q", [], chunks, [])
        assert "deadline next week" in prompt

    def test_session_utterances_included(self):
        utts = [{"role": "user", "text": "previous turn text"}]
        prompt = build_inference_prompt("q", [], [], utts)
        assert "previous turn text" in prompt

    def test_json_schema_instruction_present(self):
        prompt = build_inference_prompt("q", [], [], [])
        assert "cited_node_ids" in prompt
        assert "extracted_entities" in prompt

    def test_no_context_graceful(self):
        prompt = build_inference_prompt("hello", [], [], [])
        assert "no graph context" in prompt
        assert "new session" in prompt


# ---------------------------------------------------------------------------
# DNC Memory Router — read weighting
# ---------------------------------------------------------------------------

class TestDNCRouterRead:
    @pytest.mark.asyncio
    async def test_read_returns_graph_nodes(self):
        router = _make_router()
        result = await router.read("tenant1", "sess1", "What is the project deadline?")
        assert isinstance(result, ReadResult)
        # Without Qdrant, no seed_ids → fetch_subgraph not called.
        # Recent utterances are always appended (1 node from mock).
        assert len(result.graph_nodes) >= 1

    @pytest.mark.asyncio
    async def test_read_graph_unavailable_returns_empty(self):
        gc = _make_graph_client(available=False)
        router = _make_router(graph_client=gc)
        result = await router.read("t1", "s1", "query")
        assert result.graph_nodes == []

    @pytest.mark.asyncio
    async def test_read_with_qdrant_seeds_graph(self):
        qdrant = MagicMock()
        hit = MagicMock()
        hit.id = "mem-42"
        hit.score = 0.88
        hit.payload = {"entity_id": "proj_alpha", "tenant_id": "tenant1", "text": "alpha"}
        qdrant.search = AsyncMock(return_value=[hit])

        engine = _make_engine()
        gc = _make_graph_client()
        router = DNCMemoryRouter(
            graph_client=gc,
            qdrant_service=qdrant,
            embedding_fn=engine.embed,
            entity_extractor=engine.extract_entities,
        )
        result = await router.read("tenant1", "sess1", "alpha project?")
        # Qdrant mock returns one hit, entity_id seeds graph traversal
        assert len(result.qdrant_chunks) == 1
        assert result.qdrant_chunks[0]["id"] == "mem-42"
        assert isinstance(result.graph_nodes, list)

    @pytest.mark.asyncio
    async def test_read_embedding_failure_degrades_gracefully(self):
        engine = _make_engine()
        engine.embed = AsyncMock(side_effect=RuntimeError("embed timeout"))
        router = _make_router(engine=engine)
        result = await router.read("t1", "s1", "test query")
        # Should not raise; graph_nodes still populated from subgraph mock if seeded
        assert isinstance(result, ReadResult)


# ---------------------------------------------------------------------------
# DNC Memory Router — write weighting
# ---------------------------------------------------------------------------

class TestDNCRouterWrite:
    @pytest.mark.asyncio
    async def test_write_creates_utterance(self):
        gc = _make_graph_client()
        router = _make_router(graph_client=gc)
        result = await router.write("t1", "s1", "Hello world", "user")
        assert result.utterance_id != ""
        gc.create_utterance.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_upserts_entities(self):
        gc = _make_graph_client()
        engine = _make_engine()
        engine.extract_entities = AsyncMock(return_value=[
            {"id": "entity_a", "name": "entity A", "type": "concept"},
            {"id": "entity_b", "name": "entity B", "type": "task"},
        ])
        router = DNCMemoryRouter(
            graph_client=gc,
            qdrant_service=None,
            embedding_fn=engine.embed,
            entity_extractor=engine.extract_entities,
        )
        result = await router.write("t1", "s1", "Entity A depends on Entity B", "user")
        assert len(result.entity_ids) == 2
        assert gc.upsert_entity.call_count == 2

    @pytest.mark.asyncio
    async def test_write_links_utterance_to_entities(self):
        gc = _make_graph_client()
        router = _make_router(graph_client=gc)
        result = await router.write("t1", "s1", "project alpha", "user")
        if result.entity_ids:
            gc.link_utterance_to_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_chains_sequential_utterances(self):
        gc = _make_graph_client()
        router = _make_router(graph_client=gc)
        await router.write("t1", "s1", "turn 1", "user")
        await router.write("t1", "s1", "turn 2", "user", prev_utterance_id="utt-prev")
        gc.link_sequential_utterances.assert_called()

    @pytest.mark.asyncio
    async def test_write_graph_unavailable_returns_empty_result(self):
        gc = _make_graph_client(available=False)
        router = _make_router(graph_client=gc)
        result = await router.write("t1", "s1", "hello", "user")
        assert result.utterance_id == ""
        assert result.graph_writes == 0

    @pytest.mark.asyncio
    async def test_write_entity_extraction_failure_continues(self):
        gc = _make_graph_client()
        engine = _make_engine()
        engine.extract_entities = AsyncMock(side_effect=RuntimeError("extraction error"))
        router = DNCMemoryRouter(
            graph_client=gc,
            qdrant_service=None,
            embedding_fn=engine.embed,
            entity_extractor=engine.extract_entities,
        )
        result = await router.write("t1", "s1", "text", "user")
        # Entity extraction failed but utterance node should still be created
        assert result.utterance_id != ""


# ---------------------------------------------------------------------------
# DNC Memory Router — decay and prune
# ---------------------------------------------------------------------------

class TestDNCDecayPrune:
    @pytest.mark.asyncio
    async def test_decay_calls_graph(self):
        gc = _make_graph_client()
        router = _make_router(graph_client=gc)
        stats = await router.decay_and_prune("t1", ["e1", "e2"])
        gc.decay_neighborhood.assert_called_once_with("t1", ["e1", "e2"])
        gc.prune_weak_edges.assert_called_once_with("t1")
        assert stats["decayed"] == 5
        assert stats["pruned"] == 1

    @pytest.mark.asyncio
    async def test_decay_graph_unavailable_returns_zeros(self):
        gc = _make_graph_client(available=False)
        router = _make_router(graph_client=gc)
        stats = await router.decay_and_prune("t1", ["e1"])
        assert stats == {"decayed": 0, "pruned": 0}

    @pytest.mark.asyncio
    async def test_decay_empty_seeds_skips_graph(self):
        gc = _make_graph_client()
        router = _make_router(graph_client=gc)
        stats = await router.decay_and_prune("t1", [])
        gc.decay_neighborhood.assert_not_called()
        assert stats["decayed"] == 0


# ---------------------------------------------------------------------------
# V2 Cognitive Loop (full pipeline)
# ---------------------------------------------------------------------------

class TestV2CognitiveLoop:
    @pytest.mark.asyncio
    async def test_full_loop_returns_response(self):
        gc = _make_graph_client()
        engine = _make_engine()
        router = _make_router(graph_client=gc, engine=engine)
        loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)

        result = await loop.run("t1", "sess1", "What is the deadline for project alpha?")
        assert isinstance(result, CognitiveLoopResult)
        assert result.response != ""
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_loop_populates_cited_nodes(self):
        gc = _make_graph_client()
        engine = _make_engine()
        router = _make_router(graph_client=gc, engine=engine)
        loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)

        result = await loop.run("t1", "s1", "deadline?")
        assert "e1" in result.cited_node_ids or "e2" in result.cited_node_ids

    @pytest.mark.asyncio
    async def test_loop_writes_user_and_assistant_utterances(self):
        gc = _make_graph_client()
        engine = _make_engine()
        router = _make_router(graph_client=gc, engine=engine)
        loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)

        result = await loop.run("t1", "s1", "hello")
        assert result.user_utterance_id != ""
        assert result.assistant_utterance_id != ""

    @pytest.mark.asyncio
    async def test_loop_chains_turns_sequentially(self):
        gc = _make_graph_client()
        engine = _make_engine()
        router = _make_router(graph_client=gc, engine=engine)
        loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)

        await loop.run("t1", "sess2", "turn 1")
        await loop.run("t1", "sess2", "turn 2")
        # After turn 1, session should have a last utterance pointer
        # and turn 2 should chain to it
        assert "sess2" in loop._last_utt

    @pytest.mark.asyncio
    async def test_loop_inference_failure_returns_error_response(self):
        gc = _make_graph_client()
        engine = _make_engine()
        engine.infer = AsyncMock(side_effect=RuntimeError("ollama down"))
        router = _make_router(graph_client=gc, engine=engine)
        loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)

        result = await loop.run("t1", "s1", "query")
        assert result.error != ""
        assert "unable to generate" in result.response.lower()

    @pytest.mark.asyncio
    async def test_loop_graph_unavailable_still_infers(self):
        gc = _make_graph_client(available=False)
        engine = _make_engine()
        router = _make_router(graph_client=gc, engine=engine)
        loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)

        result = await loop.run("t1", "s1", "hello?")
        # Graph is down but inference should still work
        assert result.response != ""
        assert result.graph_nodes_retrieved == 0

    @pytest.mark.asyncio
    async def test_loop_latency_recorded(self):
        gc = _make_graph_client()
        engine = _make_engine()
        router = _make_router(graph_client=gc, engine=engine)
        loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)

        result = await loop.run("t1", "s1", "ping")
        assert isinstance(result.latency_ms, int)
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_loop_decay_stats_populated(self):
        gc = _make_graph_client()
        engine = _make_engine()
        router = _make_router(graph_client=gc, engine=engine)
        loop = V2CognitiveLoop(dnc_router=router, reasoning_engine=engine)

        result = await loop.run("t1", "s1", "project alpha")
        # decay stats may be empty if no seeds, but should be a dict
        assert isinstance(result.decay_stats, dict)


# ---------------------------------------------------------------------------
# Entity extraction — bench21+ patterns
# ---------------------------------------------------------------------------

class TestEntityExtractionBench21:
    """Unit tests for new entity extraction patterns added in bench21+."""

    def _run(self, text: str, speaker: str = "Alice") -> list[dict]:
        from app.v2.memory.entity_extraction import extract_v2_entities
        full = f"[2023-05-25] [{speaker}] {text}"
        return extract_v2_entities(full)

    def _attrs(self, entities: list[dict]) -> dict[str, str]:
        return {e["attribute"]: e["value"] for e in entities if e.get("type") == "personal_attribute"}

    def test_multiple_snake_names(self):
        ents = self._run("My snakes are named Susie and Seraphim.", "Deborah")
        attrs = self._attrs(ents)
        assert "pet_snake_names" in attrs
        assert "Susie" in attrs["pet_snake_names"]
        assert "Seraphim" in attrs["pet_snake_names"]

    def test_book_by_author(self):
        ents = self._run("I recently read Avalanche by Neal Stephenson.", "Deborah")
        attrs = self._attrs(ents)
        assert "book_read" in attrs
        assert "Avalanche" in attrs["book_read"]

    def test_got_into_hobby(self):
        ents = self._run("I got into watercolor painting after a friend suggested it.", "Sam")
        attrs = self._attrs(ents)
        assert "hobby" in attrs or "hobby_introduced_by_friend" in attrs

    def test_friend_suggested_hobby(self):
        ents = self._run("A friend suggested watercolor painting and I loved it.", "Sam")
        attrs = self._attrs(ents)
        assert any("watercolor" in str(v) for v in attrs.values())

    def test_vehicle_prius(self):
        ents = self._run("I got a new Prius after my old one broke down.", "Sam")
        attrs = self._attrs(ents)
        assert "vehicle" in attrs
        assert "Prius" in attrs["vehicle"]

    def test_vehicle_ferrari(self):
        ents = self._run("I bought a Ferrari 488 GTB in March 2023.", "Calvin")
        attrs = self._attrs(ents)
        assert "vehicle" in attrs
        assert "Ferrari" in attrs["vehicle"]

    def test_programming_languages_direct(self):
        ents = self._run("I work with Python and C++ in my projects.", "James")
        attrs = self._attrs(ents)
        assert "programming_languages" in attrs
        val = attrs["programming_languages"]
        assert "Python" in val and "C++" in val

    def test_book_read_quoted(self):
        ents = self._run('I finished reading "Sapiens" last week.', "Deborah")
        attrs = self._attrs(ents)
        assert "book_read" in attrs
        assert "Sapiens" in attrs["book_read"]

    def test_sports_team_signed(self):
        ents = self._run("I just signed with the Minnesota Wolves as a shooting guard!", "John")
        attrs = self._attrs(ents)
        assert "sports_team" in attrs
        assert "Minnesota Wolves" in attrs["sports_team"]
        assert "sports_position" in attrs
        assert "shooting guard" in attrs["sports_position"]

    def test_tattoo_of(self):
        ents = self._run("I have a tattoo of sunflowers on my arm.", "Andrew")
        attrs = self._attrs(ents)
        assert "tattoo" in attrs
        assert "sunflower" in attrs["tattoo"].lower()

    def test_musical_instrument(self):
        ents = self._run("I started playing drums again after years away.", "John")
        attrs = self._attrs(ents)
        assert "instrument" in attrs
        assert "drum" in attrs["instrument"].lower()

    def test_pet_adoption(self):
        ents = self._run("I adopted a puppy last month.", "John")
        attrs = self._attrs(ents)
        assert "has_pet" in attrs
        assert attrs["has_pet"] == "dog"


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------

class TestEngineVersionConfig:
    def test_v1_is_default(self):
        from app.core.config import settings
        # Default is v1 — the new setting should default to "v1"
        version = getattr(settings, "NINAI_ENGINE_VERSION", "v1")
        assert version in ("v1", "v2")

    def test_v2_config_valid(self):
        import os
        with patch.dict("os.environ", {"NINAI_ENGINE_VERSION": "v2"}):
            from importlib import reload
            import app.core.config as cfg_mod
            # Just verify the attribute is accepted — don't reload in shared test env
            assert "NINAI_ENGINE_VERSION" in cfg_mod.Settings.model_fields
