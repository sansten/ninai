from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.organization import Organization
from app.services.memory_provenance_service import MemoryProvenanceService


@pytest.mark.asyncio
class TestRecordEdge:
    async def test_record_edge_creates_row_with_fields(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        edge = await svc.record_edge(
            db=db_session,
            org_id=test_org_id,
            source_id="ingest:slack",
            target_id="mem-1",
            edge_type="ingest",
            agent_name="IngestService",
            metadata={"channel": "incidents"},
        )
        assert edge.org_id == test_org_id
        assert edge.source_id == "ingest:slack"
        assert edge.target_id == "mem-1"
        assert edge.edge_type == "ingest"
        assert edge.agent_name == "IngestService"
        assert edge.edge_metadata["channel"] == "incidents"

    async def test_record_edge_default_metadata_empty_dict(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        edge = await svc.record_edge(
            db=db_session,
            org_id=test_org_id,
            source_id="s",
            target_id="t",
            edge_type="enrichment",
            agent_name="A",
        )
        assert edge.edge_metadata == {}

    async def test_record_edge_stringifies_inputs(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        edge = await svc.record_edge(
            db=db_session,
            org_id=test_org_id,
            source_id=123,
            target_id=456,
            edge_type=789,
            agent_name=1011,
        )
        assert edge.source_id == "123"
        assert edge.target_id == "456"
        assert edge.edge_type == "789"
        assert edge.agent_name == "1011"


@pytest.mark.asyncio
class TestGetLineage:
    async def test_get_lineage_single_hop_root_source(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(
            db=db_session,
            org_id=test_org_id,
            source_id="ingest:slack",
            target_id="mem-1",
            edge_type="ingest",
            agent_name="IngestService",
        )
        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="mem-1")
        assert lineage["root_sources"] == ["ingest:slack"]
        assert lineage["depth"] == 1

    async def test_get_lineage_max_depth_limits_traversal(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="b", edge_type="ingest", agent_name="A")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="b", target_id="c", edge_type="enrichment", agent_name="B")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="c", target_id="d", edge_type="writeback", agent_name="C")

        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="d", max_depth=1)
        assert lineage["depth"] == 1
        assert lineage["root_sources"] == ["c"]

    async def test_agent_chain_ordered_root_to_target(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="root", target_id="mid", edge_type="ingest", agent_name="RootAgent")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="mid", target_id="leaf", edge_type="enrichment", agent_name="LeafAgent")

        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="leaf")
        assert lineage["agent_chain"] == ["RootAgent", "LeafAgent"]

    async def test_empty_lineage_returns_memory_as_root(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="missing")
        assert lineage["root_sources"] == ["missing"]
        assert lineage["depth"] == 0
        assert lineage["edges"] == []

    async def test_cycle_detection_with_depth_limit(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="b", edge_type="e1", agent_name="A1")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="b", target_id="c", edge_type="e2", agent_name="A2")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="c", target_id="a", edge_type="e3", agent_name="A3")

        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="c", max_depth=4)
        assert lineage["depth"] <= 4
        assert len(lineage["edges"]) >= 2

    async def test_lineage_edges_include_expected_keys(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="x", target_id="y", edge_type="ingest", agent_name="Agent")
        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="y")
        edge = lineage["edges"][0]
        assert set(edge) == {"id", "source_id", "target_id", "edge_type", "agent_name", "created_at", "metadata"}

    async def test_lineage_scoped_by_org(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        other_org = str(uuid4())
        db_session.add(Organization(id=other_org, name="Other Org", slug=f"other-{other_org[:8]}", is_active=True))
        await db_session.flush()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="s1", target_id="t1", edge_type="ingest", agent_name="A")
        await svc.record_edge(db=db_session, org_id=other_org, source_id="s2", target_id="t1", edge_type="ingest", agent_name="B")

        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="t1")
        assert lineage["root_sources"] == ["s1"]

    async def test_lineage_multiple_roots_sorted(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="r2", target_id="m", edge_type="ingest", agent_name="A")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="r1", target_id="m", edge_type="ingest", agent_name="B")

        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="m")
        assert lineage["root_sources"] == ["r1", "r2"]

    async def test_lineage_depth_two_for_two_hops(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="b", edge_type="ingest", agent_name="A")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="b", target_id="c", edge_type="enrichment", agent_name="B")
        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="c")
        assert lineage["depth"] == 2

    async def test_lineage_negative_depth_treated_zero(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="b", edge_type="ingest", agent_name="A")
        lineage = await svc.get_lineage(db=db_session, org_id=test_org_id, memory_id="b", max_depth=-1)
        assert lineage["depth"] == 0
        assert lineage["root_sources"] == ["b"]


@pytest.mark.asyncio
class TestGetDescendants:
    async def test_get_descendants_all_reachable_targets(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="b", edge_type="ingest", agent_name="A")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="b", target_id="c", edge_type="enrichment", agent_name="B")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="d", edge_type="writeback", agent_name="C")

        descendants = await svc.get_descendants(db=db_session, org_id=test_org_id, source_id="a")
        assert descendants == ["b", "c", "d"]

    async def test_get_descendants_empty_when_no_edges(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        assert await svc.get_descendants(db=db_session, org_id=test_org_id, source_id="none") == []

    async def test_get_descendants_scoped_by_org(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        other_org = str(uuid4())
        db_session.add(Organization(id=other_org, name="Other Org 2", slug=f"other2-{other_org[:8]}", is_active=True))
        await db_session.flush()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="b", edge_type="ingest", agent_name="A")
        await svc.record_edge(db=db_session, org_id=other_org, source_id="a", target_id="x", edge_type="ingest", agent_name="X")

        descendants = await svc.get_descendants(db=db_session, org_id=test_org_id, source_id="a")
        assert descendants == ["b"]

    async def test_get_descendants_handles_cycle(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="b", edge_type="e1", agent_name="A1")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="b", target_id="a", edge_type="e2", agent_name="A2")

        descendants = await svc.get_descendants(db=db_session, org_id=test_org_id, source_id="a")
        assert descendants == ["b"]

    async def test_get_descendants_sorted(self, db_session: AsyncSession, test_org_id: str):
        svc = MemoryProvenanceService()
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="z", edge_type="e", agent_name="A")
        await svc.record_edge(db=db_session, org_id=test_org_id, source_id="a", target_id="m", edge_type="e", agent_name="A")
        descendants = await svc.get_descendants(db=db_session, org_id=test_org_id, source_id="a")
        assert descendants == ["m", "z"]


class TestSummariseLineage:
    def test_summarise_contains_agent_names_in_order(self):
        svc = MemoryProvenanceService()
        text = svc.summarise_lineage({"agent_chain": ["CredibilityAgent", "MemoryConsolidationAgent"]})
        assert "CredibilityAgent" in text
        assert "MemoryConsolidationAgent" in text
        assert text.index("CredibilityAgent") < text.index("MemoryConsolidationAgent")

    def test_summarise_empty_lineage_message(self):
        svc = MemoryProvenanceService()
        assert svc.summarise_lineage({"agent_chain": []}) == "No provenance lineage available"

    def test_summarise_single_agent(self):
        svc = MemoryProvenanceService()
        text = svc.summarise_lineage({"agent_chain": ["IngestService"]})
        assert text == "enriched by IngestService"

    def test_summarise_missing_agent_chain(self):
        svc = MemoryProvenanceService()
        assert svc.summarise_lineage({}) == "No provenance lineage available"


class TestServiceSanity:
    def test_service_instantiates(self):
        svc = MemoryProvenanceService()
        assert isinstance(svc, MemoryProvenanceService)

    def test_summarise_deterministic(self):
        svc = MemoryProvenanceService()
        lineage = {"agent_chain": ["A", "B", "C"]}
        assert svc.summarise_lineage(lineage) == svc.summarise_lineage(lineage)

    def test_summarise_uses_arrow_separator(self):
        svc = MemoryProvenanceService()
        text = svc.summarise_lineage({"agent_chain": ["A", "B"]})
        assert "->" in text

    def test_summarise_phrase_prefix(self):
        svc = MemoryProvenanceService()
        text = svc.summarise_lineage({"agent_chain": ["A"]})
        assert text.startswith("enriched by")
