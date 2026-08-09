"""Unit tests for SchemaFormationService — Phase 87."""
from __future__ import annotations

import pytest

from app.services.schema_formation_service import (
    SchemaFormationService,
    SlotDefinition,
    _MIN_INSTANCES,
)


@pytest.fixture
def svc() -> SchemaFormationService:
    return SchemaFormationService()


def _mem(text: str, mem_id: str = "abc", confidence: float = 0.8) -> dict:
    return {"id": mem_id, "text": text, "confidence": confidence}


# ---------------------------------------------------------------------------
# extract_event_type
# ---------------------------------------------------------------------------

class TestExtractEventType:
    def test_acquisition(self, svc):
        assert svc.extract_event_type("Acme acquired TechCorp for $2B in 2023.") == "acquisition"

    def test_project_kickoff(self, svc):
        assert svc.extract_event_type("The team launched the new API platform.") == "project_kickoff"

    def test_decision(self, svc):
        assert svc.extract_event_type("The board approved the budget increase.") == "decision"

    def test_meeting(self, svc):
        assert svc.extract_event_type("Alice and Bob met to review the roadmap.") == "meeting"

    def test_milestone(self, svc):
        assert svc.extract_event_type("The team shipped version 2.0 on Friday.") == "milestone"

    def test_incident(self, svc):
        assert svc.extract_event_type("The database crashed at 3am UTC.") == "incident"

    def test_hiring(self, svc):
        assert svc.extract_event_type("Sarah joined as the new VP of Engineering.") == "hiring"

    def test_unknown_returns_none(self, svc):
        assert svc.extract_event_type("The sky is blue today.") is None


# ---------------------------------------------------------------------------
# induce_schema
# ---------------------------------------------------------------------------

class TestInduceSchema:
    def _acquisition_cluster(self, n: int = 4) -> list[dict]:
        templates = [
            "Acme Corp acquired TechStart for $500M in January 2022. CEO Alice approved.",
            "GlobalFund purchased InnovateCo for $1.2B in March 2022. CFO Bob signed off.",
            "MegaCorp bought DataLabs for $300M in June 2022. Director Carol led the deal.",
            "Alpha Inc acquired BetaSoft for $800M in September 2022. VP Dave approved.",
            "Horizon Ltd acquired CloudPeak for $2B in December 2022. CEO Eve announced it.",
        ]
        return [_mem(t, str(i)) for i, t in enumerate(templates[:n])]

    def test_returns_candidate_from_sufficient_cluster(self, svc):
        candidate = svc.induce_schema("acquisition", self._acquisition_cluster(4))
        assert candidate is not None

    def test_returns_none_below_min_instances(self, svc):
        candidate = svc.induce_schema("acquisition", self._acquisition_cluster(2))
        assert candidate is None

    def test_candidate_has_correct_event_type(self, svc):
        candidate = svc.induce_schema("acquisition", self._acquisition_cluster(4))
        assert candidate.event_type == "acquisition"

    def test_candidate_has_slots(self, svc):
        candidate = svc.induce_schema("acquisition", self._acquisition_cluster(4))
        assert len(candidate.slots) > 0

    def test_slot_types_include_entity_or_money(self, svc):
        candidate = svc.induce_schema("acquisition", self._acquisition_cluster(4))
        slot_names = {s.name for s in candidate.slots}
        assert slot_names & {"entity", "money", "date"}

    def test_instance_ids_populated(self, svc):
        cluster = self._acquisition_cluster(4)
        candidate = svc.induce_schema("acquisition", cluster)
        assert len(candidate.instance_ids) == 4

    def test_avg_confidence_in_range(self, svc):
        candidate = svc.induce_schema("acquisition", self._acquisition_cluster(4))
        assert 0.0 <= candidate.avg_confidence <= 1.0


# ---------------------------------------------------------------------------
# extract_slots_from_text
# ---------------------------------------------------------------------------

class TestExtractSlots:
    def test_extracts_entities(self, svc):
        slots = svc.extract_slots_from_text("Alice and Bob met at Microsoft HQ.")
        assert len(slots["entity"]) >= 2

    def test_extracts_dates(self, svc):
        slots = svc.extract_slots_from_text("The deal closed on 2022-06-15.")
        assert len(slots["date"]) >= 1

    def test_extracts_money(self, svc):
        slots = svc.extract_slots_from_text("Acme paid $500M for the startup.")
        assert len(slots["money"]) >= 1

    def test_extracts_roles(self, svc):
        slots = svc.extract_slots_from_text("The CEO and VP of Engineering signed the contract.")
        assert len(slots["role"]) >= 2


# ---------------------------------------------------------------------------
# complete_frame
# ---------------------------------------------------------------------------

class TestCompleteFrame:
    class _FakeSchema:
        slots = [
            {"name": "entity", "required": True},
            {"name": "date", "required": True},
            {"name": "money", "required": False},
        ]
        slot_distributions = {
            "entity": {"Alice": 5, "Bob": 3},
            "date": {"2022": 4, "2023": 2},
            "money": {"$500M": 3},
        }

    def test_fills_missing_required_slot_from_distribution(self, svc):
        schema = self._FakeSchema()
        result = svc.complete_frame(schema, partial_slots={"entity": "Carol"})
        assert "date" in result.inferred_slots
        assert result.inferred_slots["date"] == "2022"

    def test_preserves_provided_slots(self, svc):
        schema = self._FakeSchema()
        result = svc.complete_frame(schema, partial_slots={"entity": "Carol", "date": "2024"})
        assert result.filled_slots["entity"] == "Carol"
        assert result.filled_slots["date"] == "2024"

    def test_completeness_is_1_when_all_required_filled(self, svc):
        schema = self._FakeSchema()
        result = svc.complete_frame(
            schema,
            partial_slots={"entity": "Carol", "date": "2024"},
        )
        assert result.completeness == 1.0

    def test_completeness_is_partial_when_gap_remains(self, svc):
        schema = self._FakeSchema()
        # entity provided but date has no distribution and not provided
        schema2 = type("S", (), {
            "slots": [
                {"name": "entity", "required": True},
                {"name": "date", "required": True},
            ],
            "slot_distributions": {},
        })()
        result = svc.complete_frame(schema2, partial_slots={"entity": "Carol"})
        assert result.completeness < 1.0
