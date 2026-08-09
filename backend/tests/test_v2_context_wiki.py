"""Tests for GET /v2/context/wiki — structured world briefing endpoint."""

from __future__ import annotations

import pytest

from app.v2.api.v2_router import _build_wiki_text
from app.v2.api.schemas import V2WikiEvent, V2WikiPerson, V2WikiTopic


# ---------------------------------------------------------------------------
# _build_wiki_text unit tests (pure, no I/O)
# ---------------------------------------------------------------------------

def _person(name: str, profile: str) -> V2WikiPerson:
    return V2WikiPerson(name=name, profile=profile)


def _event(date: str, subject: str, summary: str) -> V2WikiEvent:
    return V2WikiEvent(date=date, subject=subject, summary=summary)


def _topic(name: str, summary: str, entity_type: str = "concept", weight: float = 1.0) -> V2WikiTopic:
    return V2WikiTopic(name=name, summary=summary, entity_type=entity_type, weight=weight)


def test_build_wiki_text_empty():
    result = _build_wiki_text([], [], [])
    assert result == ""


def test_build_wiki_text_people_section():
    text = _build_wiki_text(
        [_person("Alice", "software engineer, Python expert")],
        [], [],
    )
    assert "## People" in text
    assert "**Alice**" in text
    assert "software engineer" in text


def test_build_wiki_text_events_section():
    text = _build_wiki_text(
        [],
        [_event("2024-03-15", "Alice", "completed onboarding")],
        [],
    )
    assert "## Recent Events" in text
    assert "2024-03-15" in text
    assert "completed onboarding" in text


def test_build_wiki_text_topics_section():
    text = _build_wiki_text(
        [], [],
        [_topic("Project Phoenix", "Q3 migration initiative", entity_type="project")],
    )
    assert "## Key Topics" in text
    assert "Project Phoenix" in text
    assert "Q3 migration initiative" in text


def test_build_wiki_text_all_sections_in_order():
    text = _build_wiki_text(
        [_person("Bob", "manager")],
        [_event("2024-01-10", "Bob", "joined team")],
        [_topic("Budget 2024", "annual budget planning")],
    )
    assert text.index("## People") < text.index("## Recent Events") < text.index("## Key Topics")


def test_build_wiki_text_event_without_date():
    text = _build_wiki_text(
        [],
        [_event("", "Charlie", "presented quarterly results")],
        [],
    )
    assert "Charlie" in text
    assert "presented quarterly results" in text


def test_build_wiki_text_event_without_subject():
    text = _build_wiki_text(
        [],
        [_event("2024-05-20", "", "team offsite event")],
        [],
    )
    assert "2024-05-20" in text
    assert "team offsite event" in text


def test_build_wiki_text_multiple_people():
    text = _build_wiki_text(
        [_person("Alice", "engineer"), _person("Bob", "designer")],
        [], [],
    )
    assert "**Alice**" in text
    assert "**Bob**" in text


def test_build_wiki_text_entity_type_shown_for_topics():
    text = _build_wiki_text(
        [], [],
        [_topic("Salesforce", "CRM tool", entity_type="tool")],
    )
    assert "(tool)" in text


# ---------------------------------------------------------------------------
# Schema field validation
# ---------------------------------------------------------------------------

def test_context_wiki_response_schema_has_wiki_text():
    from app.v2.api.schemas import V2ContextWikiResponse
    resp = V2ContextWikiResponse(
        tenant_id="t1",
        people=[_person("Eve", "data scientist")],
        recent_events=[_event("2024-06-01", "Eve", "submitted model")],
        topics=[_topic("ML Pipeline", "internal training pipeline")],
        wiki_text="## People\n**Eve**: data scientist",
    )
    assert resp.tenant_id == "t1"
    assert resp.wiki_text.startswith("## People")
    assert len(resp.people) == 1
    assert len(resp.recent_events) == 1
    assert len(resp.topics) == 1
