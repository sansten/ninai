"""Shared fixtures for GAP-1 (Four-Level Hierarchy) + GAP-6 (kNN Navigation Graph) tests.

Provides:
    - Fake model factories (no DB required)
    - AsyncMock session with configurable execute side effects
    - Mock EmbeddingService and QdrantService
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


# ── IDs ─────────────────────────────────────────────────────────────────
ORG_ID = str(uuid4())
USER_ID = str(uuid4())
TOPIC_ID = str(uuid4())

FAKE_EMBEDDING = [0.1] * 768  # nomic-embed-text dimension
ZERO_EMBEDDING = [0.0] * 768


# ── Lightweight model stubs ─────────────────────────────────────────────
# These avoid importing real SQLAlchemy models and hitting metadata issues.

class FakeMemory:
    """Minimal MemoryMetadata stub."""

    def __init__(self, *, id: Optional[str] = None, content: str = "test",
                 org_id: str = ORG_ID, owner_id: str = USER_ID,
                 scope: str = "personal", created_at: Optional[datetime] = None):
        self.id = id or str(uuid4())
        self.content_preview = content
        self.organization_id = org_id
        self.owner_id = owner_id
        self.scope = scope
        self.scope_id = None
        self.created_at = created_at or datetime.now(timezone.utc)
        self.memory_type = "conversation"


class FakeEpisode:
    """Minimal MemoryEpisode stub."""

    def __init__(self, *, id: Optional[str] = None, org_id: str = ORG_ID,
                 owner_id: str = USER_ID, status: str = "open",
                 message_count: int = 3, vector_id: Optional[str] = None,
                 topic_id: Optional[str] = None, title: Optional[str] = None,
                 narrative_summary: Optional[str] = None,
                 boundary_start: Optional[datetime] = None,
                 boundary_end: Optional[datetime] = None,
                 scope: str = "personal", scope_id: Optional[str] = None,
                 updated_at: Optional[datetime] = None):
        self.id = id or str(uuid4())
        self.organization_id = org_id
        self.owner_id = owner_id
        self.status = status
        self.message_count = message_count
        self.vector_id = vector_id
        self.topic_id = topic_id
        self.title = title
        self.narrative_summary = narrative_summary
        self.boundary_start = boundary_start
        self.boundary_end = boundary_end
        self.scope = scope
        self.scope_id = scope_id
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.boundary_reason = "initial"
        self.boundary_confidence = 1.0


class FakeMembership:
    """Minimal MemoryEpisodeMembership stub."""

    def __init__(self, *, episode_id: str, memory_id: str, position: int = 0):
        self.id = str(uuid4())
        self.organization_id = ORG_ID
        self.episode_id = episode_id
        self.memory_id = memory_id
        self.position = position


class FakeSemanticNode:
    """Minimal MemorySemanticNode stub."""

    def __init__(self, *, id: Optional[str] = None, content: str = "fact",
                 org_id: str = ORG_ID, owner_id: str = USER_ID,
                 vector_id: Optional[str] = None, topic_id: Optional[str] = None,
                 source_episode_ids: Optional[list] = None,
                 composite_quality: float = 0.8):
        self.id = id or str(uuid4())
        self.organization_id = org_id
        self.owner_id = owner_id
        self.content = content
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()
        self.vector_id = vector_id
        self.topic_id = topic_id
        self.source_episode_ids = source_episode_ids or []
        self.source_memory_ids = []
        self.composite_quality = composite_quality
        self.scope = "personal"
        self.scope_id = None
        self.reference_count = 0
        self.status = "active"


class FakeNavigationEdge:
    """Minimal NavigationEdge stub."""

    def __init__(self, *, source_type: str, source_id: str,
                 target_type: str, target_id: str,
                 similarity: float = 0.85, k_rank: int = 1,
                 generation: int = 1, org_id: str = ORG_ID):
        self.id = str(uuid4())
        self.organization_id = org_id
        self.source_type = source_type
        self.source_id = source_id
        self.target_type = target_type
        self.target_id = target_id
        self.similarity = similarity
        self.k_rank = k_rank
        self.generation = generation


class FakeTopic:
    """Minimal MemoryTopic stub."""

    def __init__(self, *, id: Optional[str] = None, label: str = "test topic",
                 org_id: str = ORG_ID, keywords: Optional[list] = None,
                 scope: str = "personal", scope_id: Optional[str] = None):
        self.id = id or str(uuid4())
        self.organization_id = org_id
        self.label = label
        self.keywords = keywords or ["test"]
        self.scope = scope
        self.scope_id = scope_id
        self.scope_key = f"{scope}:{scope_id or ''}"


# ── Scalars result helpers ──────────────────────────────────────────────

class ScalarOneResult:
    """Mimics session.execute().scalar_one_or_none()."""

    def __init__(self, value=None):
        self._value = value

    def scalar_one_or_none(self):
        return self._value

    def scalars(self):
        return self

    def all(self):
        if self._value is None:
            return []
        return [self._value] if not isinstance(self._value, list) else self._value

    def scalar(self):
        return self._value


class ScalarsListResult:
    """Mimics session.execute().scalars().all() returning a list."""

    def __init__(self, items: list):
        self._items = items

    def scalar_one_or_none(self):
        return self._items[0] if self._items else None

    def scalars(self):
        return self

    def all(self):
        return self._items

    def scalar(self):
        return self._items[0] if self._items else None


# ── Session factory ─────────────────────────────────────────────────────

class DeleteResult:
    """Mimics result of session.execute(delete(...))."""

    def __init__(self, rowcount: int = 0):
        self.rowcount = rowcount


@pytest.fixture
def mock_session():
    """An AsyncMock(spec=AsyncSession) with a default flush/commit."""
    session = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    return session


@pytest.fixture
def mock_embed():
    """Patch EmbeddingService.embed to return a deterministic vector."""
    with patch(
        "app.services.embedding_service.EmbeddingService.embed",
        new_callable=AsyncMock,
        return_value=FAKE_EMBEDDING,
    ) as mock:
        yield mock


@pytest.fixture
def mock_qdrant_upsert():
    """Patch QdrantService.upsert_memory to be a no-op."""
    with patch(
        "app.core.qdrant.QdrantService.upsert_memory",
        new_callable=AsyncMock,
        return_value=True,
    ) as mock:
        yield mock


@pytest.fixture
def mock_qdrant_client():
    """Patch QdrantService.get_client to return a MagicMock."""
    client = MagicMock()
    client.search = MagicMock(return_value=[])
    client.retrieve = MagicMock(return_value=[])
    with patch(
        "app.core.qdrant.QdrantService.get_client",
        return_value=client,
    ) as mock:
        yield client


@pytest.fixture
def mock_summarize():
    """Patch summarize_short_term_memories to return a canned summary."""
    with patch(
        "app.services.summarization_service.summarize_short_term_memories",
        new_callable=AsyncMock,
        return_value="Summary of the episode.",
    ) as mock:
        yield mock
