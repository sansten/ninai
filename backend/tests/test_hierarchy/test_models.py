"""Tests for GAP-1 models: MemoryEpisode, MemoryEpisodeMembership, MemorySemanticNode.
Tests for GAP-6 model: NavigationEdge.

Pure unit tests — no DB, no network. Validates model instantiation,
field defaults, and table args.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership
from app.models.memory_semantic_node import MemorySemanticNode
from app.models.navigation_edge import NavigationEdge


# ════════════════════════════════════════════════════════════════════════
# MemoryEpisode
# ════════════════════════════════════════════════════════════════════════

class TestMemoryEpisode:

    def test_tablename(self):
        assert MemoryEpisode.__tablename__ == "memory_episodes"

    def test_default_values(self):
        table = MemoryEpisode.__table__
        assert table.c.scope.default.arg == "personal"
        assert table.c.status.default.arg == "open"
        assert table.c.message_count.default.arg == 0
        assert table.c.boundary_confidence.default.arg == 1.0
        assert table.c.created_by.default.arg == "system"

    def test_scope_values_assignable(self):
        ep = MemoryEpisode(
            id=str(uuid4()),
            organization_id=str(uuid4()),
            owner_id=str(uuid4()),
            scope="team",
        )
        assert ep.scope == "team"

    def test_boundary_metadata(self):
        now = datetime.now(timezone.utc)
        ep = MemoryEpisode(
            id=str(uuid4()),
            organization_id=str(uuid4()),
            owner_id=str(uuid4()),
            boundary_start=now,
            boundary_end=now,
            boundary_reason="topic_shift",
            boundary_confidence=0.87,
        )
        assert ep.boundary_reason == "topic_shift"
        assert ep.boundary_confidence == 0.87
        assert ep.boundary_start == now

    def test_has_composite_indexes(self):
        index_names = {idx.name for idx in MemoryEpisode.__table_args__ if hasattr(idx, "name")}
        assert "ix_memory_episodes_org_owner" in index_names
        assert "ix_memory_episodes_org_topic" in index_names
        assert "ix_memory_episodes_org_status" in index_names
        assert "ix_memory_episodes_org_boundary" in index_names


# ════════════════════════════════════════════════════════════════════════
# MemoryEpisodeMembership
# ════════════════════════════════════════════════════════════════════════

class TestMemoryEpisodeMembership:

    def test_tablename(self):
        assert MemoryEpisodeMembership.__tablename__ == "memory_episode_memberships"

    def test_default_position(self):
        table = MemoryEpisodeMembership.__table__
        assert table.c.position.default.arg == 0
        assert table.c.created_by.default.arg == "system"

    def test_has_unique_index(self):
        index_names = {idx.name for idx in MemoryEpisodeMembership.__table_args__ if hasattr(idx, "name")}
        assert "ux_memory_episode_membership" in index_names


# ════════════════════════════════════════════════════════════════════════
# MemorySemanticNode
# ════════════════════════════════════════════════════════════════════════

class TestMemorySemanticNode:

    def test_tablename(self):
        assert MemorySemanticNode.__tablename__ == "memory_semantic_nodes"

    def test_default_scores(self):
        table = MemorySemanticNode.__table__
        assert table.c.persistence_score.default.arg == 0.0
        assert table.c.specificity_score.default.arg == 0.0
        assert table.c.utility_score.default.arg == 0.0
        assert table.c.independence_score.default.arg == 0.0
        assert table.c.composite_quality.default.arg == 0.0
        assert table.c.status.default.arg == "active"
        assert table.c.created_by.default.arg == "agent"
        assert table.c.reference_count.default.arg == 0

    def test_jsonb_defaults(self):
        table = MemorySemanticNode.__table__
        # Expect list factory defaults for JSON arrays
        assert callable(table.c.source_episode_ids.default.arg)
        assert callable(table.c.source_memory_ids.default.arg)
        assert callable(table.c.entities.default.arg)
        assert callable(table.c.tags.default.arg)

    def test_has_composite_indexes(self):
        index_names = {idx.name for idx in MemorySemanticNode.__table_args__ if hasattr(idx, "name")}
        assert "ix_memory_semantic_nodes_org_owner" in index_names
        assert "ix_memory_semantic_nodes_org_quality" in index_names
        assert "ix_memory_semantic_nodes_content_hash" in index_names


# ════════════════════════════════════════════════════════════════════════
# NavigationEdge
# ════════════════════════════════════════════════════════════════════════

class TestNavigationEdge:

    def test_tablename(self):
        assert NavigationEdge.__tablename__ == "navigation_edges"

    def test_fields(self):
        edge = NavigationEdge(
            id=str(uuid4()),
            organization_id=str(uuid4()),
            source_type="episode",
            source_id=str(uuid4()),
            target_type="semantic_node",
            target_id=str(uuid4()),
            similarity=0.92,
            k_rank=1,
        )
        assert edge.source_type == "episode"
        assert edge.target_type == "semantic_node"
        assert edge.similarity == 0.92
        assert edge.k_rank == 1
        table = NavigationEdge.__table__
        assert table.c.generation.default.arg == 1
        assert table.c.created_by.default.arg == "system"

    def test_has_unique_pair_index(self):
        index_names = {idx.name for idx in NavigationEdge.__table_args__ if hasattr(idx, "name")}
        assert "ux_navigation_edges_pair" in index_names
        assert "ix_navigation_edges_source" in index_names
        assert "ix_navigation_edges_target" in index_names
        assert "ix_navigation_edges_generation" in index_names

    def test_node_types_are_strings(self):
        """source_type/target_type should accept the three hierarchy node types."""
        for ntype in ("episode", "semantic_node", "topic"):
            edge = NavigationEdge(
                id=str(uuid4()),
                organization_id=str(uuid4()),
                source_type=ntype,
                source_id=str(uuid4()),
                target_type=ntype,
                target_id=str(uuid4()),
                similarity=0.5,
                k_rank=1,
            )
            assert edge.source_type == ntype
