"""Regression test: upsert_edge's rel_type used to be interpolated raw into
Cypher (`MERGE (a)-[r:{rel_type}]->(b)`), which is an identifier position
_esc() can't protect (it only escapes quoted string literals). Every
current caller passes a hardcoded literal, but nothing stopped a future
caller from passing something dynamic. upsert_edge must now reject any
rel_type that isn't a plain identifier before building the query.
"""
from __future__ import annotations

import pytest

from app.v2.graph.client import V2GraphClient


@pytest.fixture
def client() -> V2GraphClient:
    # A bogus redis_url fails the connect and leaves self._redis = None
    # (fail-open by design), so this never touches a real FalkorDB instance.
    return V2GraphClient(redis_url="redis://nonexistent-host:1/0")


class TestUpsertEdgeRelTypeValidation:
    @pytest.mark.asyncio
    async def test_valid_rel_type_is_accepted(self, client):
        # Doesn't raise — connection is unavailable so it just returns False.
        result = await client.upsert_edge("t1", "a", "b", "FOLLOWED_BY")
        assert result is False

    @pytest.mark.asyncio
    async def test_injection_attempt_is_rejected(self, client):
        malicious = "X]->(b) MATCH (n) DETACH DELETE n //"
        with pytest.raises(ValueError):
            await client.upsert_edge("t1", "a", "b", malicious)

    @pytest.mark.asyncio
    async def test_rel_type_with_space_is_rejected(self, client):
        with pytest.raises(ValueError):
            await client.upsert_edge("t1", "a", "b", "FOO BAR")

    @pytest.mark.asyncio
    async def test_empty_rel_type_is_rejected(self, client):
        with pytest.raises(ValueError):
            await client.upsert_edge("t1", "a", "b", "")

    @pytest.mark.asyncio
    async def test_rel_type_starting_with_digit_is_rejected(self, client):
        with pytest.raises(ValueError):
            await client.upsert_edge("t1", "a", "b", "1FOO")
