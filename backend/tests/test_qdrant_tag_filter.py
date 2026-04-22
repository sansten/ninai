from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_search_builds_tag_filter_conditions() -> None:
    """Tags passed to search() become MatchAny conditions in the Qdrant filter."""
    from app.core.qdrant import QdrantService
    from qdrant_client.http.models import FieldCondition, MatchAny

    mock_client = MagicMock()
    mock_client.search.return_value = []

    with patch.object(QdrantService, "get_client", return_value=mock_client), patch.object(
        QdrantService, "ensure_collection", new_callable=AsyncMock
    ):
        await QdrantService.search(
            org_id="org-1",
            query_vector=[0.1, 0.2, 0.3],
            limit=5,
            tags=["conv_001", "run-abc"],
        )

    call_kwargs = mock_client.search.call_args.kwargs
    query_filter = call_kwargs.get("query_filter") or call_kwargs.get("filter")
    assert query_filter is not None

    tag_conditions = [
        condition
        for condition in query_filter.must
        if isinstance(condition, FieldCondition)
        and condition.key == "tags"
        and isinstance(condition.match, MatchAny)
    ]
    assert len(tag_conditions) == 2
    assert [condition.match.any for condition in tag_conditions] == [["conv_001"], ["run-abc"]]


@pytest.mark.asyncio
async def test_search_without_tags_omits_tag_conditions() -> None:
    """No tags should produce no tag filter conditions."""
    from app.core.qdrant import QdrantService
    from qdrant_client.http.models import FieldCondition

    mock_client = MagicMock()
    mock_client.search.return_value = []

    with patch.object(QdrantService, "get_client", return_value=mock_client), patch.object(
        QdrantService, "ensure_collection", new_callable=AsyncMock
    ):
        await QdrantService.search(
            org_id="org-1",
            query_vector=[0.1, 0.2, 0.3],
            limit=5,
        )

    call_kwargs = mock_client.search.call_args.kwargs
    query_filter = call_kwargs.get("query_filter") or call_kwargs.get("filter")
    assert query_filter is not None

    tag_conditions = [
        condition
        for condition in query_filter.must
        if isinstance(condition, FieldCondition) and condition.key == "tags"
    ]
    assert tag_conditions == []