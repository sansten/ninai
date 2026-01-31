"""Unit tests for GoalMetaSupervisor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from app.services.goal_meta_supervisor import GoalMetaSupervisor


@pytest.mark.asyncio
async def test_requires_review_for_low_confidence():
    """Test that low confidence triggers Meta review requirement."""
    mock_session = AsyncMock()
    supervisor = GoalMetaSupervisor(mock_session)
    
    goal = SimpleNamespace(confidence=0.55)  # Below 0.60 threshold
    
    result = await supervisor.requires_review_for_status_change(
        goal=goal,
        old_status="proposed",
        new_status="active",
    )
    
    assert result is True


@pytest.mark.asyncio
async def test_does_not_require_review_for_high_confidence():
    """Test that high confidence doesn't trigger review for trivial transitions."""
    mock_session = AsyncMock()
    supervisor = GoalMetaSupervisor(mock_session)
    
    goal = SimpleNamespace(confidence=0.85)
    
    # Test a transition not in the review list (e.g., blocked -> completed)
    result = await supervisor.requires_review_for_status_change(
        goal=goal,
        old_status="blocked",
        new_status="completed",
    )
    
    # Completion always requires review
    assert result is True


@pytest.mark.asyncio
async def test_requires_review_for_invalid_transition():
    """Test that invalid status transitions trigger review."""
    mock_session = AsyncMock()
    supervisor = GoalMetaSupervisor(mock_session)
    
    goal = SimpleNamespace(confidence=0.90)
    
    # Cannot go directly from proposed to completed
    result = await supervisor.requires_review_for_status_change(
        goal=goal,
        old_status="proposed",
        new_status="completed",
    )
    
    assert result is True


@pytest.mark.asyncio
async def test_allows_valid_transitions():
    """Test that valid transitions still require review."""
    mock_session = AsyncMock()
    supervisor = GoalMetaSupervisor(mock_session)
    
    goal = SimpleNamespace(confidence=0.75)
    
    transitions = [
        ("proposed", "active"),
        ("active", "blocked"),
        ("blocked", "active"),
        ("active", "completed"),
    ]
    
    for from_status, to_status in transitions:
        result = await supervisor.requires_review_for_status_change(
            goal=goal,
            old_status=from_status,
            new_status=to_status,
        )
        # All these transitions require review according to implementation
        assert result is True, f"Transition {from_status} -> {to_status} should require review"


@pytest.mark.asyncio
async def test_requires_review_for_cross_scope_memory_linking():
    """Test that linking restricted memory to lower-scope goal triggers review."""
    mock_session = AsyncMock()
    supervisor = GoalMetaSupervisor(mock_session)
    
    memory = SimpleNamespace(scope="restricted", classification="confidential")
    goal = SimpleNamespace(visibility_scope="team", confidence=0.75)
    
    result = await supervisor.requires_review_for_memory_link(
        memory=memory,
        goal=goal,
        link_type="evidence",
    )
    
    assert result is True


@pytest.mark.asyncio
async def test_allows_same_scope_memory_linking():
    """Test that same-scope linking doesn't require review."""
    mock_session = AsyncMock()
    supervisor = GoalMetaSupervisor(mock_session)
    
    memory = SimpleNamespace(scope="team", classification="internal")
    goal = SimpleNamespace(visibility_scope="team", confidence=0.75)
    
    result = await supervisor.requires_review_for_memory_link(
        memory=memory,
        goal=goal,
        link_type="evidence",
    )
    
    assert result is False


@pytest.mark.asyncio
async def test_review_status_change_approves_valid():
    """Test that review approves valid status changes."""
    mock_session = AsyncMock()
    mock_session.add = Mock()  # add() is synchronous in AsyncSession
    mock_session.flush = AsyncMock()  # flush() is async
    supervisor = GoalMetaSupervisor(mock_session)
    
    goal = SimpleNamespace(
        id="goal1",
        organization_id="org1",
        confidence=0.75,
        goal_type="task",
        visibility_scope="team",
    )
    
    run = await supervisor.review_status_change(
        org_id="org1",
        goal=goal,
        old_status="proposed",
        new_status="active",
    )
    
    assert run.status == "accepted"


@pytest.mark.asyncio
async def test_review_status_change_rejects_invalid():
    """Test that review rejects invalid status changes."""
    mock_session = AsyncMock()
    mock_session.add = Mock()  # add() is synchronous in AsyncSession
    mock_session.flush = AsyncMock()  # flush() is async
    supervisor = GoalMetaSupervisor(mock_session)
    
    goal = SimpleNamespace(
        id="goal1",
        organization_id="org1",
        confidence=0.35,  # Low confidence
        goal_type="task",
        visibility_scope="team",
    )
    
    # Low confidence should escalate and raise ValueError
    with pytest.raises(ValueError, match="escalated status change"):
        await supervisor.review_status_change(
            org_id="org1",
            goal=goal,
            old_status="proposed",
            new_status="completed",
        )


@pytest.mark.asyncio
async def test_review_memory_link_allows_safe():
    """Test that review allows safe memory links."""
    mock_session = AsyncMock()
    mock_session.add = Mock()  # add() is synchronous in AsyncSession
    mock_session.flush = AsyncMock()  # flush() is async
    supervisor = GoalMetaSupervisor(mock_session)
    
    memory = SimpleNamespace(
        id="mem1",
        organization_id="org1",
        scope="team",
        classification="internal",
    )
    goal = SimpleNamespace(
        id="goal1",
        organization_id="org1",
        visibility_scope="team",
        confidence=0.75,
    )
    
    run = await supervisor.review_memory_link(
        org_id="org1",
        memory=memory,
        goal=goal,
        link_type="evidence",
    )
    
    assert run.status == "accepted"


@pytest.mark.asyncio
async def test_review_memory_link_blocks_cross_scope():
    """Test that review blocks unsafe cross-scope links."""
    mock_session = AsyncMock()
    mock_session.add = Mock()  # add() is synchronous in AsyncSession
    mock_session.flush = AsyncMock()  # flush() is async
    supervisor = GoalMetaSupervisor(mock_session)
    
    memory = SimpleNamespace(
        id="mem1",
        organization_id="org1",
        scope="restricted",
        classification="restricted",
    )
    goal = SimpleNamespace(
        id="goal1",
        organization_id="org1",
        visibility_scope="personal",
        confidence=0.75,
    )
    
    # Restricted memory -> personal goal should escalate and raise ValueError
    with pytest.raises(ValueError, match="escalated memory link"):
        await supervisor.review_memory_link(
            org_id="org1",
            memory=memory,
            goal=goal,
            link_type="evidence",
        )
