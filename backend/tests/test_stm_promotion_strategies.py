"""Tests for short-term memory promotion strategies (count, spacing, hybrid)."""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.short_term_memory import ShortTermMemoryService


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.set = AsyncMock()
    client.incr = AsyncMock()
    client.rpush = AsyncMock()
    client.lrange = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_count_strategy_ignores_spacing(mock_redis_client):
    """Test count strategy promotes after N accesses regardless of timing."""
    service = ShortTermMemoryService("user-001", "org-001")
    service.PROMOTION_STRATEGY = "count"
    service.ACCESS_COUNT_THRESHOLD = 3
    
    # 3 accesses in 1 second (same-day burst)
    result = await service._check_promotion_eligibility(
        "mem-001", access_count=3, client=mock_redis_client
    )
    
    assert result is True  # Should promote despite no spacing


@pytest.mark.asyncio
async def test_spacing_strategy_requires_temporal_gaps(mock_redis_client):
    """Test spacing strategy requires temporal separation between accesses."""
    service = ShortTermMemoryService("user-001", "org-001")
    service.PROMOTION_STRATEGY = "spacing"
    service.ACCESS_COUNT_THRESHOLD = 3
    service.MIN_ACCESS_SPACING_HOURS = 24.0
    
    now = datetime.now(timezone.utc).timestamp()
    day_1 = now - (2 * 86400)  # 2 days ago
    day_2 = now - (1 * 86400)  # 1 day ago
    day_3 = now  # today
    
    # Properly spaced: 3 accesses, 24h apart
    mock_redis_client.lrange.return_value = [
        str(day_1).encode(), 
        str(day_2).encode(), 
        str(day_3).encode()
    ]
    
    result = await service._check_promotion_eligibility(
        "mem-001", access_count=3, client=mock_redis_client
    )
    
    assert result is True  # Should promote with proper spacing


@pytest.mark.asyncio
async def test_spacing_strategy_rejects_same_day_burst(mock_redis_client):
    """Test spacing strategy rejects accesses too close together."""
    service = ShortTermMemoryService("user-001", "org-001")
    service.PROMOTION_STRATEGY = "spacing"
    service.ACCESS_COUNT_THRESHOLD = 3
    service.MIN_ACCESS_SPACING_HOURS = 24.0
    
    now = datetime.now(timezone.utc).timestamp()
    
    # Same-day burst: 3 accesses within 1 hour
    mock_redis_client.lrange.return_value = [
        str(now - 3600).encode(),  # 1 hour ago
        str(now - 1800).encode(),  # 30 min ago
        str(now).encode(),         # now
    ]
    
    result = await service._check_promotion_eligibility(
        "mem-001", access_count=3, client=mock_redis_client
    )
    
    assert result is False  # Should reject due to insufficient spacing


@pytest.mark.asyncio
async def test_hybrid_strategy_requires_both(mock_redis_client):
    """Test hybrid strategy requires BOTH count AND spacing."""
    service = ShortTermMemoryService("user-001", "org-001")
    service.PROMOTION_STRATEGY = "hybrid"
    service.ACCESS_COUNT_THRESHOLD = 3
    service.MIN_ACCESS_SPACING_HOURS = 24.0
    
    now = datetime.now(timezone.utc).timestamp()
    
    # Test 1: Sufficient spacing but low count (2 accesses)
    mock_redis_client.lrange.return_value = [
        str(now - 86400).encode(),  # 1 day ago
        str(now).encode(),          # now
    ]
    
    result = await service._check_promotion_eligibility(
        "mem-001", access_count=2, client=mock_redis_client
    )
    assert result is False  # Fails count requirement
    
    # Test 2: Sufficient count but poor spacing (3 accesses in 1 hour)
    mock_redis_client.lrange.return_value = [
        str(now - 3600).encode(),  # 1 hour ago
        str(now - 1800).encode(),  # 30 min ago
        str(now).encode(),         # now
    ]
    
    result = await service._check_promotion_eligibility(
        "mem-001", access_count=3, client=mock_redis_client
    )
    assert result is False  # Fails spacing requirement
    
    # Test 3: Both count AND spacing satisfied
    mock_redis_client.lrange.return_value = [
        str(now - (2 * 86400)).encode(),  # 2 days ago
        str(now - (1 * 86400)).encode(),  # 1 day ago
        str(now).encode(),                 # now
    ]
    
    result = await service._check_promotion_eligibility(
        "mem-001", access_count=3, client=mock_redis_client
    )
    assert result is True  # Passes both requirements


@pytest.mark.asyncio
async def test_spacing_check_insufficient_accesses(mock_redis_client):
    """Test spacing check with too few accesses."""
    service = ShortTermMemoryService("user-001", "org-001")
    service.ACCESS_COUNT_THRESHOLD = 3
    
    # Only 1 access
    mock_redis_client.lrange.return_value = [
        str(datetime.now(timezone.utc).timestamp()).encode()
    ]
    
    result = await service._has_sufficient_spacing("mem-001", mock_redis_client)
    assert result is False


@pytest.mark.asyncio
async def test_spacing_check_no_timestamps(mock_redis_client):
    """Test spacing check with no access history."""
    service = ShortTermMemoryService("user-001", "org-001")
    
    mock_redis_client.lrange.return_value = []
    
    result = await service._has_sufficient_spacing("mem-001", mock_redis_client)
    assert result is False
