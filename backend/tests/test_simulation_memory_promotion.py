"""Unit tests for SimulationService memory promotion feature."""

from __future__ import annotations

from app.services.simulation_service import SimulationService


def test_simulate_memory_promotion_recommends_for_high_access():
    """Test that simulation recommends promotion for frequently accessed memories."""
    service = SimulationService()
    
    result = service.simulate_memory_promotion(
        memory_content="Important customer order for Q1 2026",
        access_count=5,
        importance_score=0.75,
        memory_scope="team",
        tags=["order", "customer"],
    )
    
    assert result["should_promote"] is True
    assert result["confidence"] >= 0.5
    assert len(result["risk_factors"]) == 0


def test_simulate_memory_promotion_warns_for_sensitive_content():
    """Test that simulation detects sensitive keywords and adds risk factors."""
    service = SimulationService()
    
    result = service.simulate_memory_promotion(
        memory_content="The API secret key is abc123xyz",
        access_count=2,
        importance_score=0.60,
        memory_scope="personal",
        tags=[],
    )
    
    assert len(result["risk_factors"]) > 0
    assert any("sensitive" in r.lower() for r in result["risk_factors"])
    assert result["metadata"]["risk_score"] > 0


def test_simulate_memory_promotion_rejects_short_content():
    """Test that simulation flags very short content as low value."""
    service = SimulationService()
    
    result = service.simulate_memory_promotion(
        memory_content="OK",  # Very short
        access_count=1,
        importance_score=0.40,
        memory_scope="personal",
        tags=[],
    )
    
    assert result["should_promote"] is False
    assert any("short" in r.lower() for r in result["risk_factors"])


def test_simulate_memory_promotion_considers_restricted_scope():
    """Test that simulation adds risk for restricted scope memories."""
    service = SimulationService()
    
    result = service.simulate_memory_promotion(
        memory_content="Confidential project details for client X",
        access_count=3,
        importance_score=0.70,
        memory_scope="restricted",
        tags=["confidential"],
    )
    
    assert len(result["risk_factors"]) > 0
    assert any("scope" in r.lower() for r in result["risk_factors"])


def test_simulate_memory_promotion_metadata_includes_scores():
    """Test that simulation report includes all score components."""
    service = SimulationService()
    
    result = service.simulate_memory_promotion(
        memory_content="Regular team update meeting notes",
        access_count=3,
        importance_score=0.65,
        memory_scope="team",
        tags=["meeting"],
    )
    
    assert "access_count" in result["metadata"]
    assert "importance_score" in result["metadata"]
    assert "combined_score" in result["metadata"]
    assert "risk_score" in result["metadata"]
    assert "adjusted_score" in result["metadata"]
    assert "content_length" in result["metadata"]
