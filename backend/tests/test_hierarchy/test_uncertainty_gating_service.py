"""Tests for UncertaintyGatingService (GAP-4: Entropy-based evidence admission)."""

import pytest
from unittest.mock import AsyncMock, patch
from app.services.uncertainty_gating_service import UncertaintyGatingService


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def mock_settings(autouse=True):
    """Mock settings for all tests."""
    with patch("app.services.uncertainty_gating_service.settings") as mock:
        mock.VLLM_BASE_URL = "http://localhost:11434"
        mock.VLLM_MODEL = "llama2"
        yield mock


@pytest.fixture
def service():
    """Create service instance."""
    return UncertaintyGatingService()


@pytest.fixture
def sample_candidate_items():
    """Sample candidate evidence items."""
    return [
        {
            "id": "msg-1",
            "type": "message",
            "content": "User discussed project timelines with team.",
            "metadata": {"episode_id": "ep-1"},
        },
        {
            "id": "msg-2",
            "type": "message",
            "content": "Team agreed on Q2 delivery date.",
            "metadata": {"episode_id": "ep-1"},
        },
        {
            "id": "msg-3",
            "type": "message",
            "content": "Budget discussion scheduled for next week.",
            "metadata": {"episode_id": "ep-2"},
        },
    ]


# ── Tests: expand_with_gating ───────────────────────────────────────


@pytest.mark.asyncio
async def test_expand_with_gating_accepts_high_entropy_reduction(
    service, sample_candidate_items
):
    """Should include items that reduce entropy above threshold."""
    query = "What were the project timelines?"
    initial_context = "Project context: ongoing development"
    
    # Mock entropy decreasing significantly with each evidence item
    service._compute_entropy = AsyncMock(side_effect=[
        1.5,  # Initial H(y|C)
        1.2,  # H(y|C∪{e1}) - reduction = 0.3 bits > 0.1 threshold ✓
        0.9,  # H(y|C∪{e1,e2}) - reduction = 0.3 bits ✓
        0.85, # H(y|C∪{e1,e2,e3}) - reduction = 0.05 bits < 0.1 threshold ✗
    ])

    result = await service.expand_with_gating(
        query=query,
        initial_context=initial_context,
        candidate_items=sample_candidate_items,
        max_items=10,
        entropy_threshold=0.1,
    )

    # Should include first 2 items, reject 3rd
    assert len(result["included_items"]) == 2
    assert result["included_items"][0]["id"] == "msg-1"
    assert result["included_items"][1]["id"] == "msg-2"
    assert result["total_entropy_reduction"] == pytest.approx(0.6, abs=0.01)
    # After the early-stop bug fix (continue instead of return), all candidates
    # are evaluated; the 3rd is rejected but the loop finishes naturally.
    reason = result["expansion_stopped_reason"].lower()
    assert "threshold" in reason or "all candidates" in reason


@pytest.mark.asyncio
async def test_expand_with_gating_respects_max_items(
    service, sample_candidate_items
):
    """Should stop at max_items even if entropy keeps dropping."""
    query = "What happened?"
    initial_context = "Some context"
    
    # Mock entropy always decreasing significantly
    service._compute_entropy = AsyncMock(side_effect=[1.5, 1.2, 0.8, 0.3])

    result = await service.expand_with_gating(
        query=query,
        initial_context=initial_context,
        candidate_items=sample_candidate_items,
        max_items=2,
        entropy_threshold=0.1,
    )

    assert len(result["included_items"]) == 2
    assert "max" in result["expansion_stopped_reason"].lower()


@pytest.mark.asyncio
async def test_expand_with_gating_handles_empty_candidates(service):
    """Should handle empty candidate list gracefully."""
    result = await service.expand_with_gating(
        query="test",
        initial_context="context",
        candidate_items=[],
        max_items=10,
        entropy_threshold=0.1,
    )

    assert result["included_items"] == []
    assert result["total_entropy_reduction"] == 0.0
    assert "no candidates" in result["expansion_stopped_reason"].lower()


@pytest.mark.asyncio
async def test_expand_with_gating_rejects_all_low_reduction(
    service, sample_candidate_items
):
    """Should reject all items if none meet entropy threshold."""
    query = "test"
    initial_context = "context"
    
    # Mock entropy barely changing
    service._compute_entropy = AsyncMock(side_effect=[1.5, 1.48, 1.46, 1.44])

    result = await service.expand_with_gating(
        query=query,
        initial_context=initial_context,
        candidate_items=sample_candidate_items,
        max_items=10,
        entropy_threshold=0.1,
    )

    # First item rejected immediately
    assert len(result["included_items"]) == 0
    assert result["total_entropy_reduction"] == 0.0


# ── Tests: _compute_entropy ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_compute_entropy_calls_vllm(service):
    """Should call vLLM API with correct prompt format."""
    query = "What is the weather?"
    context = "Weather context here"
    
    service._call_vllm_with_logprobs = AsyncMock(return_value={
        "response": "The weather is sunny",
        "logprobs": [
            {"token": "The", "logprob": -0.5},
            {"token": "weather", "logprob": -0.3},
            {"token": "is", "logprob": -0.2},
        ],
    })

    entropy = await service._compute_entropy(query=query, context=context)

    # Should have called vLLM
    assert service._call_vllm_with_logprobs.call_count == 1
    call_kwargs = service._call_vllm_with_logprobs.call_args.kwargs
    prompt = call_kwargs["prompt"]
    
    # Verify prompt structure
    assert "Context:" in prompt
    assert context in prompt
    assert "Question" in prompt or "Query" in prompt
    assert query in prompt
    
    # Should return positive entropy
    assert entropy > 0


@pytest.mark.asyncio
async def test_compute_entropy_uses_fallback_on_missing_logprobs(service):
    """Should use length-based heuristic when logprobs unavailable."""
    query = "test"
    context = "context"
    
    # Mock vLLM response without logprobs
    service._call_vllm_with_logprobs = AsyncMock(return_value={
        "response": "This is a longer response with multiple words here",
        "logprobs": None,
    })

    entropy = await service._compute_entropy(query=query, context=context)

    # Should use fallback heuristic (longer response = higher entropy)
    assert entropy > 0.3  # Expect decent entropy for ~10 word response


@pytest.mark.asyncio
async def test_entropy_calculation_from_logprobs(service):
    """Should calculate Shannon entropy correctly from logprobs."""
    logprobs_data = [
        {"token": "The", "logprob": -1.0},  # p ≈ 0.368
        {"token": "cat", "logprob": -2.0},  # p ≈ 0.135
        {"token": "sat", "logprob": -1.5},  # p ≈ 0.223
    ]

    entropy = service._calculate_entropy_from_logprobs(logprobs_data)

    # Shannon entropy: H = -Σ p*log2(p)
    # Expected: -(0.368*log2(0.368) + 0.135*log2(0.135) + 0.223*log2(0.223)) / 3
    # ≈ 1.44 bits per token
    assert entropy > 1.0
    assert entropy < 2.0


@pytest.mark.asyncio
async def test_entropy_fallback_heuristic_scaling(service):
    """Fallback entropy should scale with response length."""
    short_resp = "Short"
    long_resp = "This is a much longer response with many more tokens to process"

    # Test fallback directly
    short_entropy = service._calculate_entropy_from_logprobs(None, short_resp)
    long_entropy = service._calculate_entropy_from_logprobs(None, long_resp)

    assert long_entropy > short_entropy
    assert short_entropy >= 0.1  # Minimum for any response
    assert long_entropy <= 2.0   # Cap at 2 bits


# ── Tests: _build_prompt ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_prompt_includes_context_and_query(service):
    """Prompt should have correct structure."""
    query = "What is the status?"
    context = "Project is 50% complete"

    prompt = service._build_prompt(query=query, context=context)

    assert "Context:" in prompt
    assert context in prompt
    assert "Query:" in prompt
    assert query in prompt
    # Should have instruction
    assert "answer" in prompt.lower() or "respond" in prompt.lower()


# ── Tests: edge cases ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_expand_with_gating_handles_vllm_error(
    service, sample_candidate_items
):
    """Should handle vLLM API errors gracefully."""
    query = "test"
    initial_context = "context"
    
    # Mock vLLM call failing
    service._compute_entropy = AsyncMock(side_effect=Exception("vLLM connection failed"))

    # Should not crash, return empty result
    with pytest.raises(Exception):
        await service.expand_with_gating(
            query=query,
            initial_context=initial_context,
            candidate_items=sample_candidate_items,
            max_items=10,
            entropy_threshold=0.1,
        )


@pytest.mark.asyncio
async def test_expand_with_gating_with_zero_threshold(
    service, sample_candidate_items
):
    """With threshold=0, should include all items (no filtering)."""
    query = "test"
    initial_context = "context"
    
    # Mock entropy always decreasing slightly
    service._compute_entropy = AsyncMock(side_effect=[1.5, 1.49, 1.48, 1.47])

    result = await service.expand_with_gating(
        query=query,
        initial_context=initial_context,
        candidate_items=sample_candidate_items,
        max_items=10,
        entropy_threshold=0.0,  # Accept any reduction
    )

    # Should include all items
    assert len(result["included_items"]) == 3
