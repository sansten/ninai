"""
PR-6: Meta-Cognitive Planning API Endpoints
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.meta_cognitive_pr6 import (
    ComplexityResponse,
    StrategyRequest,
    StrategyResponse,
    ConfidenceCalibrationResponse,
    EpistemicStateResponse,
)
from app.services.meta_cognitive_service import MetaCognitiveService


router = APIRouter()


async def get_meta_cognitive_service(
    session: AsyncSession = Depends(get_db),
) -> MetaCognitiveService:
    return MetaCognitiveService(session=session)


@router.get("/v1/meta-cognitive/estimate-complexity", response_model=ComplexityResponse)
async def estimate_complexity(
    query: str = Query(..., description="Query text to estimate complexity for"),
    service: MetaCognitiveService = Depends(get_meta_cognitive_service),
) -> ComplexityResponse:
    complexity = await service.estimate_query_complexity(query)
    return ComplexityResponse(query=query, complexity_score=complexity)


@router.post("/v1/meta-cognitive/strategy", response_model=StrategyResponse)
async def select_strategy(
    request: StrategyRequest,
    service: MetaCognitiveService = Depends(get_meta_cognitive_service),
) -> StrategyResponse:
    complexity = await service.estimate_query_complexity(request.query)
    result = await service.select_strategy(
        query=request.query,
        complexity=complexity,
        time_budget_seconds=request.time_budget_seconds,
        confidence_threshold=request.confidence_threshold,
    )
    return StrategyResponse(**result)


@router.get(
    "/v1/meta-cognitive/confidence-calibration",
    response_model=ConfidenceCalibrationResponse,
)
async def confidence_calibration(
    service: MetaCognitiveService = Depends(get_meta_cognitive_service),
) -> ConfidenceCalibrationResponse:
    result = await service.get_confidence_calibration()
    return ConfidenceCalibrationResponse(**result)


@router.get("/v1/epistemic-state", response_model=EpistemicStateResponse)
async def get_epistemic_state(
    service: MetaCognitiveService = Depends(get_meta_cognitive_service),
) -> EpistemicStateResponse:
    result = await service.get_epistemic_state()
    return EpistemicStateResponse(**result)
