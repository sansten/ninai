"""
PR-7: Compositional Generalization Engine endpoints.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Query

from app.schemas.compositional_generalization_pr7 import (
    ComposeProceduresRequest,
    ExtractAbstractProcedureRequest,
    FindAnalogiesResponse,
    InstantiateAbstractProcedureRequest,
    TransferSolutionRequest,
)
from app.services.procedural_abstraction_service import (
    AnalogyService,
    ProceduralAbstractionService,
)


router = APIRouter(prefix="/v1/composition", tags=["Compositional Generalization PR-7"])


@router.post("/abstract-procedure/extract", response_model=Dict[str, Any])
async def extract_abstract_procedure(request: ExtractAbstractProcedureRequest) -> Dict[str, Any]:
    svc = ProceduralAbstractionService()
    return await svc.extract_abstract_procedure(
        playbook_id=request.playbook_id,
        title=request.title,
        description=request.description,
        steps=request.steps,
        target_abstraction_level=request.target_abstraction_level,
    )


@router.post("/abstract-procedure/instantiate", response_model=Dict[str, Any])
async def instantiate_abstract_procedure(request: InstantiateAbstractProcedureRequest) -> Dict[str, Any]:
    svc = ProceduralAbstractionService()
    return await svc.instantiate_abstract(
        abstract_procedure=request.abstract_procedure,
        parameters=request.parameters,
    )


@router.post("/abstract-procedure/compose", response_model=Dict[str, Any])
async def compose_abstract_procedures(request: ComposeProceduresRequest) -> Dict[str, Any]:
    svc = ProceduralAbstractionService()
    return await svc.compose_procedures(
        abstract_procedures=request.abstract_procedures,
        glue_logic=request.glue_logic,
    )


@router.get("/analogies", response_model=FindAnalogiesResponse)
async def find_analogies(
    problem_domain: str = Query(...),
    target_domain: str = Query(...),
) -> FindAnalogiesResponse:
    svc = AnalogyService()
    analogies = await svc.find_analogies(problem_domain=problem_domain, target_domain=target_domain)
    return FindAnalogiesResponse(analogies=analogies)


@router.post("/transfer-solution", response_model=Dict[str, Any])
async def transfer_solution(request: TransferSolutionRequest) -> Dict[str, Any]:
    svc = AnalogyService()
    return await svc.transfer_solution(
        source_playbook=request.source_playbook,
        source_domain=request.source_domain,
        target_domain=request.target_domain,
        problem_context=request.problem_context,
    )
