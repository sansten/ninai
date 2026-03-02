"""
PR-7: Compositional Generalization schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ExtractAbstractProcedureRequest(BaseModel):
    playbook_id: str
    title: str
    description: str = ""
    steps: List[str] = Field(default_factory=list)
    target_abstraction_level: int = Field(1, ge=0, le=3)


class InstantiateAbstractProcedureRequest(BaseModel):
    abstract_procedure: Dict[str, Any]
    parameters: Dict[str, str] = Field(default_factory=dict)


class ComposeProceduresRequest(BaseModel):
    abstract_procedures: List[Dict[str, Any]]
    glue_logic: str = ""


class FindAnalogiesResponse(BaseModel):
    analogies: List[Dict[str, Any]]


class TransferSolutionRequest(BaseModel):
    source_playbook: Dict[str, Any]
    source_domain: str
    target_domain: str
    problem_context: Dict[str, Any] = Field(default_factory=dict)
