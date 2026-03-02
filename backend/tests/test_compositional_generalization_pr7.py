"""
PR-7 Compositional Generalization Engine tests.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AbstractProcedure, Analogy, AnalogyApplicability
from app.services.procedural_abstraction_service import (
    AnalogyService,
    ProceduralAbstractionService,
)


@pytest.fixture
async def test_org_id() -> str:
    return str(uuid4())


@pytest.mark.asyncio
async def test_abstract_procedure_model_creation(db_session: AsyncSession, test_org_id: str):
    model = AbstractProcedure(
        organization_id=test_org_id,
        concrete_playbook_id=str(uuid4()),
        abstraction_level=2,
        title="Deploy [artifact] to [platform]",
        description="Generalized deployment flow",
        parameters={"artifact": "string", "platform": "string"},
        prerequisites=["access available"],
        postconditions=["deployment completed"],
        invariants=["rollback path exists"],
        instances=[],
    )
    db_session.add(model)
    await db_session.commit()

    loaded = await db_session.get(AbstractProcedure, model.id)
    assert loaded is not None
    assert loaded.abstraction_level == 2


@pytest.mark.asyncio
async def test_analogy_model_creation(db_session: AsyncSession, test_org_id: str):
    model = Analogy(
        organization_id=test_org_id,
        source_domain="deployment",
        target_domain="customer_onboarding",
        structural_similarity=0.84,
        mapped_concepts={"artifact": "onboarding_package"},
        constraints=["domain compliance"],
        applicability=AnalogyApplicability.PARTIAL.value,
        discovered_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
    )
    db_session.add(model)
    await db_session.commit()

    loaded = await db_session.get(Analogy, model.id)
    assert loaded is not None
    assert loaded.source_domain == "deployment"


@pytest.mark.asyncio
async def test_extract_abstract_procedure_level_0():
    svc = ProceduralAbstractionService()
    result = await svc.extract_abstract_procedure(
        playbook_id="pb-1",
        title="Deploy app to AWS EKS",
        description="Deploy docker image to cluster",
        steps=["Build docker image", "Deploy to EKS cluster"],
        target_abstraction_level=0,
    )
    assert result["abstraction_level"] == 0
    assert "AWS" in result["title"]


@pytest.mark.asyncio
async def test_extract_abstract_procedure_level_1_generalizes_tokens():
    svc = ProceduralAbstractionService()
    result = await svc.extract_abstract_procedure(
        playbook_id="pb-2",
        title="Deploy app to AWS EKS",
        description="Deploy docker image to cluster",
        steps=["Build docker image", "Deploy to EKS cluster"],
        target_abstraction_level=1,
    )
    assert result["abstraction_level"] == 1
    assert "[platform]" in result["title"] or "[infrastructure]" in result["title"]


@pytest.mark.asyncio
async def test_extract_abstract_procedure_level_3_principle_mode():
    svc = ProceduralAbstractionService()
    result = await svc.extract_abstract_procedure(
        playbook_id="pb-3",
        title="Any title",
        description="Any desc",
        steps=["one", "two"],
        target_abstraction_level=3,
    )
    assert result["abstraction_level"] == 3
    assert len(result["steps"]) >= 4


@pytest.mark.asyncio
async def test_instantiate_abstract_replaces_parameters():
    svc = ProceduralAbstractionService()
    abstract_proc = {
        "id": "ap-1",
        "title": "Deploy [artifact] to [platform]",
        "description": "Ship [artifact]",
        "steps": ["Build [artifact]", "Deploy [artifact] to [platform]"],
    }
    result = await svc.instantiate_abstract(
        abstract_procedure=abstract_proc,
        parameters={"artifact": "docker_image", "platform": "on_prem_cluster"},
    )
    assert "docker_image" in result["title"]
    assert any("on_prem_cluster" in s for s in result["steps"])


@pytest.mark.asyncio
async def test_compose_procedures_merges_steps_and_glue():
    svc = ProceduralAbstractionService()
    result = await svc.compose_procedures(
        abstract_procedures=[
            {"id": "a", "steps": ["Deploy to platform"]},
            {"id": "b", "steps": ["Monitor health"]},
        ],
        glue_logic="if unhealthy then rollback",
    )
    assert len(result["steps"]) >= 3
    assert "glue" in result["steps"][-1].lower()


@pytest.mark.asyncio
async def test_find_analogies_returns_candidates():
    svc = AnalogyService()
    result = await svc.find_analogies("deployment", "customer_onboarding")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert result[0]["source_domain"] == "deployment"


@pytest.mark.asyncio
async def test_find_analogies_default_mapping_path():
    svc = AnalogyService()
    result = await svc.find_analogies("unknown_source", "unknown_target")
    assert isinstance(result, list)
    assert len(result) == 1
    assert "mapped_concepts" in result[0]


@pytest.mark.asyncio
async def test_transfer_solution_maps_steps_and_context():
    svc = AnalogyService()
    result = await svc.transfer_solution(
        source_playbook={"title": "Deployment Runbook", "steps": ["deploy artifact", "run health_check"]},
        source_domain="deployment",
        target_domain="customer_onboarding",
        problem_context={"tenant": "enterprise"},
    )
    assert result["source_domain"] == "deployment"
    assert result["target_domain"] == "customer_onboarding"
    assert len(result["transferred_steps"]) >= 2


@pytest.mark.asyncio
async def test_transfer_solution_confidence_in_range():
    svc = AnalogyService()
    result = await svc.transfer_solution(
        source_playbook={"title": "Runbook", "steps": ["prepare", "execute", "validate"]},
        source_domain="deployment",
        target_domain="incident_response",
        problem_context={},
    )
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_analogy_applicability_enum_values():
    assert AnalogyApplicability.FULL.value == "full"
    assert AnalogyApplicability.PARTIAL.value == "partial"
    assert AnalogyApplicability.METAPHORICAL.value == "metaphorical"


@pytest.mark.asyncio
async def test_persist_multiple_abstract_procedures(db_session: AsyncSession, test_org_id: str):
    first = AbstractProcedure(
        organization_id=test_org_id,
        concrete_playbook_id=str(uuid4()),
        abstraction_level=1,
        title="A",
        description="A",
        parameters={},
        prerequisites=[],
        postconditions=[],
        invariants=[],
        instances=[],
    )
    second = AbstractProcedure(
        organization_id=test_org_id,
        concrete_playbook_id=str(uuid4()),
        abstraction_level=2,
        title="B",
        description="B",
        parameters={},
        prerequisites=[],
        postconditions=[],
        invariants=[],
        instances=[],
    )
    db_session.add(first)
    db_session.add(second)
    await db_session.commit()

    rows = (await db_session.execute(
        select(AbstractProcedure).where(AbstractProcedure.organization_id == test_org_id)
    )).scalars().all()
    assert len(rows) >= 2


@pytest.mark.asyncio
async def test_persist_multiple_analogies(db_session: AsyncSession, test_org_id: str):
    a1 = Analogy(
        organization_id=test_org_id,
        source_domain="deployment",
        target_domain="onboarding",
        structural_similarity=0.7,
        mapped_concepts={},
        constraints=[],
        applicability="partial",
        discovered_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
    )
    a2 = Analogy(
        organization_id=test_org_id,
        source_domain="deployment",
        target_domain="incident_response",
        structural_similarity=0.8,
        mapped_concepts={},
        constraints=[],
        applicability="full",
        discovered_at=datetime.utcnow(),
        created_at=datetime.utcnow(),
    )
    db_session.add(a1)
    db_session.add(a2)
    await db_session.commit()

    rows = (await db_session.execute(
        select(Analogy).where(Analogy.organization_id == test_org_id)
    )).scalars().all()
    assert len(rows) >= 2
