"""Public well-known endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.schemas.cognitive_manifest import CognitiveManifestResponse
from app.services.agent_manifest_service import AgentManifestService

router = APIRouter()

_manifest_service = AgentManifestService()


@router.get(
    "/.well-known/cognitive-manifest.json",
    response_model=CognitiveManifestResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
async def get_cognitive_manifest() -> CognitiveManifestResponse:
    return _manifest_service.build()
