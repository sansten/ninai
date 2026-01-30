"""
Pydantic Schemas
================

Request and response schemas for API validation.
"""

from app.schemas.base import BaseSchema, PaginatedResponse
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserResponse,
)
from app.schemas.memory import (
    MemoryCreate,
    MemoryUpdate,
    MemoryResponse,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryShareRequest,
    MemorySharingResponse,
    AccessExplanation,
)
from app.schemas.organization import (
    OrganizationCreate,
    OrganizationUpdate,
    OrganizationResponse,
    HierarchyNodeCreate,
    HierarchyNodeResponse,
)
from app.schemas.team import (
    TeamCreate,
    TeamUpdate,
    TeamResponse,
    TeamMemberAdd,
    TeamMemberResponse,
)

from app.schemas.cognitive import (
    PlannerOutput,
    ExecutorOutput,
    CriticOutput,
    EvaluationReportPayload,
    CognitiveSessionCreateRequest,
    CognitiveSessionResponse,
)

from app.schemas.meta_agent import (
    MetaAgentRunOut,
    MetaConflictOut,
    CalibrationProfileOut,
    CalibrationProfileUpdateIn,
)

__all__ = [
    # Base
    "BaseSchema",
    "PaginatedResponse",
    # Auth
    "LoginRequest",
    "LoginResponse",
    "RefreshTokenRequest",
    "TokenResponse",
    "UserResponse",
    # Memory
    "MemoryCreate",
    "MemoryUpdate",
    "MemoryResponse",
    "MemorySearchRequest",
    "MemorySearchResponse",
    "MemoryShareRequest",
    "MemorySharingResponse",
    "AccessExplanation",
    # Organization
    "OrganizationCreate",
    "OrganizationUpdate",
    "OrganizationResponse",
    "HierarchyNodeCreate",
    "HierarchyNodeResponse",
    # Team
    "TeamCreate",
    "TeamUpdate",
    "TeamResponse",
    "TeamMemberAdd",
    "TeamMemberResponse",

    # Cognitive loop
    "PlannerOutput",
    "ExecutorOutput",
    "CriticOutput",
    "EvaluationReportPayload",
    "CognitiveSessionCreateRequest",
    "CognitiveSessionResponse",

    # Meta agent supervision & calibration
    "MetaAgentRunOut",
    "MetaConflictOut",
    "CalibrationProfileOut",
    "CalibrationProfileUpdateIn",
]
