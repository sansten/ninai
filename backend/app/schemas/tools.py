"""Tool listing/invocation schemas (Pydantic v2)."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.schemas.base import BaseSchema
from app.services.cognitive_tooling.tool_invoker import ToolInvocationResult
from app.services.cognitive_tooling.tool_registry import ToolSpec


class ToolSpecOut(BaseSchema):
    name: str
    version: str
    required_permissions: list[str] = Field(default_factory=list)
    allowed_scopes: list[str] | None = None
    require_justification: bool = False
    min_clearance_level: int = 0
    sensitivity: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def from_spec(spec: ToolSpec) -> "ToolSpecOut":
        return ToolSpecOut(
            name=spec.name,
            version=spec.version,
            required_permissions=list(spec.required_permissions or ()),
            allowed_scopes=list(spec.allowed_scopes) if spec.allowed_scopes is not None else None,
            require_justification=bool(spec.require_justification),
            min_clearance_level=int(spec.min_clearance_level or 0),
            sensitivity={
                "allow_persist_input": bool(spec.sensitivity.allow_persist_input),
                "allow_persist_output": bool(spec.sensitivity.allow_persist_output),
                "redacted_input_fields": list(spec.sensitivity.redacted_input_fields or ()),
                "redacted_output_fields": list(spec.sensitivity.redacted_output_fields or ()),
            },
        )


class ToolInvokeRequest(BaseSchema):
    tool_name: str = Field(..., min_length=1, max_length=255)
    tool_input: dict[str, Any] | None = None

    # Required for ToolCallLog FK integrity
    session_id: str | None = None
    iteration_id: str | None = None

    # Optional context
    scope: str | None = None
    scope_id: str | None = None
    classification: str | None = None
    justification: str | None = None


class ToolInvokeResponse(BaseSchema):
    trace_id: str | None = None
    result: ToolInvocationResult
