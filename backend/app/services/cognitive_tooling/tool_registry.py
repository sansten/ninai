"""ToolRegistry for Cognitive Loop.

Registry is in-memory and designed for deterministic tool invocation:
- Validate inputs/outputs with Pydantic models when provided
- Track tool metadata: permissions, version, sensitivity

No network calls happen here; handlers implement the tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ValidationError


ToolHandler = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class ToolSensitivity:
    """Controls what can be persisted in tool_call_logs."""

    allow_persist_input: bool = False
    allow_persist_output: bool = False
    redacted_input_fields: tuple[str, ...] = ()
    redacted_output_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolSpec:
    name: str
    version: str = "v1"
    required_permissions: tuple[str, ...] = ()
    input_schema: type[BaseModel] | None = None
    output_schema: type[BaseModel] | None = None
    allowed_scopes: tuple[str, ...] | None = None
    require_justification: bool = False
    min_clearance_level: int = 0
    sensitivity: ToolSensitivity = field(default_factory=ToolSensitivity)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, tuple[ToolSpec, ToolHandler]] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        key = spec.name.strip()
        if not key:
            raise ValueError("Tool name cannot be empty")
        if key in self._tools:
            raise ValueError(f"Tool already registered: {key}")
        self._tools[key] = (spec, handler)

    def get_spec(self, tool_name: str) -> ToolSpec:
        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool: {tool_name}")
        return self._tools[tool_name][0]

    async def invoke(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._tools:
            raise KeyError(f"Unknown tool: {tool_name}")

        spec, handler = self._tools[tool_name]

        validated_input: dict[str, Any]
        if spec.input_schema is not None:
            try:
                validated_input = spec.input_schema.model_validate(tool_input).model_dump()
            except ValidationError as e:
                raise ValueError(f"Invalid input for tool '{tool_name}': {e}")
        else:
            validated_input = dict(tool_input or {})

        output = await handler(validated_input)
        if not isinstance(output, dict):
            raise ValueError(f"Tool '{tool_name}' must return a dict")

        if spec.output_schema is not None:
            try:
                return spec.output_schema.model_validate(output).model_dump()
            except ValidationError as e:
                raise ValueError(f"Invalid output from tool '{tool_name}': {e}")

        return output
