"""OpenAI-compatible tool schema export endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status

router = APIRouter()


@router.get(
    "/openai/tools",
    status_code=status.HTTP_200_OK,
    tags=["OpenAI Tool Schema"],
    operation_id="exportOpenAIToolSchema",
    summary="Export OpenAI-compatible tool schema for Ninai cognitive operations",
)
async def export_openai_tool_schema() -> dict:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "cognitive_decide",
                "description": "Run a decision cycle against current cognitive context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "context": {"type": "object"},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cognitive_plan",
                "description": "Generate a bounded cognitive plan for a goal",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "goal": {"type": "string"},
                        "depth": {"type": "integer", "minimum": 1, "maximum": 10},
                    },
                    "required": ["goal"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "cognitive_read",
                "description": "Retrieve relevant cognitive memory context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    return {
        "format": "openai-tools-v1",
        "tools": tools,
    }
