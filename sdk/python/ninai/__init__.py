"""
Ninai Python SDK
================

Official Python client for Ninai Enterprise Agentic AI Cognitive OS.

Quick Start:
    from ninai import NinaiClient
    
    client = NinaiClient(api_key="your-api-key")
    
    # Store a memory
    memory = client.memories.create(
        content="Customer called about billing issue",
        tags=["support", "billing"]
    )
    
    # Search memories
    results = client.memories.search("billing problems")

    # Topic maintenance (org admin only)
    ratio = client.topics.reassignment_ratio()
    print(ratio.reassignment_ratio)

    result = client.topics.periodic_restructure(scope="personal")
    print(result.reassignments, result.guidance_score_after)

For more information, visit: https://github.com/your-org/ninai
"""

from ninai.client import NinaiClient
from ninai.agents import GoalPlannerAgent, GoalLinkingAgent, MetaAgent
from ninai.tools import ToolInvoker
from ninai.observability import InMemoryEventSink
from ninai.exceptions import (
    NinaiError,
    AuthenticationError,
    AuthorizationError,
    EnterpriseFeatureRequired,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
)
from ninai import adapters  # noqa: F401 — makes `ninai.adapters` accessible

__version__ = "0.0.1b1"
__all__ = [
    # Client
    "NinaiClient",
    # Agents
    "GoalPlannerAgent",
    "GoalLinkingAgent",
    "MetaAgent",
    # Tools / observability
    "ToolInvoker",
    "InMemoryEventSink",
    # Exceptions
    "NinaiError",
    "AuthenticationError",
    "AuthorizationError",
    "EnterpriseFeatureRequired",
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
    "ServerError",
    # Adapters sub-package (import from ninai.adapters.langchain / ninai.adapters.adk)
    "adapters",
]
