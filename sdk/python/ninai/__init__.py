"""
Ninai Python SDK
================

Official Python client for Ninai Enterprise Agentic AI Memory OS.

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

For more information, visit: https://github.com/your-org/ninai
"""

from ninai.client import NinaiClient
from ninai.agents import GoalPlannerAgent, GoalLinkingAgent, MetaAgent
from ninai.tools import ToolInvoker
from ninai.observability import InMemoryEventSink
from ninai.exceptions import (
    NinaiError,
    AuthenticationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
)

__version__ = "0.1.0"
__all__ = [
    "NinaiClient",
    "GoalPlannerAgent",
    "GoalLinkingAgent",
    "MetaAgent",
    "ToolInvoker",
    "InMemoryEventSink",
    "NinaiError",
    "AuthenticationError", 
    "NotFoundError",
    "ValidationError",
    "RateLimitError",
]
