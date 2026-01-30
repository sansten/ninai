"""
Pydantic schemas for Memory Syscall API and Capability Management
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import uuid
from datetime import datetime


# ============================================================================
# CAPABILITY TOKEN SCHEMAS
# ============================================================================

class CapabilityTokenCreateRequest(BaseModel):
    """Request to issue a new capability token."""
    name: str = Field(..., description="Token name, e.g., 'agent_search_token'")
    scopes: List[str] = Field(..., description="List of scopes: read, append, search, upsert, consolidate, promote")
    session_id: Optional[uuid.UUID] = Field(None, description="Tie to agent session")
    agent_name: Optional[str] = Field(None, description="Agent this token is for")
    issued_to_user_id: Optional[uuid.UUID] = Field(None, description="User this token is issued to")
    ttl_seconds: Optional[int] = Field(86400, description="Time to live in seconds (default 24h)")
    max_tokens_per_month: Optional[int] = Field(None, description="Monthly token quota")
    max_storage_bytes: Optional[int] = Field(None, description="Storage quota in bytes")
    max_requests_per_minute: Optional[int] = Field(None, description="Rate limit requests/minute")


class CapabilityTokenResponse(BaseModel):
    """Response with capability token details."""
    id: str
    token: str  # The actual Bearer token value
    name: str
    scopes: str
    agent_name: Optional[str]
    expires_at: str
    created_at: str
    tokens_used: int
    storage_used_bytes: int
    revoked_at: Optional[str] = None


# ============================================================================
# MEMORY SYSCALL SCHEMAS
# ============================================================================

class MemoryReadRequest(BaseModel):
    """Request to read a knowledge item."""
    knowledge_id: uuid.UUID = Field(..., description="Knowledge item ID")


class MemoryAppendRequest(BaseModel):
    """Request to append new knowledge."""
    content: str = Field(..., description="Knowledge content")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MemorySearchRequest(BaseModel):
    """Request to search knowledge via vector similarity."""
    embedding: List[float] = Field(..., description="Query vector")
    limit: Optional[int] = Field(10, description="Max results")


class MemoryUpsertRequest(BaseModel):
    """Request to update or insert knowledge."""
    knowledge_id: uuid.UUID = Field(..., description="Knowledge ID (create if doesn't exist)")
    content: Optional[str] = Field(None, description="Updated content")
    embedding: Optional[List[float]] = Field(None, description="Updated embedding")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class MemoryConsolidateRequest(BaseModel):
    """Request to consolidate multiple knowledge items."""
    knowledge_ids: List[uuid.UUID] = Field(..., description="Source knowledge IDs")
    merged_content: str = Field(..., description="Merged content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata for merged item")


class MemorySyscallResponse(BaseModel):
    """Generic memory syscall response."""
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
