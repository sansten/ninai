"""
Ninai SDK Resources
===================

Resource classes for different API endpoints.
"""

from typing import Optional, List, Dict, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from ninai.client import NinaiClient

from ninai.models import (
    Memory,
    MemoryList,
    SearchResult,
    Organization,
    Team,
    SelfModelBundle,
    ToolSpec,
    ToolInvocationResult,
    LLMCompleteJsonResponse,
)


class MemoriesResource:
    """
    Memories API resource.
    
    Usage:
        # Create a memory
        memory = client.memories.create(
            content="Meeting notes from Q4 planning",
            title="Q4 Planning Meeting",
            tags=["meeting", "planning", "q4"]
        )
        
        # Search memories
        results = client.memories.search("planning meeting")
        
        # Get a specific memory
        memory = client.memories.get("memory-id")
        
        # List memories
        memories = client.memories.list(tags=["meeting"])
        
        # Delete a memory
        client.memories.delete("memory-id")
    """
    
    def __init__(self, client: "NinaiClient"):
        self._client = client
    
    def create(
        self,
        content: str,
        title: Optional[str] = None,
        scope: str = "personal",
        scope_id: Optional[str] = None,
        memory_type: str = "long_term",
        classification: str = "internal",
        tags: Optional[List[str]] = None,
        entities: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        source_type: Optional[str] = None,
        source_id: Optional[str] = None,
    ) -> Memory:
        """
        Create a new memory.
        
        Args:
            content: The content to store (max 100,000 characters)
            title: Optional title for the memory
            scope: Visibility scope (personal, team, department, organization)
            scope_id: ID of the scope entity (required for non-personal scopes)
            memory_type: Type of memory (short_term, long_term, semantic, procedural)
            classification: Security classification (public, internal, confidential, restricted)
            tags: List of searchable tags
            entities: Extracted entities (people, places, etc.)
            metadata: Additional metadata as JSON
            source_type: Source type (manual, agent, integration)
            source_id: Source identifier
            
        Returns:
            Memory: The created memory object
        """
        payload = {
            "content": content,
            "scope": scope,
            "memory_type": memory_type,
            "classification": classification,
            "tags": tags or [],
            "entities": entities or {},
            "extra_metadata": metadata or {},
        }
        
        if title:
            payload["title"] = title
        if scope_id:
            payload["scope_id"] = scope_id
        if source_type:
            payload["source_type"] = source_type
        if source_id:
            payload["source_id"] = source_id
        
        response = self._client._post("/memories", json=payload)
        return Memory(**response)
    
    def get(self, memory_id: str) -> Memory:
        """
        Get a memory by ID.
        
        Args:
            memory_id: The memory's unique identifier
            
        Returns:
            Memory: The memory object
        """
        response = self._client._get(f"/memories/{memory_id}")
        return Memory(**response)
    
    def list(
        self,
        scope: Optional[str] = None,
        tags: Optional[List[str]] = None,
        memory_type: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> MemoryList:
        """
        List memories with optional filters.
        
        Args:
            scope: Filter by scope
            tags: Filter by tags
            memory_type: Filter by memory type
            page: Page number (1-indexed)
            page_size: Number of items per page
            
        Returns:
            MemoryList: Paginated list of memories
        """
        params = {"page": page, "page_size": page_size}
        
        if scope:
            params["scope"] = scope
        if tags:
            params["tags"] = ",".join(tags)
        if memory_type:
            params["memory_type"] = memory_type
        
        response = self._client._get("/memories", params=params)
        return MemoryList(**response)
    
    def search(
        self,
        query: str,
        scope: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        threshold: float = 0.7,
    ) -> SearchResult:
        """
        Search memories using semantic similarity.
        
        Args:
            query: Natural language search query
            scope: Filter by scope
            tags: Filter by tags
            limit: Maximum number of results
            threshold: Minimum similarity score (0-1)
            
        Returns:
            SearchResult: Search results with matching memories
        """
        params = {
            "query": query,
            "limit": limit,
        }
        
        if scope:
            params["scope"] = scope
        if tags:
            params["tags"] = ",".join(tags)
        
        response = self._client._get("/memories/search", params=params)

        # API returns MemorySearchResponse with 'results' field; SDK expects 'items'
        if "items" not in response:
            response = {
                **response,
                "items": response.get("results", []),
            }
        return SearchResult(**response)
    
    def update(
        self,
        memory_id: str,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None,
        classification: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """
        Update a memory.
        
        Args:
            memory_id: The memory's unique identifier
            title: New title
            tags: New tags
            classification: New classification
            metadata: New metadata
            
        Returns:
            Memory: The updated memory object
        """
        payload = {}
        
        if title is not None:
            payload["title"] = title
        if tags is not None:
            payload["tags"] = tags
        if classification is not None:
            payload["classification"] = classification
        if metadata is not None:
            payload["extra_metadata"] = metadata
        
        response = self._client._patch(f"/memories/{memory_id}", json=payload)
        return Memory(**response)
    
    def delete(self, memory_id: str) -> None:
        """
        Delete a memory.
        
        Args:
            memory_id: The memory's unique identifier
        """
        self._client._delete(f"/memories/{memory_id}")

    # ---------------------------------------------------------------------
    # Attachments (multimodal)
    # ---------------------------------------------------------------------

    def list_attachments(self, memory_id: str) -> Dict[str, Any]:
        """List attachments for a memory."""
        return self._client._get(f"/memories/{memory_id}/attachments")

    def upload_attachment(self, memory_id: str, file_path: str) -> Dict[str, Any]:
        """Upload an attachment for a memory."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(file_path)

        headers = self._client._get_headers()
        # Let httpx set multipart boundary content-type.
        headers.pop("Content-Type", None)

        with path.open("rb") as f:
            files = {"file": (path.name, f)}
            resp = self._client._client.post(
                f"/memories/{memory_id}/attachments",
                headers=headers,
                files=files,
            )
        self._client._handle_response_errors(resp)
        return resp.json()

    def download_attachment(self, memory_id: str, attachment_id: str, dest_path: str) -> str:
        """Download an attachment to dest_path; returns dest_path."""
        headers = self._client._get_headers()
        headers.pop("Content-Type", None)

        resp = self._client._client.get(
            f"/memories/{memory_id}/attachments/{attachment_id}",
            headers=headers,
        )
        self._client._handle_response_errors(resp)

        out = Path(dest_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(resp.content)
        return str(out)

    def delete_attachment(self, memory_id: str, attachment_id: str) -> None:
        """Delete an attachment."""
        self._client._delete(f"/memories/{memory_id}/attachments/{attachment_id}")


class OrganizationsResource:
    """Organizations API resource."""
    
    def __init__(self, client: "NinaiClient"):
        self._client = client
    
    def get_current(self) -> Organization:
        """Get current organization."""
        response = self._client._get("/organizations/me")
        return Organization(**response)
    
    def list(self) -> List[Organization]:
        """List organizations the user belongs to."""
        response = self._client._get("/organizations")
        return [Organization(**org) for org in response.get("items", [])]


class TeamsResource:
    """Teams API resource."""
    
    def __init__(self, client: "NinaiClient"):
        self._client = client
    
    def list(self) -> List[Team]:
        """List teams in the current organization."""
        response = self._client._get("/teams")
        if isinstance(response, list):
            items = response
        else:
            items = (response or {}).get("items", [])
        return [Team(**team) for team in items]
    
    def get(self, team_id: str) -> Team:
        """Get a team by ID."""
        response = self._client._get(f"/teams/{team_id}")
        return Team(**response)


class SelfModelResource:
    """SelfModel API resource."""

    def __init__(self, client: "NinaiClient"):
        self._client = client

    def bundle(self) -> SelfModelBundle:
        response = self._client._get("/self-model/bundle")
        return SelfModelBundle(**response)

    def recompute(self) -> Dict[str, Any]:
        return self._client._post("/self-model/recompute")

    def submit_tool_outcome_sample(
        self,
        *,
        tool_name: str,
        success: bool,
        duration_ms: float | None = None,
        session_id: str | None = None,
        memory_id: str | None = None,
        notes: str | None = None,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return self._client._post(
            "/self-model/samples/tool-outcome",
            json={
                "tool_name": tool_name,
                "success": bool(success),
                "duration_ms": duration_ms,
                "session_id": session_id,
                "memory_id": memory_id,
                "notes": notes,
                "extra": extra or {},
            },
        )

    def get_tool_reliability(self, tool_name: str) -> Dict[str, Any]:
        return self._client._get(f"/self-model/reliability/tools/{tool_name}")


class LLMResource:
    """LLM helper endpoints."""

    def __init__(self, client: "NinaiClient"):
        self._client = client

    def complete_json(self, *, prompt: str, schema_hint: Dict[str, Any] | None = None) -> LLMCompleteJsonResponse:
        response = self._client._post(
            "/llm/complete-json",
            json={
                "prompt": prompt,
                "schema_hint": schema_hint or {},
            },
        )
        return LLMCompleteJsonResponse(**response)


class ToolsResource:
    """Tools endpoints."""

    def __init__(self, client: "NinaiClient"):
        self._client = client

    def list(self) -> List[ToolSpec]:
        response = self._client._get("/tools")
        return [ToolSpec(**item) for item in (response or [])]

    def invoke(
        self,
        *,
        tool_name: str,
        tool_input: Dict[str, Any] | None = None,
        session_id: str | None = None,
        iteration_id: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
        classification: str | None = None,
        justification: str | None = None,
        trace_id: str | None = None,
    ) -> ToolInvocationResult:
        headers = self._client._get_headers()
        if trace_id:
            headers = dict(headers)
            headers["X-Trace-ID"] = trace_id

        resp = self._client._client.post(
            "/tools/invoke",
            headers=headers,
            json={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "session_id": session_id,
                "iteration_id": iteration_id,
                "scope": scope,
                "scope_id": scope_id,
                "classification": classification,
                "justification": justification,
            },
        )
        self._client._handle_response_errors(resp)

        payload = resp.json()
        result = payload.get("result") if isinstance(payload, dict) else None
        return ToolInvocationResult(**(result or {}))
