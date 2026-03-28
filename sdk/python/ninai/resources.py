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
    TopicReassignmentRatio,
    TopicRestructureResult,
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
        # Auto-resolve scope_id from the client's organization when not provided
        resolved_scope_id = scope_id or (
            self._client.organization_id if scope != "personal" else None
        )
        if resolved_scope_id:
            payload["scope_id"] = resolved_scope_id
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
        hybrid: bool = False,
    ) -> SearchResult:
        """
        Search memories using semantic similarity.
        
        Args:
            query: Natural language search query
            scope: Filter by scope
            tags: Filter by tags
            limit: Maximum number of results
            threshold: Minimum similarity score (0-1)
            hybrid: Enable hybrid lexical+vector ranking
            
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
        if threshold is not None:
            params["score_threshold"] = threshold
        if hybrid:
            params["hybrid"] = True
        
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


class TopicsResource:
    """Topic maintenance endpoints (admin only)."""

    def __init__(self, client: "NinaiClient"):
        self._client = client

    def reassignment_ratio(
        self,
        *,
        org_id: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> TopicReassignmentRatio:
        params: Dict[str, Any] = {}
        if org_id:
            params["org_id"] = org_id
        if scope:
            params["scope"] = scope
        if scope_id:
            params["scope_id"] = scope_id

        response = self._client._get("/topics/admin/reassignment-ratio", params=params)
        return TopicReassignmentRatio(**response)

    def periodic_restructure(
        self,
        *,
        org_id: str | None = None,
        scope: str | None = None,
        scope_id: str | None = None,
    ) -> TopicRestructureResult:
        payload: Dict[str, Any] = {}
        if org_id:
            payload["org_id"] = org_id
        if scope:
            payload["scope"] = scope
        if scope_id:
            payload["scope_id"] = scope_id

        response = self._client._post("/topics/admin/restructure", json=payload)
        return TopicRestructureResult(**response)


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

class CausalResource:
    """
    Causal Reasoning API resource.
    
    Provides methods for causal discovery, prediction, explanation,
    counterfactual reasoning, and edge validation.
    
    Usage:
        # Discover causal edges from an episode
        edges = client.causal.discover(episode_id="episode-123")
        
        # Predict outcomes
        predictions = client.causal.predict(cause_fact_ids=["fact-1", "fact-2"])
        
        # Explain decisions
        explanation = client.causal.explain(effect_id="fact-purchase")
        
        # Run counterfactuals
        scenario = client.causal.counterfactual(
            scenario="Free shipping on all orders",
            intervention={"free_shipping": True}
        )
        
        # Validate edges
        client.causal.validate(edge_id="edge-123", observed=True)
    """
    
    def __init__(self, client: "NinaiClient"):
        self._client = client
    
    def discover(
        self,
        episode_id: str,
        min_strength: float = 0.3,
        max_latency_seconds: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Discover causal edges from a memory episode.
        
        Args:
            episode_id: Memory episode ID containing facts
            min_strength: Minimum causal strength (0.0-1.0)
            max_latency_seconds: Maximum time between cause and effect
            
        Returns:
            Dict with discovered edges and metadata
        """
        payload = {"min_strength": min_strength}
        if max_latency_seconds:
            payload["max_latency_seconds"] = max_latency_seconds
        
        return self._client._post(f"/causal/discover/{episode_id}", json=payload)
    
    def query_edges(
        self,
        cause_id: Optional[str] = None,
        effect_id: Optional[str] = None,
        min_strength: Optional[float] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> Dict[str, Any]:
        """
        Query causal edges with filters.
        
        Args:
            cause_id: Filter by cause fact ID
            effect_id: Filter by effect fact ID
            min_strength: Minimum causal strength
            page: Page number (1-indexed)
            page_size: Items per page
            
        Returns:
            Dict with matching edges
        """
        params = {"page": page, "page_size": page_size}
        if cause_id:
            params["cause_id"] = cause_id
        if effect_id:
            params["effect_id"] = effect_id
        if min_strength is not None:
            params["min_strength"] = min_strength
        
        return self._client._get("/causal/edges", params=params)
    
    def predict(
        self,
        cause_fact_ids: List[str],
        max_depth: int = 3,
        min_probability: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Predict outcomes from given causes.
        
        Args:
            cause_fact_ids: List of cause fact IDs
            max_depth: Maximum causal path depth
            min_probability: Minimum prediction probability
            
        Returns:
            Dict with predicted outcomes and causal paths
        """
        return self._client._post(
            "/causal/predict",
            json={
                "cause_fact_ids": cause_fact_ids,
                "max_depth": max_depth,
                "min_probability": min_probability,
            },
        )
    
    def explain(
        self,
        effect_id: str,
        max_depth: int = 3,
        min_contribution: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Find root causes of an effect (causal explanation).
        
        Args:
            effect_id: Effect fact ID to explain
            max_depth: Maximum backward trace depth
            min_contribution: Minimum causal contribution (0.0-1.0)
            
        Returns:
            Dict with root causes ranked by contribution
        """
        params = {
            "max_depth": max_depth,
            "min_contribution": min_contribution,
        }
        return self._client._get(f"/causal/explain/{effect_id}", params=params)
    
    def counterfactual(
        self,
        scenario_description: str,
        intervention: Dict[str, Any],
        baseline_episode_id: str,
        outcome_metric: str,
    ) -> Dict[str, Any]:
        """
        Run a counterfactual \"what-if\" scenario.
        
        Args:
            scenario_description: Human-readable scenario description
            intervention: Intervention specification (type, action, parameters)
            baseline_episode_id: Baseline episode for comparison
            outcome_metric: Metric to evaluate (e.g., 'purchase_completion')
            
        Returns:
            Dict with predicted outcome and baseline comparison
        """
        return self._client._post(
            "/causal/counterfactual",
            json={
                "scenario_description": scenario_description,
                "intervention": intervention,
                "baseline_episode_id": baseline_episode_id,
                "outcome_metric": outcome_metric,
            },
        )
    
    def do_calculus(
        self,
        cause_id: str,
        effect_id: str,
        intervention_type: str = "force_true",
    ) -> Dict[str, Any]:
        """
        Estimate causal effect using Pearl's do-calculus.
        
        Args:
            cause_id: Cause fact ID
            effect_id: Effect fact ID
            intervention_type: Type of intervention ('force_true', 'force_false', 'remove')
            
        Returns:
            Dict with observational vs interventional probabilities
        """
        params = {
            "cause_id": cause_id,
            "effect_id": effect_id,
            "intervention_type": intervention_type,
        }
        return self._client._get("/causal/do-calculus", params=params)
    
    def validate(
        self,
        edge_id: str,
        observed: bool,
        observation_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validate and update a causal edge with new observation.
        
        Uses Bayesian inference to update edge strength based on
        whether the effect was observed given the cause.
        
        Args:
            edge_id: Causal edge ID
            observed: Whether the effect was observed
            observation_metadata: Additional context about observation
            
        Returns:
            Dict with updated edge strength and evidence count
        """
        return self._client._post(
            "/causal/validate",
            json={
                "edge_id": edge_id,
                "observed": observed,
                "observation_metadata": observation_metadata or {},
            },
        )


class ConsolidationResource:
    """PR-2 memory consolidation (sleep cycle) API resource."""

    def __init__(self, client: "NinaiClient"):
        self._client = client

    def start(self, session_type: str = "triggered") -> Dict[str, Any]:
        return self._client._post(
            "/v1/consolidation/start",
            json={"session_type": session_type},
        )

    def sessions(self, limit: int = 20) -> Dict[str, Any]:
        return self._client._get(
            "/v1/consolidation/sessions",
            params={"limit": limit},
        )

    def report(self, session_id: str) -> Dict[str, Any]:
        return self._client._get(f"/v1/consolidation/{session_id}/report")

    def memory_arc(self, memory_id: str) -> Dict[str, Any]:
        return self._client._get(f"/v1/memory/{memory_id}/arc")

    def pin(self, memory_id: str) -> Dict[str, Any]:
        return self._client._post(f"/v1/consolidation/pin/{memory_id}")

    def unpin(self, memory_id: str) -> Dict[str, Any]:
        return self._client._post(f"/v1/consolidation/unpin/{memory_id}")


class CompositionResource:
    """PR-7 compositional generalization endpoints."""

    def __init__(self, client: "NinaiClient"):
        self._client = client

    def extract_abstract_procedure(
        self,
        playbook_id: str,
        title: str,
        description: str,
        steps: List[str],
        target_abstraction_level: int = 1,
    ) -> Dict[str, Any]:
        return self._client._post(
            "/v1/composition/abstract-procedure/extract",
            json={
                "playbook_id": playbook_id,
                "title": title,
                "description": description,
                "steps": steps,
                "target_abstraction_level": target_abstraction_level,
            },
        )

    def instantiate_abstract_procedure(
        self,
        abstract_procedure: Dict[str, Any],
        parameters: Dict[str, str],
    ) -> Dict[str, Any]:
        return self._client._post(
            "/v1/composition/abstract-procedure/instantiate",
            json={
                "abstract_procedure": abstract_procedure,
                "parameters": parameters,
            },
        )

    def compose_procedures(
        self,
        abstract_procedures: List[Dict[str, Any]],
        glue_logic: str,
    ) -> Dict[str, Any]:
        return self._client._post(
            "/v1/composition/abstract-procedure/compose",
            json={
                "abstract_procedures": abstract_procedures,
                "glue_logic": glue_logic,
            },
        )

    def find_analogies(self, problem_domain: str, target_domain: str) -> Dict[str, Any]:
        return self._client._get(
            "/v1/composition/analogies",
            params={"problem_domain": problem_domain, "target_domain": target_domain},
        )

    def transfer_solution(
        self,
        source_playbook: Dict[str, Any],
        source_domain: str,
        target_domain: str,
        problem_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._client._post(
            "/v1/composition/transfer-solution",
            json={
                "source_playbook": source_playbook,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "problem_context": problem_context,
            },
        )


class TemporalResource:
    """
    PR-5 temporal reasoning API resource.
    
    Enables time-aware intelligent analysis for optimal decision-making.
    """

    def __init__(self, client: "NinaiClient"):
        self._client = client

    def tag_fact_validity(
        self,
        fact_id: str,
        valid_from: str,
        valid_to: Optional[str] = None,
        confidence_at_time: float = 0.8,
        change_type: str = "stable",
    ) -> Dict[str, Any]:
        """
        Tag a fact with temporal validity interval.
        
        Args:
            fact_id: Identifier for the fact
            valid_from: ISO timestamp when fact became true
            valid_to: ISO timestamp when fact stopped being true (None = ongoing)
            confidence_at_time: Confidence in the fact during this period
            change_type: Type of change (onset, offset, stable, transient)
            
        Returns:
            Dict with created temporal fact details
        """
        return self._client._post(
            "/v1/temporal/facts/tag-validity",
            json={
                "fact_id": fact_id,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "confidence_at_time": confidence_at_time,
                "change_type": change_type,
            },
        )

    def detect_sequences(
        self, entity_timelines: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect recurring event sequences and patterns.
        
        Args:
            entity_timelines: List of timeline dicts with entity_timeline and min_occurrences
            
        Returns:
            List of detected sequences with pattern analysis
        """
        return self._client._post(
            "/v1/temporal/sequences/detect",
            json=entity_timelines,
        )

    def compute_trajectories(
        self, trajectory_requests: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Compute trajectories for quantities over time.
        
        Args:
            trajectory_requests: List of trajectory dicts with entity_id, quantity, measurements
            
        Returns:
            List of trajectory analyses with trends
        """
        return self._client._post(
            "/v1/temporal/trajectories/compute",
            json=trajectory_requests,
        )

    def forecast_trajectory(
        self, trajectory_id: str, horizon_periods: int
    ) -> Dict[str, Any]:
        """
        Forecast future values for a trajectory.
        
        Args:
            trajectory_id: ID of computed trajectory
            horizon_periods: Number of periods to forecast
            
        Returns:
            Dict with predictions and confidence intervals
        """
        return self._client._post(
            "/v1/temporal/trajectories/forecast",
            json={
                "trajectory_id": trajectory_id,
                "horizon_periods": horizon_periods,
            },
        )

    def get_inflection_points(
        self, entity_id: str, sensitivity: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Detect inflection points in a trajectory.
        
        Args:
            entity_id: Entity identifier for the trajectory
            sensitivity: Threshold in std devs for change detection
            
        Returns:
            List of inflection points with timestamps and severity
        """
        return self._client._get(
            f"/v1/temporal/trajectories/{entity_id}/inflection-points",
            params={"sensitivity": sensitivity},
        )

    def temporal_query(
        self, query_type: str, **kwargs
    ) -> Dict[str, Any]:
        """
        Execute temporal query.
        
        Args:
            query_type: Type of temporal query
            **kwargs: Additional query parameters
            
        Returns:
            Query results
        """
        return self._client._get(
            "/v1/temporal/query",
            params={"query_type": query_type, **kwargs},
        )

    def when_should_act(
        self,
        goal_context: Dict[str, Any],
        trajectory: Dict[str, Any],
        action_lead_time_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Determine optimal timing for an action.
        
        Args:
            goal_context: Goal information dict
            trajectory: Trajectory data dict
            action_lead_time_hours: Hours needed to prepare action
            
        Returns:
            Dict with timing recommendation and urgency
        """
        return self._client._post(
            "/v1/temporal/when-should-act",
            json={
                "goal_context": goal_context,
                "trajectory": trajectory,
                "action_lead_time_hours": action_lead_time_hours,
            },
        )


class MetaCognitiveResource:
    """
    PR-6 meta-cognitive planning API resource.

    Provides executive-function endpoints for strategy selection and
    epistemic-state introspection.
    """

    def __init__(self, client: "NinaiClient"):
        self._client = client

    def estimate_complexity(self, query: str) -> Dict[str, Any]:
        """Estimate query complexity on a 0-1 scale."""
        return self._client._get(
            "/v1/meta-cognitive/estimate-complexity",
            params={"query": query},
        )

    def recommend_strategy(
        self,
        query: str,
        time_budget_seconds: int = 30,
        confidence_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Select best cognitive strategy for the given query."""
        return self._client._post(
            "/v1/meta-cognitive/strategy",
            json={
                "query": query,
                "time_budget_seconds": time_budget_seconds,
                "confidence_threshold": confidence_threshold,
            },
        )

    def confidence_calibration(self) -> Dict[str, Any]:
        """Get confidence calibration metrics for the agent."""
        return self._client._get("/v1/meta-cognitive/confidence-calibration")

    def epistemic_state(self) -> Dict[str, Any]:
        """Get known/uncertain/unknown domain snapshot."""
        return self._client._get("/v1/epistemic-state")

