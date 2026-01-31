"""
Ninai SDK Client
================

Main client class for interacting with the Ninai API.
"""

import os
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import httpx

from ninai.exceptions import (
    NinaiError,
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    ValidationError,
    RateLimitError,
    ServerError,
)
from ninai.resources import (
    MemoriesResource,
    OrganizationsResource,
    TeamsResource,
    SelfModelResource,
    ToolsResource,
    LLMResource,
)
from ninai.agents import GoalPlannerAgent, GoalLinkingAgent, MetaAgent
from ninai.models import AuthTokens, User


class NinaiClient:
    """
    Ninai API Client with JWT authentication.
    
    There are three ways to authenticate:
    
    1. API Key (recommended for production):
        client = NinaiClient(api_key="your-api-key")
    
    2. Email and Password:
        client = NinaiClient()
        client.login(email="user@example.com", password="password")
    
    3. Access Token (if you already have one):
        client = NinaiClient(access_token="your-jwt-token")
    
    Example:
        from ninai import NinaiClient
        
        # Initialize with API key
        client = NinaiClient(api_key="nai_abc123...")
        
        # Create a memory
        memory = client.memories.create(
            content="Customer requested refund for order #12345",
            tags=["support", "refund", "order"]
        )
        
        # Search memories
        results = client.memories.search("refund request")
        for memory in results.items:
            print(f"- {memory.title}: {memory.content_preview}")
    """
    
    DEFAULT_BASE_URL = "http://localhost:8000/api/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        access_token: Optional[str] = None,
        base_url: Optional[str] = None,
        organization_id: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the Ninai client.
        
        Args:
            api_key: Ninai API key (starts with 'nai_')
            access_token: JWT access token (if already authenticated)
            base_url: API base URL (default: http://localhost:8000/api/v1)
            organization_id: Organization ID for multi-tenant requests
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or os.getenv("NINAI_API_URL", self.DEFAULT_BASE_URL)
        self.api_key = api_key or os.getenv("NINAI_API_KEY")
        self.organization_id = organization_id or os.getenv("NINAI_ORGANIZATION_ID")
        
        self._access_token = access_token
        self._refresh_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        self._current_user: Optional[User] = None
        
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
        )
        
        # Initialize resource accessors
        self.memories = MemoriesResource(self)
        self.organizations = OrganizationsResource(self)
        self.teams = TeamsResource(self)
        self.self_model = SelfModelResource(self)
        self.tools = ToolsResource(self)
        self.llm = LLMResource(self)

        # Typed agent helpers
        self._goal_planner_agent: GoalPlannerAgent | None = None
        self._goal_linking_agent: GoalLinkingAgent | None = None
        self._meta_agent: MetaAgent | None = None

    def goal_planner_agent(self) -> GoalPlannerAgent:
        if self._goal_planner_agent is None:
            self._goal_planner_agent = GoalPlannerAgent(llm_client=self.llm)
        return self._goal_planner_agent

    def goal_linking_agent(self) -> GoalLinkingAgent:
        if self._goal_linking_agent is None:
            self._goal_linking_agent = GoalLinkingAgent(llm_client=self.llm)
        return self._goal_linking_agent

    def meta_agent(self) -> MetaAgent:
        if self._meta_agent is None:
            self._meta_agent = MetaAgent(self)
        return self._meta_agent
    
    def login(self, email: str, password: str) -> User:
        """
        Authenticate with email and password.
        
        Args:
            email: User's email address
            password: User's password
            
        Returns:
            User: Authenticated user information
        """
        response = self._client.post(
            "/auth/login",
            json={"email": email, "password": password},
        )
        
        self._handle_response_errors(response)
        
        data = response.json()
        tokens = AuthTokens(**data)
        
        self._access_token = tokens.access_token
        self._refresh_token = tokens.refresh_token
        self._token_expires_at = datetime.now() + timedelta(seconds=tokens.expires_in)
        self._current_user = tokens.user
        
        if tokens.user.organization_id and not self.organization_id:
            self.organization_id = tokens.user.organization_id
        
        return tokens.user
    
    def logout(self) -> None:
        """Clear authentication tokens."""
        self._access_token = None
        self._refresh_token = None
        self._token_expires_at = None
        self._current_user = None
    
    def refresh_auth(self) -> None:
        """Refresh the access token using the refresh token."""
        if not self._refresh_token:
            raise AuthenticationError("No refresh token available. Please login again.")
        
        response = self._client.post(
            "/auth/refresh",
            json={"refresh_token": self._refresh_token},
        )
        
        self._handle_response_errors(response)
        
        data = response.json()
        self._access_token = data["access_token"]
        self._token_expires_at = datetime.now() + timedelta(seconds=data.get("expires_in", 1800))
    
    @property
    def current_user(self) -> Optional[User]:
        """Get the currently authenticated user."""
        return self._current_user
    
    @property
    def is_authenticated(self) -> bool:
        """Check if the client is authenticated."""
        return bool(self._access_token or self.api_key)
    
    def _get_headers(self) -> Dict[str, str]:
        """Build request headers with authentication."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Auto-refresh token if expired
        if self._token_expires_at and datetime.now() >= self._token_expires_at:
            if self._refresh_token:
                self.refresh_auth()
        
        # Add authentication
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        elif self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        
        # Add organization context
        if self.organization_id:
            headers["X-Organization-ID"] = self.organization_id
        
        return headers
    
    def _handle_response_errors(self, response: httpx.Response) -> None:
        """Handle HTTP error responses."""
        if response.status_code < 400:
            return
        
        try:
            error_data = response.json()
            message = error_data.get("detail", error_data.get("message", "Unknown error"))
        except Exception:
            message = response.text or f"HTTP {response.status_code}"
        
        if response.status_code == 401:
            raise AuthenticationError(message, response.status_code, error_data if 'error_data' in dir() else {})
        elif response.status_code == 403:
            raise AuthorizationError(message, response.status_code)
        elif response.status_code == 404:
            raise NotFoundError(message, response.status_code)
        elif response.status_code == 422:
            raise ValidationError(message, response.status_code)
        elif response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                message, 
                retry_after=int(retry_after) if retry_after else None,
                status_code=response.status_code
            )
        elif response.status_code >= 500:
            raise ServerError(message, response.status_code)
        else:
            raise NinaiError(message, response.status_code)
    
    def _get(self, path: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a GET request."""
        response = self._client.get(path, headers=self._get_headers(), params=params)
        self._handle_response_errors(response)
        return response.json()
    
    def _post(self, path: str, json: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a POST request."""
        response = self._client.post(path, headers=self._get_headers(), json=json)
        self._handle_response_errors(response)
        return response.json()
    
    def _patch(self, path: str, json: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a PATCH request."""
        response = self._client.patch(path, headers=self._get_headers(), json=json)
        self._handle_response_errors(response)
        return response.json()
    
    def _delete(self, path: str) -> None:
        """Make a DELETE request."""
        response = self._client.delete(path, headers=self._get_headers())
        self._handle_response_errors(response)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the HTTP client."""
        self._client.close()
    
    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()
