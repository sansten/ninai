"""
Authentication Schemas
======================

Request and response schemas for authentication endpoints.
"""

from typing import Optional, List
from datetime import datetime

from pydantic import EmailStr

from app.schemas.base import BaseSchema
from app.schemas.organization import OrganizationResponse


class LoginRequest(BaseSchema):
    """Login request with email and password."""
    
    email: EmailStr
    password: str


class LoginResponse(BaseSchema):
    """Login response with tokens and user info."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    user: "UserResponse"
    organization: Optional[OrganizationResponse] = None


class RefreshTokenRequest(BaseSchema):
    """Request to refresh access token."""
    
    refresh_token: str


class TokenResponse(BaseSchema):
    """Token response for refresh."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserResponse(BaseSchema):
    """User information response."""
    
    id: str
    email: str
    full_name: str
    avatar_url: Optional[str] = None
    is_active: bool
    clearance_level: int
    created_at: datetime
    last_login_at: Optional[datetime] = None
    
    # Current org context
    organization_id: Optional[str] = None
    organization_name: Optional[str] = None
    roles: List[str] = []


class UserCreate(BaseSchema):
    """User creation request."""
    
    email: EmailStr
    password: str
    full_name: str
    organization_id: Optional[str] = None


class UserUpdate(BaseSchema):
    """User update request."""
    
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    preferences: Optional[dict] = None


class PasswordChange(BaseSchema):
    """Password change request."""
    
    current_password: str
    new_password: str


class OrgSwitchRequest(BaseSchema):
    """Request to switch organization context."""
    
    organization_id: str


class OidcExchangeRequest(BaseSchema):
    """Exchange an OIDC ID token for Ninai tokens."""

    id_token: str


class AuthMethodsResponse(BaseSchema):
    """Which authentication methods are enabled."""

    password_enabled: bool
    oidc_enabled: bool


# Rebuild models to resolve forward references
LoginResponse.model_rebuild()
