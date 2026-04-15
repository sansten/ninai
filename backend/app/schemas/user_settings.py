"""
User Settings Schemas
=====================

Request/response schemas for user-level identity preference settings.
"""

from __future__ import annotations

from typing import List, Literal

from app.schemas.base import BaseSchema


class UserIdentityPreferenceUpdate(BaseSchema):
    """Update the current user's identity attribution preference."""

    preference: Literal["full", "role_only", "anonymous"]
    """
    full       — actor_id + role + department stored on memory rows.
    role_only  — role + department only; actor_id omitted from memory rows.
    anonymous  — nothing stored on memory rows (audit table may still capture).
    """


class UserIdentityPreferenceResponse(BaseSchema):
    """Current user's identity attribution preference with org policy context."""

    user_id: str
    preference: str
    mandate_active: bool
    """True when the org mandate overrides this preference."""
    allowed_modes: List[str]
    """Modes the org permits (so UI can disable disallowed options)."""
