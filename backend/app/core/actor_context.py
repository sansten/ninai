"""Actor context normalization for read/write personalization and auditing.

This module is the single common place that defines supported actor types,
role responsibilities, and anonymous fallback behavior.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional


class ActorType(str, Enum):
    EMPLOYEE = "employee"
    BOT = "bot"
    ANONYMOUS = "anonymous"


ROLE_RESPONSIBILITIES: dict[str, str] = {
    "anonymous": "No role-specific responsibilities. Generic access only.",
    "employee_general": "General employee responsibilities across standard workflows.",
    "employee_support": "Customer support communications, triage, and status updates.",
    "employee_engineering": "Technical diagnosis, remediation, validation, and deployment safety.",
    "employee_leadership": "Business risk assessment, escalation decisions, and stakeholder communication.",
    "bot_observer": "Automated observation, telemetry collection, and passive monitoring.",
    "bot_operator": "Automated operational execution, runbooks, and controlled system actions.",
}


def normalize_actor_context(
    *,
    actor_id: Optional[str],
    actor_type: Optional[ActorType | str],
    role: Optional[str],
    responsibility: Optional[str],
) -> dict[str, str | bool]:
    """Normalize actor context with safe anonymous defaults.

    Rules:
    - Missing actor_type defaults to anonymous.
    - Missing role defaults to anonymous.
    - Anonymous forces actor_id=anonymous and role=anonymous.
    - Responsibility defaults from ROLE_RESPONSIBILITIES for the normalized role.
    """

    normalized_type = str(actor_type.value if isinstance(actor_type, ActorType) else (actor_type or "")).strip().lower()
    if normalized_type not in {ActorType.EMPLOYEE.value, ActorType.BOT.value, ActorType.ANONYMOUS.value}:
        normalized_type = ActorType.ANONYMOUS.value

    normalized_role = (role or "anonymous").strip().lower() or "anonymous"

    if normalized_type == ActorType.ANONYMOUS.value:
        normalized_actor_id = "anonymous"
        normalized_role = "anonymous"
    else:
        normalized_actor_id = (actor_id or "anonymous").strip() or "anonymous"

    normalized_responsibility = (responsibility or "").strip()
    if not normalized_responsibility:
        normalized_responsibility = ROLE_RESPONSIBILITIES.get(
            normalized_role,
            ROLE_RESPONSIBILITIES["anonymous"],
        )

    return {
        "actor_id": normalized_actor_id,
        "actor_type": normalized_type,
        "role": normalized_role,
        "responsibility": normalized_responsibility,
        "is_anonymous": normalized_type == ActorType.ANONYMOUS.value,
        "role_specific": normalized_type != ActorType.ANONYMOUS.value and normalized_role != "anonymous",
    }
