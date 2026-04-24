"""RequesterContext — request-time caller envelope.

Carries everything Ninai knows about *who is calling* a cognitive verb,
*when* they are calling, and *where* they are.  Built once per request by
get_requester_context() in app.middleware.tenant_context and passed to every
cognitive gateway call so the intelligence stack can shape its output without
callers having to describe themselves.

Defined here (rather than in middleware or service) so neither layer imports
from the other — both import from this shared module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class RequesterContext:
    """Request-time context envelope for cognitive gateway calls.

    Populated by get_requester_context() from three sources:
      - Auth token         → user_id, org_id, roles  (always present)
      - Request headers    → timezone, location, org_context  (optional)
      - UserActivityProfile (async DB, Redis-cached 5 min per user)
                           → job_role, dominant_domains, expertise_signals

    Degrades gracefully: job_role / domain fields are None / empty when the
    profile hasn't been synthesised yet.  The urgency_signal is always set.
    """

    # ── Identity (always present) ─────────────────────────────────────────
    user_id: str
    org_id: str
    roles: list[str] = field(default_factory=list)

    # ── Temporal (derived from X-Timezone header + server clock) ─────────
    request_utc: datetime | None = None
    timezone: str = "UTC"
    local_hour: int = 12               # safe mid-day default before resolution

    # ── Location (X-Location header, optional) ───────────────────────────
    location: str | None = None

    # ── Job profile (UserActivityProfile, async, cached) ─────────────────
    job_role: str | None = None
    dominant_domains: list[str] = field(default_factory=list)
    expertise_signals: dict[str, float] = field(default_factory=dict)
    profile_confidence: float = 0.0

    # ── Inferred ──────────────────────────────────────────────────────────
    # crisis      → outside business hours, likely urgent page
    # pre_meeting → early morning or X-Org-Context contains board/meeting/exec
    # end_of_day  → local hour >= 17
    # routine     → everything else
    urgency_signal: str = "routine"
