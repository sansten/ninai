"""Bootstrap Meridian Technologies demo org for the 5-case notebook.

Creates one company with two divisions, five teams, and fifteen users.
Run this as a subprocess from the notebook so it gets its own event loop.

Company: Meridian Technologies
  Division: Engineering
    Team: devops        — 5 members  (Cases 1, 2, 3)
    Team: security      — 3 members  (Case 3)
    Team: sre           — 3 members  (Cases 4, 5)
  Division: Operations
    Team: support-l1    — 3 members  (CSR-facing, all cases)
    Team: support-l2    — 2 members  (escalation, Cases 3, 4)

Usage (from notebook):
    import subprocess, textwrap, json
    result = subprocess.run(
        [sys.executable, 'notebooks/demo/bootstrap.py'],
        capture_output=True, text=True, cwd=str(backend_dir)
    )
    org_data = json.loads(result.stdout)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Meridian Technologies org definition
# ---------------------------------------------------------------------------

ORG_SLUG = "meridian-technologies"
ORG_NAME = "Meridian Technologies"

ADMIN_EMAIL = "admin@meridian-tech.com"
ADMIN_PASSWORD = "MeridianDemo2026!"

# team_slug -> {name, description, division}
TEAMS = {
    "devops": {
        "name": "DevOps",
        "description": "Platform engineering, CI/CD, deployments",
        "division": "Engineering",
    },
    "security": {
        "name": "Security",
        "description": "Auth, certs, IAM, threat response",
        "division": "Engineering",
    },
    "sre": {
        "name": "SRE",
        "description": "Reliability, on-call, incident response",
        "division": "Engineering",
    },
    "support-l1": {
        "name": "Support L1",
        "description": "First-line customer support (CSR)",
        "division": "Operations",
    },
    "support-l2": {
        "name": "Support L2",
        "description": "Escalation, complex incidents",
        "division": "Operations",
    },
}

# email -> {full_name, team_slug, team_role}
USERS = {
    # DevOps — Cases 1, 2, 3
    "alex.rivera@meridian-tech.com": {
        "full_name": "Alex Rivera",
        "team": "devops",
        "team_role": "lead",
    },
    "priya.patel@meridian-tech.com": {
        "full_name": "Priya Patel",
        "team": "devops",
        "team_role": "member",
    },
    "marcus.johnson@meridian-tech.com": {
        "full_name": "Marcus Johnson",
        "team": "devops",
        "team_role": "member",
    },
    "sofia.chen@meridian-tech.com": {
        "full_name": "Sofia Chen",
        "team": "devops",
        "team_role": "member",
    },
    "ryan.osei@meridian-tech.com": {
        "full_name": "Ryan Osei",
        "team": "devops",
        "team_role": "member",
    },
    # Security — Case 3 (conflict), Case 4
    "james.kim@meridian-tech.com": {
        "full_name": "James Kim",
        "team": "security",
        "team_role": "lead",
    },
    "natasha.volkov@meridian-tech.com": {
        "full_name": "Natasha Volkov",
        "team": "security",
        "team_role": "member",
    },
    "yuki.tanaka@meridian-tech.com": {
        "full_name": "Yuki Tanaka",
        "team": "security",
        "team_role": "member",
    },
    # SRE — Cases 4, 5
    "tom.bradley@meridian-tech.com": {
        "full_name": "Tom Bradley",
        "team": "sre",
        "team_role": "lead",
    },
    "aisha.hassan@meridian-tech.com": {
        "full_name": "Aisha Hassan",
        "team": "sre",
        "team_role": "member",
    },
    "luca.ferrari@meridian-tech.com": {
        "full_name": "Luca Ferrari",
        "team": "sre",
        "team_role": "member",
    },
    # Support L1 — all cases (CSR-facing)
    "carlos.mendez@meridian-tech.com": {
        "full_name": "Carlos Mendez",
        "team": "support-l1",
        "team_role": "lead",
    },
    "sarah.okafor@meridian-tech.com": {
        "full_name": "Sarah Okafor",
        "team": "support-l1",
        "team_role": "member",
    },
    "mei.zhang@meridian-tech.com": {
        "full_name": "Mei Zhang",
        "team": "support-l1",
        "team_role": "member",
    },
    # Support L2 — Cases 3, 4 (escalation)
    "oliver.smith@meridian-tech.com": {
        "full_name": "Oliver Smith",
        "team": "support-l2",
        "team_role": "lead",
    },
    "grace.chen@meridian-tech.com": {
        "full_name": "Grace Chen",
        "team": "support-l2",
        "team_role": "member",
    },
}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

async def _run() -> dict:
    """Create org, teams, and users. Return IDs as dict."""
    os.environ.setdefault("AGENT_STRATEGY", "heuristic")

    from app.core.database import async_session_factory
    from app.core.security import get_password_hash
    from app.models import Organization, User, Role, UserRole
    from app.models.team import Team, TeamMember
    from sqlalchemy import select

    result: dict = {"org_id": None, "teams": {}, "users": {}}

    async with async_session_factory() as db:
        # --- org ---
        org = (
            await db.execute(
                select(Organization).where(Organization.slug == ORG_SLUG)
            )
        ).scalar_one_or_none()
        if org is None:
            org = Organization(name=ORG_NAME, slug=ORG_SLUG)
            db.add(org)
            await db.flush()
        result["org_id"] = org.id

        # --- teams ---
        team_objects: dict[str, Team] = {}
        for slug, info in TEAMS.items():
            team = (
                await db.execute(
                    select(Team).where(
                        Team.organization_id == org.id,
                        Team.slug == slug,
                    )
                )
            ).scalar_one_or_none()
            if team is None:
                team = Team(
                    organization_id=org.id,
                    name=info["name"],
                    slug=slug,
                    description=info["description"],
                    settings={"division": info["division"]},
                )
                db.add(team)
                await db.flush()
            team_objects[slug] = team
            result["teams"][slug] = {
                "team_id": team.id,
                "name": info["name"],
                "division": info["division"],
                "members": [],
            }

        # --- admin role lookup ---
        admin_role = (
            await db.execute(select(Role).where(Role.name == "admin"))
        ).scalar_one_or_none()

        # --- users ---
        for email, info in USERS.items():
            user = (
                await db.execute(select(User).where(User.email == email))
            ).scalar_one_or_none()
            if user is None:
                user = User(
                    email=email,
                    hashed_password=get_password_hash(ADMIN_PASSWORD),
                    is_active=True,
                    full_name=info["full_name"],
                )
                db.add(user)
                await db.flush()

            # org role
            if admin_role:
                existing_ur = (
                    await db.execute(
                        select(UserRole).where(
                            UserRole.user_id == user.id,
                            UserRole.organization_id == org.id,
                        )
                    )
                ).scalar_one_or_none()
                if not existing_ur:
                    db.add(
                        UserRole(
                            id=str(uuid.uuid4()),
                            user_id=user.id,
                            organization_id=org.id,
                            role_id=admin_role.id,
                        )
                    )

            # team membership
            team_slug = info["team"]
            team_obj = team_objects[team_slug]
            existing_tm = (
                await db.execute(
                    select(TeamMember).where(
                        TeamMember.team_id == team_obj.id,
                        TeamMember.user_id == user.id,
                    )
                )
            ).scalar_one_or_none()
            if not existing_tm:
                db.add(
                    TeamMember(
                        team_id=team_obj.id,
                        user_id=user.id,
                        organization_id=org.id,
                        role=info["team_role"],
                        joined_at=datetime.now(timezone.utc),
                    )
                )

            result["users"][email] = {
                "user_id": user.id,
                "full_name": info["full_name"],
                "team": team_slug,
                "team_role": info["team_role"],
            }
            result["teams"][team_slug]["members"].append(email)

        await db.commit()

    return result


if __name__ == "__main__":
    data = asyncio.run(_run())
    print(json.dumps(data))
