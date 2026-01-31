"""
Seed Data Script
================

Populates the database with initial demo data for development and testing.

Usage:
    python -m scripts.seed_data
"""

import asyncio
import hashlib
import sys
from datetime import datetime, timezone, timedelta
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

# Add parent directory to path for imports
sys.path.insert(0, ".")

from app.core.config import settings
from app.core.security import get_password_hash
from app.models.base import Base
from app.models.organization import Organization, OrganizationHierarchy
from app.models.user import User, Role, UserRole
from app.models.team import Team, TeamMember
from app.models.agent import Agent
from app.models.memory import MemoryMetadata


async def seed_database():
    """Seed the database with demo data."""
    
    engine = create_async_engine(settings.DATABASE_URL, echo=True)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with async_session() as db:
        print("🌱 Seeding database...")
        
        # =====================================================================
        # Create Demo Organization
        # =====================================================================
        print("  Creating organization...")
        
        org = Organization(
            id=str(uuid4()),
            name="Ninai Demo",
            slug="ninai-demo",
            settings={
                "tier": "professional",
                "features": ["vector_search", "audit_logs", "team_sharing"],
                "limits": {"users": 50, "memories": 10000, "agents": 10},
            },
        )
        db.add(org)
        await db.flush()
        
        # =====================================================================
        # Create Hierarchy
        # =====================================================================
        print("  Creating organization hierarchy...")
        
        root_node = OrganizationHierarchy(
            id=str(uuid4()),
            organization_id=org.id,
            name="Ninai Demo",
            node_type="root",
            path="root",
        )
        db.add(root_node)
        await db.flush()
        
        engineering = OrganizationHierarchy(
            id=str(uuid4()),
            organization_id=org.id,
            parent_id=root_node.id,
            name="Engineering",
            node_type="department",
            path="root.engineering",
        )
        db.add(engineering)
        
        product = OrganizationHierarchy(
            id=str(uuid4()),
            organization_id=org.id,
            parent_id=root_node.id,
            name="Product",
            node_type="department",
            path="root.product",
        )
        db.add(product)
        await db.flush()
        
        # =====================================================================
        # Get System Roles
        # =====================================================================
        result = await db.execute(select(Role).where(Role.is_system == True))
        roles = {r.name: r for r in result.scalars().all()}
        
        # If no roles exist, create them
        if not roles:
            print("  Creating system roles...")
            role_data = [
                ("system_admin", "System Admin", "Full system access", ["*:*"]),
                (
                    "org_admin",
                    "Org Admin",
                    "Organization administrator",
                    [
                        "org:*",
                        "users:*",
                        "teams:*",
                        "memory:*",
                        "audit:read",
                        "goal:*",
                        "selfmodel:read:org",
                        "selfmodel:manage:org",
                    ],
                ),
                ("knowledge_reviewer", "Knowledge Reviewer", "Can review and approve/reject knowledge submissions", ["knowledge:review"]),
                (
                    "team_lead",
                    "Team Lead",
                    "Team leader",
                    [
                        "team:*",
                        "memory:read",
                        "memory:create:*",
                        "memory:update:own",
                        "memory:delete:own",
                        "goal:read:team",
                        "goal:create:team",
                        "goal:update:team",
                        "goal:read:personal",
                        "goal:create:personal",
                        "goal:update:own",
                        "selfmodel:read:org",
                    ],
                ),
                (
                    "member",
                    "Member",
                    "Standard member",
                    [
                        "memory:read",
                        "memory:create:personal",
                        "memory:update:own",
                        "memory:delete:own",
                        "goal:read:personal",
                        "goal:create:personal",
                        "goal:update:own",
                        "selfmodel:read:org",
                    ],
                ),
                ("viewer", "Viewer", "Read-only access", ["memory:read"]),
            ]
            for name, display_name, desc, perms in role_data:
                role = Role(
                    id=str(uuid4()),
                    name=name,
                    display_name=display_name,
                    description=desc,
                    permissions=perms,
                    is_system=True,
                )
                db.add(role)
                roles[name] = role
            await db.flush()
        
        # =====================================================================
        # Create Demo Users
        # =====================================================================
        print("  Creating demo users...")
        
        # Admin user
        admin = User(
            id=str(uuid4()),
            email="admin@ninai.dev",
            hashed_password=get_password_hash("admin1234"),
            full_name="Admin User",
        )
        db.add(admin)
        
        # Demo user
        demo = User(
            id=str(uuid4()),
            email="demo@ninai.dev",
            hashed_password=get_password_hash("demo1234"),
            full_name="Demo User",
        )
        db.add(demo)
        
        # Developer user
        dev = User(
            id=str(uuid4()),
            email="dev@ninai.dev",
            hashed_password=get_password_hash("dev12345"),
            full_name="Developer",
        )
        db.add(dev)

        # Knowledge reviewer (non-admin)
        reviewer = User(
            id=str(uuid4()),
            email="reviewer@ninai.dev",
            hashed_password=get_password_hash("review1234"),
            full_name="Knowledge Reviewer",
        )
        db.add(reviewer)
        await db.flush()
        
        # =====================================================================
        # Assign Roles
        # =====================================================================
        print("  Assigning roles...")
        
        # Admin gets org_admin
        db.add(UserRole(
            id=str(uuid4()),
            user_id=admin.id,
            role_id=roles["org_admin"].id,
            organization_id=org.id,
            scope_type="organization",
        ))

        # Admin also gets system_admin (required to access Admin Settings UI)
        db.add(UserRole(
            id=str(uuid4()),
            user_id=admin.id,
            role_id=roles["system_admin"].id,
            organization_id=org.id,
            scope_type="organization",
        ))
        
        # Demo gets member
        db.add(UserRole(
            id=str(uuid4()),
            user_id=demo.id,
            role_id=roles["member"].id,
            organization_id=org.id,
            scope_type="organization",
        ))
        
        # Dev gets team_lead
        db.add(UserRole(
            id=str(uuid4()),
            user_id=dev.id,
            role_id=roles["team_lead"].id,
            organization_id=org.id,
            scope_type="organization",
        ))

        # Reviewer gets knowledge_reviewer
        if "knowledge_reviewer" in roles:
            db.add(UserRole(
                id=str(uuid4()),
                user_id=reviewer.id,
                role_id=roles["knowledge_reviewer"].id,
                organization_id=org.id,
                scope_type="organization",
            ))
        await db.flush()
        
        # =====================================================================
        # Create Teams
        # =====================================================================
        print("  Creating teams...")
        
        ai_team = Team(
            id=str(uuid4()),
            organization_id=org.id,
            hierarchy_node_id=engineering.id,
            name="AI Team",
            slug="ai-team",
            description="Artificial Intelligence and Machine Learning team",
        )
        db.add(ai_team)
        
        platform_team = Team(
            id=str(uuid4()),
            organization_id=org.id,
            hierarchy_node_id=engineering.id,
            name="Platform Team",
            slug="platform-team",
            description="Infrastructure and platform engineering",
        )
        db.add(platform_team)
        await db.flush()
        
        # Add members to teams
        db.add(TeamMember(
            id=str(uuid4()),
            team_id=ai_team.id,
            user_id=dev.id,
            organization_id=org.id,
            role="lead",
            joined_at=datetime.now(),
        ))
        
        db.add(TeamMember(
            id=str(uuid4()),
            team_id=ai_team.id,
            user_id=demo.id,
            organization_id=org.id,
            role="member",
            joined_at=datetime.now(),
        ))
        await db.flush()
        
        # =====================================================================
        # Create Agents
        # =====================================================================
        print("  Creating agents...")
        
        assistant = Agent(
            id=str(uuid4()),
            organization_id=org.id,
            owner_id=admin.id,
            scope="team",
            scope_id=ai_team.id,
            name="Research Assistant",
            description="AI assistant for research and documentation",
            llm_provider="openai",
            llm_model="gpt-4",
            system_prompt="You are a helpful research assistant...",
            config={"temperature": 0.7, "capabilities": ["research", "summarization", "qa"]},
        )
        db.add(assistant)
        
        code_agent = Agent(
            id=str(uuid4()),
            organization_id=org.id,
            owner_id=dev.id,
            scope="team",
            scope_id=ai_team.id,
            name="Code Helper",
            description="AI agent for code review and assistance",
            llm_provider="openai",
            llm_model="gpt-4",
            system_prompt="You are an expert code reviewer...",
            config={"temperature": 0.3, "capabilities": ["code_review", "debugging", "refactoring"]},
        )
        db.add(code_agent)
        await db.flush()
        
        # =====================================================================
        # Create Sample Memories
        # =====================================================================
        print("  Creating sample memories...")
        
        memories_data = [
            {
                "memory_type": "semantic",
                "title": "Project Architecture Overview",
                "content": "The Ninai Memory OS is built with a microservices architecture...",
                "tags": ["architecture", "documentation", "overview"],
                "scope": "organization",
            },
            {
                "memory_type": "procedural",
                "title": "Deployment Checklist",
                "content": "1. Run tests\n2. Build Docker images\n3. Deploy to staging...",
                "tags": ["deployment", "devops", "checklist"],
                "scope": "team",
                "scope_id": platform_team.id,
            },
            {
                "memory_type": "semantic",
                "title": "Sprint 12 Retrospective Notes",
                "content": "What went well: Team collaboration improved...",
                "tags": ["retro", "sprint", "team"],
                "scope": "team",
                "scope_id": ai_team.id,
            },
            {
                "memory_type": "semantic",
                "title": "API Design Guidelines",
                "content": "All APIs should follow RESTful conventions...",
                "tags": ["api", "guidelines", "rest"],
                "scope": "organization",
            },
            {
                "memory_type": "short_term",
                "title": "Current Investigation: Memory Leak",
                "content": "Investigating memory leak in vector search...",
                "tags": ["debugging", "investigation", "active"],
                "scope": "personal",
                "owner_id": dev.id,
            },
        ]
        
        for mem_data in memories_data:
            content = mem_data["content"]
            content_hash = hashlib.sha256(content.encode()).hexdigest()
            memory = MemoryMetadata(
                id=str(uuid4()),
                organization_id=org.id,
                owner_id=str(mem_data.get("owner_id", admin.id)),
                scope=mem_data["scope"],
                scope_id=str(mem_data.get("scope_id")) if mem_data.get("scope_id") else None,
                memory_type=mem_data["memory_type"],
                title=mem_data["title"],
                content_preview=content[:200],
                content_hash=content_hash,
                tags=mem_data["tags"],
                vector_id=str(uuid4()),  # Placeholder - would be set when indexed in Qdrant
                embedding_model="text-embedding-3-small",  # Default embedding model
            )
            db.add(memory)
        
        await db.commit()
        
        print("✅ Database seeded successfully!")
        print(f"\n  Demo credentials:")
        print(f"    Admin:     admin@ninai.dev / admin1234")
        print(f"    Demo user: demo@ninai.dev / demo1234")
        print(f"    Developer: dev@ninai.dev / dev12345")
        print(f"    Reviewer:  reviewer@ninai.dev / review1234")


if __name__ == "__main__":
    asyncio.run(seed_database())
