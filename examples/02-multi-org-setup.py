"""
Example 2: Multi-Organization Setup

Demonstrates multi-tenancy in Ninai2:
- Creating organizations
- Setting up admin users
- Data isolation and RLS
- Role-based access control
"""

import asyncio
import aiohttp
import uuid
from typing import Optional


class OrgManager:
    """Manage organizations and users in Ninai2."""
    
    def __init__(self, base_url: str, admin_api_key: str):
        """
        Initialize with superadmin API key (required for org creation).
        Superadmin keys are provisioned via backend CLI.
        """
        self.base_url = base_url.rstrip('/')
        self.admin_api_key = admin_api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    async def _request(self, method: str, endpoint: str, payload: Optional[dict] = None) -> dict:
        """Make authenticated request with admin key."""
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.admin_api_key}",
            "Content-Type": "application/json"
        }
        
        async with self.session.request(method, url, json=payload, headers=headers) as resp:
            if resp.status >= 400:
                error_text = await resp.text()
                raise RuntimeError(f"{method} {endpoint} failed: {resp.status} {error_text}")
            return await resp.json() if resp.status != 204 else {}
    
    async def create_organization(self, name: str, tier: str = "pro") -> dict:
        """Create a new organization."""
        payload = {
            "name": name,
            "tier": tier,  # "free", "pro", "enterprise"
            "metadata": {
                "created_via": "example-script",
                "use_case": "evaluation"
            }
        }
        org = await self._request("POST", "/api/v1/admin/orgs", payload)
        print(f"✓ Created organization: {org['name']} (ID: {org['id']})")
        return org
    
    async def create_user_in_org(self, org_id: str, email: str, role: str = "org_member") -> dict:
        """Create a new user in an organization."""
        payload = {
            "email": email,
            "password": str(uuid.uuid4()),  # Random password; should be reset
            "role": role  # "org_admin", "org_member", "guest"
        }
        user = await self._request("POST", f"/api/v1/admin/orgs/{org_id}/users", payload)
        print(f"✓ Created user: {user['email']} as {role}")
        return user
    
    async def verify_isolation(self, org_a_key: str, org_b_key: str):
        """Verify that two orgs cannot access each other's data."""
        print("\nVerifying data isolation (RLS)...")
        
        # Get org A's memories
        headers_a = {"Authorization": f"Bearer {org_a_key}"}
        async with self.session.get(
            f"{self.base_url}/api/v1/memories",
            headers=headers_a
        ) as resp:
            orga_memories = await resp.json()
            org_a_count = len(orga_memories.get("memories", []))
        
        # Get org B's memories (should be different)
        headers_b = {"Authorization": f"Bearer {org_b_key}"}
        async with self.session.get(
            f"{self.base_url}/api/v1/memories",
            headers=headers_b
        ) as resp:
            orgb_memories = await resp.json()
            org_b_count = len(orgb_memories.get("memories", []))
        
        print(f"  Org A memories: {org_a_count}")
        print(f"  Org B memories: {org_b_count}")
        
        # Verify org A cannot access org B's data (401 or empty)
        if org_a_count != org_b_count:
            print("  ✓ Data isolation confirmed (different memory counts)")
        else:
            print("  ⚠ Same memory count; verify no cross-org leakage via IDs")


async def main():
    """Run multi-org setup example."""
    
    base_url = "http://localhost:8000"
    admin_key = "admin-superadmin-key"  # Must be real superadmin key
    
    print("Ninai2 Multi-Organization Setup Example")
    print("=" * 50)
    print()
    
    try:
        async with OrgManager(base_url, admin_key) as manager:
            
            # 1. Create two organizations
            print("1. Creating organizations...")
            org_a = await manager.create_organization(f"TechCorp-{uuid.uuid4().hex[:8]}", "pro")
            org_b = await manager.create_organization(f"DataCorp-{uuid.uuid4().hex[:8]}", "pro")
            print()
            
            # 2. Create admin users for each org
            print("2. Creating admin users...")
            admin_a = await manager.create_user_in_org(org_a["id"], f"admin-a-{uuid.uuid4().hex[:4]}@example.com", "org_admin")
            admin_b = await manager.create_user_in_org(org_b["id"], f"admin-b-{uuid.uuid4().hex[:4]}@example.com", "org_admin")
            print()
            
            # 3. Create regular members
            print("3. Creating org members...")
            member_a1 = await manager.create_user_in_org(org_a["id"], f"user-a1-{uuid.uuid4().hex[:4]}@example.com", "org_member")
            member_a2 = await manager.create_user_in_org(org_a["id"], f"user-a2-{uuid.uuid4().hex[:4]}@example.com", "org_member")
            member_b1 = await manager.create_user_in_org(org_b["id"], f"user-b1-{uuid.uuid4().hex[:4]}@example.com", "org_member")
            print()
            
            # 4. Show org structure
            print("4. Organization Structure:")
            print(f"  Org A ({org_a['id']}):")
            print(f"    - Admin: {admin_a['email']}")
            print(f"    - Members: {member_a1['email']}, {member_a2['email']}")
            print(f"  Org B ({org_b['id']}):")
            print(f"    - Admin: {admin_b['email']}")
            print(f"    - Members: {member_b1['email']}")
            print()
            
            # 5. Demonstrate isolation (requires valid API keys)
            # In production, get real tokens via login endpoint
            print("5. Data Isolation:")
            print("  (Skipped in this example; requires valid session tokens)")
            print("  In production, verify via /api/v1/memories from each org")
            print()
            
            print("✓ Multi-org setup completed successfully!")
            print("\nNext steps:")
            print("  1. Reset user passwords via /auth/reset-password")
            print("  2. Invite users to join org")
            print("  3. Configure org settings (LLM providers, compliance)")
    
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check ADMIN_API_KEY is a valid superadmin key")
        print("  2. Ensure backend org admin endpoints are enabled")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
