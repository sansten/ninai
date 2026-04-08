"""
Example 1: Session Basics

Demonstrates fundamental Ninai2 session and API interaction patterns:
- Authentication and token management
- Creating a session context
- Making authenticated API requests
- Handling errors
"""

import asyncio
import aiohttp
import json
from datetime import datetime


class NinaiClient:
    """Simple Ninai2 API client with auth handling."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.access_token = None
        self.session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        await self.authenticate()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def authenticate(self):
        """Exchange API key for access token."""
        endpoint = f"{self.base_url}/auth/login"
        payload = {
            "username": "api",  # API key auth
            "password": self.api_key,
            "grant_type": "password"
        }
        
        async with self.session.post(endpoint, json=payload) as response:
            if response.status != 200:
                raise RuntimeError(f"Auth failed: {response.status}")
            
            data = await response.json()
            self.access_token = data["access_token"]
            print(f"✓ Authenticated (token expires in {data.get('expires_in', 'N/A')}s)")
    
    def _headers(self) -> dict:
        """Get headers with auth token."""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def get(self, endpoint: str) -> dict:
        """GET request."""
        url = f"{self.base_url}{endpoint}"
        async with self.session.get(url, headers=self._headers()) as response:
            if response.status == 401:
                raise RuntimeError("Auth token expired")
            if response.status >= 400:
                error = await response.text()
                raise RuntimeError(f"API error {response.status}: {error}")
            return await response.json()
    
    async def post(self, endpoint: str, payload: dict) -> dict:
        """POST request."""
        url = f"{self.base_url}{endpoint}"
        async with self.session.post(url, json=payload, headers=self._headers()) as response:
            if response.status == 401:
                raise RuntimeError("Auth token expired")
            if response.status >= 400:
                error = await response.text()
                raise RuntimeError(f"API error {response.status}: {error}")
            return await response.json()


async def main():
    """Run session basics example."""
    
    # Configuration (set these from environment in production)
    base_url = "http://localhost:8000"
    api_key = "test-api-key"  # Get from /api-keys endpoint
    
    print("Ninai2 Session Basics Example")
    print("=" * 50)
    print()
    
    try:
        # Create authenticated client using context manager
        async with NinaiClient(base_url, api_key) as client:
            
            # 1. Get current org info
            print("1. Fetching organization info...")
            org_info = await client.get("/api/v1/org")
            print(f"   Organization: {org_info['name']} ({org_info['id']})")
            print(f"   Tier: {org_info['tier']}")
            print()
            
            # 2. Get current user
            print("2. Fetching user info...")
            user = await client.get("/api/v1/users/me")
            print(f"   User: {user['email']}")
            print(f"   Role: {user.get('role', 'member')}")
            print()
            
            # 3. List available agents
            print("3. Listing available agents...")
            agents = await client.get("/api/v1/agents")
            for agent in agents.get("agents", [])[:3]:  # Show first 3
                print(f"   - {agent['id']}: {agent['purpose']}")
            print()
            
            # 4. Create a new session/interaction
            print("4. Creating new session...")
            session_data = {
                "title": f"Test Session {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "system_prompt": "You are a helpful assistant."
            }
            session = await client.post("/api/v1/sessions", session_data)
            print(f"   Session ID: {session['id']}")
            print()
            
            # 5. Send message to agent
            print("5. Sending message to agent...")
            message_data = {
                "session_id": session['id'],
                "content": "Hello! What can you help me with?",
                "agent_id": "default"  # Use default agent
            }
            response = await client.post("/api/v1/chat", message_data)
            print(f"   Agent: {response['content'][:100]}...")
            print()
            
            # 6. Error handling example
            print("6. Demonstrating error handling...")
            try:
                await client.get("/api/v1/nonexistent")
            except RuntimeError as e:
                print(f"   Caught expected error: {e}")
            print()
            
            print("✓ Session basics example completed successfully!")
    
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is the backend running? (docker-compose ps)")
        print("  2. Check BASE_URL and API_KEY are correct")
        print("  3. Check logs: docker-compose logs backend")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
