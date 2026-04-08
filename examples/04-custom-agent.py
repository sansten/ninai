"""
Example 4: Custom Agent Configuration

Demonstrates building and configuring custom agents:
- Agent manifest creation
- Function binding and schemas
- Multi-provider LLM routing
- Agent state and context
"""

import asyncio
import aiohttp
import json
from typing import Optional


class AgentConfigurator:
    """Configure and manage custom agents."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}
    
    async def create_agent(self, agent_config: dict) -> dict:
        """Create a custom agent."""
        async with self.session.post(
            f"{self.base_url}/api/v1/agents",
            json=agent_config,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Failed to create agent: {resp.status} {await resp.text()}")
            return await resp.json()
    
    async def configure_llm(self, agent_id: str, llm_config: dict) -> dict:
        """Configure LLM routing for an agent."""
        async with self.session.put(
            f"{self.base_url}/api/v1/agents/{agent_id}/llm-config",
            json=llm_config,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"LLM config failed: {resp.status}")
            return await resp.json()
    
    async def bind_function(self, agent_id: str, function_schema: dict) -> dict:
        """Bind an external function to an agent."""
        async with self.session.post(
            f"{self.base_url}/api/v1/agents/{agent_id}/functions",
            json=function_schema,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Function binding failed: {resp.status}")
            return await resp.json()
    
    async def invoke_agent(self, agent_id: str, prompt: str, context: Optional[dict] = None) -> dict:
        """Invoke the agent with a prompt."""
        payload = {
            "prompt": prompt,
            "context": context or {}
        }
        async with self.session.post(
            f"{self.base_url}/api/v1/agents/{agent_id}/invoke",
            json=payload,
            headers=self._headers()
        ) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"Invocation failed: {resp.status}")
            return await resp.json()


async def main():
    """Run custom agent configuration example."""
    
    base_url = "http://localhost:8000"
    api_key = "test-api-key"
    
    print("Ninai2 Custom Agent Configuration Example")
    print("=" * 50)
    print()
    
    try:
        async with AgentConfigurator(base_url, api_key) as agent_cfg:
            
            # 1. Create a specialized research agent
            print("1. Creating specialized research agent...")
            research_agent = {
                "name": "Research Assistant",
                "purpose": "Deep analysis and research synthesis",
                "system_prompt": """You are a research specialist focused on:
1. Breaking down complex topics into components
2. Identifying knowledge gaps
3. Synthesizing information from multiple sources
4. Creating actionable insights

Always cite sources and maintain academic rigor.""",
                "capabilities": ["semantic_search", "reasoning", "synthesis"],
                "metadata": {
                    "domain": "research",
                    "risk_level": "low"
                }
            }
            agent = await agent_cfg.create_agent(research_agent)
            agent_id = agent["id"]
            print(f"✓ Created agent: {agent['name']} ({agent_id})")
            print()
            
            # 2. Configure multi-provider LLM routing
            print("2. Configuring LLM routing...")
            llm_config = {
                "primary_provider": "openai",  # openai, anthropic, ollama
                "primary_model": "gpt-4",
                "fallback_provider": "ollama",
                "fallback_model": "qwen2.5:7b",
                "temperature": 0.7,
                "max_tokens": 2000,
                "routing_strategy": "cost-optimized"  # cost-optimized, performance, latency
            }
            llm_result = await agent_cfg.configure_llm(agent_id, llm_config)
            print(f"✓ Configured LLM: {llm_result.get('primary_provider')} with {llm_result.get('fallback_provider')} fallback")
            print()
            
            # 3. Bind external functions
            print("3. Binding external functions...")
            
            # Web search function
            search_function = {
                "name": "search_web",
                "description": "Search the internet for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "num_results": {
                            "type": "integer",
                            "description": "Number of results to return (1-20)"
                        }
                    },
                    "required": ["query"]
                },
                "handler_endpoint": "https://api.example.com/search",
                "auth_type": "api_key"
            }
            search_fn = await agent_cfg.bind_function(agent_id, search_function)
            print(f"✓ Bound function: search_web")
            
            # Database query function
            db_function = {
                "name": "query_database",
                "description": "Execute SQL queries against the knowledge database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql": {
                            "type": "string",
                            "description": "SQL SELECT query"
                        }
                    },
                    "required": ["sql"]
                },
                "handler_endpoint": "/api/v1/database/query",
                "auth_type": "bearer_token"
            }
            db_fn = await agent_cfg.bind_function(agent_id, db_function)
            print(f"✓ Bound function: query_database")
            print()
            
            # 4. Show agent configuration
            print("4. Agent Configuration Summary:")
            print(f"   Name: {research_agent['name']}")
            print(f"   Purpose: {research_agent['purpose']}")
            print(f"   Capabilities: {', '.join(research_agent['capabilities'])}")
            print(f"   LLM: {llm_config['primary_provider']} ({llm_config['primary_model']})")
            print(f"   Fallback: {llm_config['fallback_provider']} ({llm_config['fallback_model']})")
            print(f"   Functions: search_web, query_database")
            print()
            
            # 5. Test agent invocation (if endpoint exists)
            print("5. Testing agent invocation...")
            try:
                result = await agent_cfg.invoke_agent(
                    agent_id,
                    prompt="What are the latest trends in AI safety?",
                    context={"domain": "AI", "academic_rigor": "high"}
                )
                print(f"✓ Agent response: {result.get('response', 'N/A')[:100]}...")
            except RuntimeError as e:
                print(f"⚠ Invocation test skipped: {e}")
            print()
            
            print("✓ Custom agent configuration completed successfully!")
            print("\nKey Features Demonstrated:")
            print("  - Custom system prompts for agent specialization")
            print("  - Multi-provider LLM routing with fallback strategy")
            print("  - Function binding for external tools and APIs")
            print("  - Agent context and metadata for lifecycle management")
    
    except RuntimeError as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check API_KEY and BASE_URL")
        print("  2. Verify agent creation endpoints are available")
        print("  3. Check LLM provider API keys are configured")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
