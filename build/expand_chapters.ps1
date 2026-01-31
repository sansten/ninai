# Script to expand all remaining chapters (12-28) to production depth
# Usage: powershell -ExecutionPolicy Bypass -File expand_chapters.ps1

$projectRoot = "d:\Sansten\Projects\Ninai2"
$chaptersDir = "$projectRoot\book\chapters"

# Chapter 12 content
$ch12_content = @"
# Chapter 12 — LangChain and LangGraph Integration

## Problem Introduction
LangChain and LangGraph are powerful orchestration frameworks. How do you integrate them with governance, memory, and policy? Wrap LangGraph as your PEC loop, instrument every edge with memory hooks and audit logging, wrap every LangChain tool with PolicyGuard.

## Architecture-Level Explanation

```mermaid
flowchart TD
    Start["Start"] --> PlanNode["Plan Node<br/>(generate steps)"]
    PlanNode -->|memory_hook| MemAppend1["Memory Append"]
    MemAppend1 -->|audit_log| AuditLog1["Audit Log"]
    AuditLog1 --> ExecuteNode["Execute Node<br/>(run tools)"]
    ExecuteNode -->|for each tool| PolicyGuard["PolicyGuard<br/>(allow/deny)"]
    PolicyGuard -->|allow| ToolExec["Tool Execute"]
    PolicyGuard -->|deny| ErrorLog["Log Violation"]
    ToolExec -->|event_emit| EventEmit["Emit Event"]
    EventEmit --> CritiqueNode["Critique Node<br/>(evaluate)"]
    CritiqueNode -->|memory_hook| MemAppend2["Memory Append"]
    MemAppend2 --> Decision{"Goal<br/>Satisfied?"}
    Decision -->|no| PlanNode
    Decision -->|yes| End["End"]
```

## Production Code Blocks

### LangGraph State Machine

```python
from langgraph.graph import StateGraph
from typing import TypedDict, List

class AgentState(TypedDict):
    goal: str
    session_id: str
    agent_id: str
    tenant_id: str
    plan_steps: List[dict]
    execution_results: List[dict]
    critique_feedback: dict
    iteration: int
    max_iterations: int

class LangGraphIntegration:
    def __init__(self, memory_client, policy_guard, llm_client, executor, db):
        self.memory = memory_client
        self.policy_guard = policy_guard
        self.llm = llm_client
        self.executor = executor
        self.db = db
        self.graph = self._build_graph()
    
    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("execute", self._execute_node)
        graph.add_node("critique", self._critique_node)
        graph.add_edge("plan", "execute")
        graph.add_edge("execute", "critique")
        graph.add_conditional_edges(
            "critique",
            lambda s: "plan" if s["iteration"] < s["max_iterations"] and not s["critique_feedback"].get("goal_satisfied") else "end",
            {"plan": "plan", "end": "__end__"}
        )
        graph.set_entry_point("plan")
        return graph.compile()
    
    async def _plan_node(self, state):
        await self._log_state_transition("plan_enter", state)
        prompt = f"Goal: {state['goal']}\nGenerate steps."
        response = await self.llm.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        state["plan_steps"] = self._parse_plan_steps(response.choices[0].message.content)
        await self.memory.append(state["session_id"], {"type": "plan", "iteration": state["iteration"], "steps": state["plan_steps"]})
        await self._audit_log("plan_generated", state, {"step_count": len(state["plan_steps"])})
        return state
    
    async def _execute_node(self, state):
        await self._log_state_transition("execute_enter", state)
        results = []
        for step in state["plan_steps"]:
            tool_name = step.get("tool_name")
            if not tool_name:
                results.append({"step": step["step_number"], "status": "skipped"})
                continue
            policy_check = await self.policy_guard.allow(tool_name, step.get("parameters", {}), context=state)
            if not policy_check["allowed"]:
                results.append({"step": step["step_number"], "status": "blocked", "reason": policy_check.get("reason")})
                await self._audit_log("tool_denied", state, {"tool": tool_name})
                continue
            try:
                output = await self.executor.call_tool(tool_name, step.get("parameters", {}))
                results.append({"step": step["step_number"], "status": "success", "tool": tool_name, "output": output})
            except Exception as e:
                results.append({"step": step["step_number"], "status": "error", "tool": tool_name, "error": str(e)})
        state["execution_results"].append({"iteration": state["iteration"], "results": results})
        await self.memory.append(state["session_id"], {"type": "execution", "iteration": state["iteration"], "results": results})
        await self._audit_log("execution_completed", state, {"result_count": len(results)})
        return state
    
    async def _critique_node(self, state):
        await self._log_state_transition("critique_enter", state)
        latest_results = state["execution_results"][-1]["results"] if state["execution_results"] else []
        prompt = f"Goal: {state['goal']}\nResults: {json.dumps(latest_results)}\nEvaluate goal achievement (true/false) and suggest improvement."
        response = await self.llm.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.2)
        state["critique_feedback"] = json.loads(response.choices[0].message.content)
        state["iteration"] += 1
        await self.memory.append(state["session_id"], {"type": "critique", "iteration": state["iteration"]-1, "feedback": state["critique_feedback"]})
        await self._audit_log("critique_completed", state, {"goal_satisfied": state["critique_feedback"].get("goal_satisfied")})
        return state
    
    async def _log_state_transition(self, event_type, state):
        await self.db.execute(insert(StateTransitionLog).values(session_id=state["session_id"], event_type=event_type, iteration=state["iteration"]))
        await self.db.commit()
    
    async def _audit_log(self, action, state, details):
        log_entry = AuditLog(action=action, agent_id=state["agent_id"], tenant_id=state["tenant_id"], session_id=state["session_id"], details=details)
        self.db.add(log_entry)
        await self.db.commit()
```

### Tool Wrapping with PolicyGuard

```python
from langchain.tools import BaseTool

class PolicyGuardedTool(BaseTool):
    name: str
    description: str
    tool_name: str
    policy_guard: object
    underlying_tool: BaseTool
    tenant_id: str
    session_id: str
    
    async def _arun(self, *args, **kwargs):
        policy_result = await self.policy_guard.allow(self.tool_name, kwargs, context={"tenant_id": self.tenant_id, "session_id": self.session_id})
        if not policy_result["allowed"]:
            raise PermissionError(f"Tool {self.tool_name} denied: {policy_result.get('reason')}")
        try:
            result = await self.underlying_tool.ainvoke(kwargs)
            return result
        except Exception as e:
            logger.error(f"Tool {self.tool_name} execution failed: {e}")
            raise

def build_tool_registry(policy_guard, tenant_id, session_id):
    tools = {}
    memory_search_tool = Tool(name="memory_search", func=lambda query: memory_client.search(query, limit=5), description="Search memory")
    tools["memory_search"] = PolicyGuardedTool(name="memory_search", description="Search long-term memory", tool_name="memory.search", policy_guard=policy_guard, underlying_tool=memory_search_tool, tenant_id=tenant_id, session_id=session_id)
    payment_refund_tool = Tool(name="process_refund", func=lambda tid, amt: payment_service.refund(tid, amt), description="Process refund")
    tools["process_refund"] = PolicyGuardedTool(name="process_refund", description="Process refund", tool_name="payment.refund", policy_guard=policy_guard, underlying_tool=payment_refund_tool, tenant_id=tenant_id, session_id=session_id)
    return tools
```

## Schemas

```sql
CREATE TABLE state_transition_logs (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    iteration INT NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    INDEX ON (session_id, iteration)
);

CREATE TABLE agent_events (
    id BIGSERIAL PRIMARY KEY,
    session_id UUID NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    data JSONB NOT NULL,
    timestamp TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    INDEX ON (session_id, timestamp DESC)
);

CREATE TABLE audit_logs (
    id BIGSERIAL PRIMARY KEY,
    action VARCHAR(100) NOT NULL,
    agent_id UUID NOT NULL,
    tenant_id UUID NOT NULL,
    session_id UUID NOT NULL,
    details JSONB,
    timestamp TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (agent_id) REFERENCES agents(id),
    INDEX ON (agent_id, timestamp DESC)
);
```

## Design Takeaways

**1. Make orchestration explicit**: Nodes = functions, edges = audited transitions.

**2. Policy guard every tool**: Wrap LangChain tools, don't assume they're safe.

**3. Hooks must be lightweight**: Pre/post-edge callbacks <100ms to avoid slowdown.

**4. State is shared context**: Pass tenant_id, session_id, agent_id through state for RLS and audit.

**5. Audit everything**: Every transition, every tool call—immutable audit trail.
"@

# Write Chapter 12
Set-Content -Path "$chaptersDir\ch12.md" -Value $ch12_content -Encoding UTF8

Write-Host "✓ Expanded Chapter 12: LangChain Integration"

# Chapter 13 content
$ch13_content = @"
# Chapter 13 — Multi-Provider LLM Routing

## Problem Introduction

One LLM provider isn't enough. OpenAI has outages. Ollama runs locally for privacy. Claude excels at reasoning. How do you route queries to the right provider? Multi-provider routing load-balances, falls back on failure, and optimizes cost by routing simple queries to cheaper models.

## Architecture-Level Explanation

```
User Query
   ↓
[Router] → classify complexity (simple/moderate/complex)
   ↓
Simple → Ollama (local, cheap, fast)
Moderate → GPT-3.5 (cheaper than 4, acceptable latency)
Complex → GPT-4 (expensive, but necessary for reasoning)
   ↓
[Fallback Chain] → if provider fails, try next
   ↓
Response
```

```mermaid
flowchart TD
    Query["User Query"] --> Classify["Classify<br/>(complexity)"]
    Classify -->|simple| RoutOllama["Route to Ollama"]
    Classify -->|moderate| RoutGPT35["Route to GPT-3.5"]
    Classify -->|complex| RoutGPT4["Route to GPT-4"]
    RoutOllama -->|success| Response["Return Response"]
    RoutOllama -->|fail| RoutGPT35
    RoutGPT35 -->|success| Response
    RoutGPT35 -->|fail| RoutGPT4
    RoutGPT4 -->|success| Response
```

## Production Code Blocks

### Multi-Provider Router

```python
class MultiProviderRouter:
    def __init__(self, openai_client, ollama_client, anthropic_client):
        self.openai = openai_client
        self.ollama = ollama_client
        self.anthropic = anthropic_client
        self.provider_costs = {"ollama": 0.00, "gpt-3.5": 0.002, "gpt-4": 0.03}
    
    async def route_and_call(self, prompt: str, context: dict = None) -> dict:
        complexity = self._classify_complexity(prompt)
        providers = self._get_provider_chain(complexity)
        
        for provider_name in providers:
            try:
                result = await self._call_provider(provider_name, prompt)
                return {"status": "success", "provider": provider_name, "response": result, "complexity": complexity}
            except Exception as e:
                logger.warning(f"Provider {provider_name} failed: {e}, trying next...")
                continue
        
        return {"status": "failed", "error": "All providers exhausted"}
    
    def _classify_complexity(self, prompt: str) -> str:
        token_count = len(prompt.split())
        if "reasoning" in prompt.lower() or "why" in prompt.lower() or "analyze" in prompt.lower():
            return "complex"
        elif token_count > 200 or "code" in prompt.lower():
            return "moderate"
        else:
            return "simple"
    
    def _get_provider_chain(self, complexity: str) -> list:
        if complexity == "simple":
            return ["ollama", "gpt-3.5", "gpt-4"]
        elif complexity == "moderate":
            return ["gpt-3.5", "ollama", "gpt-4"]
        else:  # complex
            return ["gpt-4", "anthropic-claude", "gpt-3.5"]
    
    async def _call_provider(self, provider: str, prompt: str) -> str:
        if provider == "ollama":
            response = await self.ollama.chat.completions.create(model="mistral", messages=[{"role": "user", "content": prompt}], temperature=0.7, timeout=5)
        elif provider == "gpt-3.5":
            response = await self.openai.chat.completions.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}], temperature=0.7, timeout=10)
        elif provider == "gpt-4":
            response = await self.openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": prompt}], temperature=0.3, timeout=15)
        elif provider == "anthropic-claude":
            response = await self.anthropic.messages.create(model="claude-3-sonnet", messages=[{"role": "user", "content": prompt}], temperature=0.3)
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        return response.choices[0].message.content if hasattr(response, 'choices') else response.content[0].text
```

### Cost Tracking and Optimization

```python
class ProviderCostTracker:
    def __init__(self, db):
        self.db = db
    
    async def log_call(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int, session_id: str):
        cost = self._calculate_cost(provider, model, prompt_tokens, completion_tokens)
        log_entry = ProviderCallLog(
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            session_id=session_id,
            timestamp=datetime.utcnow(),
        )
        self.db.add(log_entry)
        await self.db.commit()
    
    def _calculate_cost(self, provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        rates = {
            ("openai", "gpt-3.5-turbo"): {"input": 0.0005, "output": 0.0015},
            ("openai", "gpt-4"): {"input": 0.03, "output": 0.06},
            ("anthropic", "claude-3-sonnet"): {"input": 0.003, "output": 0.015},
            ("ollama", "mistral"): {"input": 0.0, "output": 0.0},
        }
        rate = rates.get((provider, model), {"input": 0.001, "output": 0.002})
        return (prompt_tokens * rate["input"] + completion_tokens * rate["output"]) / 1000
    
    async def get_cost_summary(self, session_id: str) -> dict:
        logs = await self.db.execute(
            select(ProviderCallLog).where(ProviderCallLog.session_id == session_id)
        ).scalars().all()
        
        total_cost = sum(log.cost for log in logs)
        by_provider = {}
        for log in logs:
            if log.provider not in by_provider:
                by_provider[log.provider] = {"calls": 0, "cost": 0.0}
            by_provider[log.provider]["calls"] += 1
            by_provider[log.provider]["cost"] += log.cost
        
        return {"total_cost": total_cost, "by_provider": by_provider, "call_count": len(logs)}
```

## Schemas

```sql
CREATE TABLE provider_call_logs (
    id BIGSERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    prompt_tokens INT,
    completion_tokens INT,
    cost DECIMAL(10, 6),
    session_id UUID,
    timestamp TIMESTAMP DEFAULT NOW(),
    INDEX ON (provider, timestamp DESC),
    INDEX ON (session_id, timestamp DESC)
);
```

## Design Takeaways

**1. Route by complexity**: Simple → cheap, complex → capable.

**2. Fallback chains prevent outages**: If primary fails, try secondary.

**3. Track cost per provider**: Optimize routing based on actual spend.

**4. Monitor provider health**: Flag providers with high error rates.

**5. Batch cheap queries to local models**: Ollama on premise saves 95% cost for simple queries.
"@

Set-Content -Path "$chaptersDir\ch13.md" -Value $ch13_content -Encoding UTF8
Write-Host "✓ Expanded Chapter 13: Multi-Provider LLM Routing"

Write-Host "`nSuccessfully expanded Chapters 12-13. Token budget preserved for remaining chapters (14-28)."
Write-Host "Next batch: Ch14-17 (Planning, Cloud, On-Prem, Observability)"
