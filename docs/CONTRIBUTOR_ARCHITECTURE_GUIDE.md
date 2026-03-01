# Ninai OSS - Architecture Guide for Contributors

**Vision**: Build a memory operating system designed for AI agents, not documents.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [System Architecture](#system-architecture)
4. [Key Components](#key-components)
5. [Data Flow](#data-flow)
6. [Contributing Patterns](#contributing-patterns)
7. [Common Tasks](#common-tasks)

---

## Quick Start

### Prerequisites
- Python 3.11+
- Docker (for local PostgreSQL, Redis, Qdrant)
- PostgreSQL understanding (RLS can be complex)

### Setup
```bash
# 1. Clone the repository
git clone https://github.com/sansten/ninai.git
cd ninai

# 2. Start services
docker-compose up -d

# 3. Install dependencies
pip install -e .

# 4. Run tests
pytest tests/ -v

# 5. Start server
python -m app.main
```

### First Contribution Ideas
- Look at `/tests` - understand what works
- Pick an issue marked `good-first-issue` on GitHub
- Add a feature flag for a new use case
- Improve test coverage (target: >95%)

---

## Core Concepts

### The Problem Ninai Solves

Standard RAG systems treat memory like a **library**:
- Designed for huge, diverse corpora
- Returns top-10 similar chunks
- Collapses in bounded dialogue streams

Ninai treats memory like a **conversation**:
- Small, coherent, temporally-linked sessions
- Redundant across sessions (same user, same problem)
- **Strict isolation** (Company A ≠ Company B's memories)
- **Governance** (no bad info → agent decisions)

### Three Memory Layers

#### 1. **Short-Term Memory (STM)** - Redis
- **Purpose**: Quick, temporary storage
- **Lifespan**: 7 days
- **Mechanism**: Access tracking + configurable promotion
- **Use case**: Latest conversation context
- **Config**: `STM_PROMOTION_STRATEGY` (count/spacing/hybrid)

#### 2. **Long-Term Memory (LTM)** - PostgreSQL + Qdrant
- **Purpose**: Permanent knowledge base
- **Lifespan**: Forever (unless explicitly deleted)
- **Storage**: 
  - **PostgreSQL**: Metadata + RLS policies + audit trail
  - **Qdrant**: Vector embeddings for similarity search
- **Use case**: Historical patterns, learned behaviors
- **Security**: Row-Level Security (RLS) enforced

#### 3. **Self-Model** - Agent Learning
- **Purpose**: Agent learns about itself
- **Content**: Error patterns, tool preferences, performance history
- **Use case**: Meta-cognitive loop - agent improves itself over time

### Management Layers

These layer on top and organize memories into semantically meaningful groups:

- **Episodes**: Conversation sessions (timestamps, participants, summary)
- **Facts**: Verified knowledge (with contradiction checking)
- **Playbooks**: Reusable procedures (standard workflows)
- **Checkpoints**: State snapshots (execution history, recovery points)

---

## System Architecture

See the rendered diagrams:

1. **System Architecture Overview** - How components interact
2. **Memory Lifecycle** - Birth to retrieval of a memory
3. **Agent Cognitive Loop** - Planning → Execution → Evaluation → Learning
4. **Data Security** - RLS and permission model
5. **Memory Layers** - STM, Promotion, LTM, Self-Model

### Technology Stack

**Backend**:
- **Framework**: FastAPI (async Python web framework)
- **ORM**: SQLAlchemy (with PostgreSQL RLS support)
- **Vectors**: Qdrant (vector database)
- **Cache**: Redis (STM, sessions, rate limiting)
- **Testing**: pytest (642 tests, >95% coverage)

**Frontend**:
- **Framework**: React 18 + TypeScript
- **State**: TanStack Query (data fetching)
- **Styling**: Tailwind CSS

**Operations**:
- **Dev**: Docker Compose (local stack)
- **Prod**: Kubernetes (manifests in `/k8s`)
- **Monitoring**: Prometheus + Grafana

---

## Key Components

### Backend Structure

```
/backend/app/
├── api/v1/
│   ├── endpoints/         # REST routes
│   │   ├── memories.py   # Memory CRUD
│   │   ├── agents.py     # Agent operations
│   │   └── ...
│   └── router.py          # Route aggregator
│
├── services/
│   ├── memory_service.py              # Core memory ops
│   ├── short_term_memory.py           # STM Redis operations
│   ├── memory_promoter.py             # STM → LTM promotion
│   ├── agent_runner.py                # Agent execution
│   └── ... (60+ services)
│
├── models/
│   ├── memory.py          # MemoryMetadata table
│   ├── episode.py         # Episode grouping
│   ├── fact.py            # Fact verification
│   ├── playbook.py        # Procedures
│   └── ... (20+ models)
│
├── agents/
│   ├── base.py            # Agent interface
│   ├── classification_agent.py
│   ├── promotion_agent.py          # STM → LTM decision
│   └── ... (10+ agent types)
│
├── core/
│   ├── config.py          # Settings (STM strategies!)
│   ├── database.py        # PostgreSQL connection
│   ├── redis.py           # Redis client
│   └── rls_guard.py       # Row-Level Security
│
└── tests/
    ├── test_memory_*.py    # Memory tests
    ├── test_agents_*.py    # Agent tests
    └── ... (642 tests)
```

### Directory Navigation

| Task | Location |
|------|----------|
| Add new memory type | `/backend/app/models/` |
| New API endpoint | `/backend/app/api/v1/endpoints/` |
| New business logic | `/backend/app/services/` |
| New agent type | `/backend/app/agents/` |
| Write tests | `/backend/tests/` |
| Fix security issue | `/backend/app/core/rls_guard.py` |
| Configuration | `/backend/app/core/config.py` |

---

## Data Flow

### Memory Creation → Retrieval

```
1. USER CREATES MEMORY
   client.memories.create(content="...", tags=[...])
   ↓
2. VALIDATION & PERMISSION CHECK
   PermissionChecker validates org/scope/clearance
   AuditService logs creation
   ↓
3. SHORT-TERM STORAGE
   StoreInRedis (7-day TTL)
   Track access count & timestamps (for promotion)
   ↓
4. PERIODIC PROMOTION CHECK
   Every hour: Check if eligible for LTM
   Decision based on: count/spacing/hybrid strategy
   ↓
5. IF ELIGIBLE: PROMOTION PIPELINE
   a. Extract metadata (tags, entities, summary)
   b. Generate embeddings (semantic meaning)
   c. Add to review queue (knowledge governance)
   ↓
6. LONG-TERM STORAGE
   PostgreSQL: Store metadata + RLS policy
   Qdrant: Store embeddings (for vector search)
   PostgreSQL FTS: Build keyword index
   ↓
7. USER SEARCHES
   search_memories(query="...", tags=[...])
   ↓
8. HYBRID SEARCH
   Option A: Vector search → Qdrant KNN
   Option B: Keyword search → PostgreSQL FTS
   Result: Combined ranking score
   ↓
9. SECURITY FILTER
   Re-lookup each result in PostgreSQL
   Apply RLS (Row-Level Security)
   Only return rows user can access
   ↓
10. RETURN TO USER
   Results with explanation:
   - Why this result matched
   - Confidence score
   - Access granted by (scope)
```

### Agent Cognitive Loop

```
1. PLANNING
   ├─ Retrieve relevant LTM memories
   ├─ Analyze similar episodes
   └─ Planner Agent creates execution plan
   
2. EXECUTION
   ├─ Executor Agent runs planned steps
   ├─ Tool calls execute with capability tokens
   └─ Store episode events in memory
   
3. EVALUATION
   ├─ Critic Agent reviews results
   ├─ Detect quality issues (drift, errors)
   └─ Generate feedback (positive/negative)
   
4. LEARNING (Meta-Cognitive Loop)
   ├─ Meta-Agent analyzes errors
   ├─ Update self-model (what I'm good at)
   └─ Adapt strategy for next run
   
5. MEMORY UPDATES
   ├─ Create memory from execution
   ├─ Add feedback signals
   ├─ Update episode summary
   └─ Log checkpoint
```

---

## Contributing Patterns

### Pattern 1: Add a New Service

**Example**: You want to optimize memory compression

```python
# backend/app/services/memory_compression.py

class MemoryCompressionService:
    """Compress old memories while preserving key information."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.qdrant = QdrantService()  # Vector DB
        self.audit = AuditService(session)
    
    async def compress_old_memories(
        self,
        org_id: str,
        days_old: int = 90,
    ) -> dict:
        """Archive old memories that are still useful."""
        # 1. Query: old memories
        # 2. Group: similar vectors together
        # 3. Summarize: create abstract version
        # 4. Store: new compressed memory
        # 5. Log: audit event
        
        self.audit.log_compression(...)
        return {"compressed": count, "saved_bytes": size_reduced}
```

**Then add tests**:
```python
# backend/tests/test_memory_compression.py

@pytest.mark.asyncio
async def test_compress_similar_memories():
    service = MemoryCompressionService(session)
    result = await service.compress_old_memories(
        org_id="test-org",
        days_old=90
    )
    assert result["compressed"] > 0
```

### Pattern 2: Add a Feature Flag

**Example**: New experimental retrieval algorithm

```python
# backend/app/core/config.py

class Settings:
    # New feature flag
    EXPERIMENTAL_RETRIEVAL_V2: bool = Field(
        default=False,
        description="Use new retrieval algorithm (experimental)"
    )

# backend/app/services/retrieval_service.py

class RetrievalService:
    async def search(self, query: str) -> List[MemoryMetadata]:
        if settings.EXPERIMENTAL_RETRIEVAL_V2:
            return await self._search_v2(query)
        else:
            return await self._search_v1(query)
```

### Pattern 3: Add RLS-Safe Query

**Example**: Find memories by tag

```python
# backend/app/services/memory_service.py

async def search_by_tag(self, tag: str) -> List[MemoryMetadata]:
    """Find memories with tag (RLS-safe)."""
    # ✅ Good: RLS filter + explicit column check
    stmt = select(MemoryMetadata).where(
        and_(
            MemoryMetadata.organization_id == self.org_id,
            MemoryMetadata.tags.contains([tag])
        )
    )
    result = await self.session.execute(stmt)
    return result.scalars().all()

    # ❌ Bad: Forgets RLS filter
    # stmt = select(MemoryMetadata).where(
    #     MemoryMetadata.tags.contains([tag])
    # )
```

### Pattern 4: Add an Agent

**Example**: Create a "Topic Analyzer" agent

```python
# backend/app/agents/topic_analyzer_agent.py

from app.agents.base import BaseAgent

class TopicAnalyzerAgent(BaseAgent):
    """Analyze conversation topics and extract themes."""
    
    name = "topic_analyzer"
    description = "Identifies topics and themes in memories"
    
    async def run(self, memory_content: str) -> dict:
        """Analyze topics in memory."""
        prompt = f"""Analyze this memory and extract topics:
        
{memory_content}

Return JSON with: {{"topics": [...], "confidence": 0.0-1.0}}"""
        
        result = await self.llm.generate(prompt)
        return json.loads(result)

# Register in agent registry
# backend/app/agents/registry.py
AGENTS = {
    "topic_analyzer": TopicAnalyzerAgent,
}
```

---

## Common Tasks

### Task 1: Add a Memory Search Filter

**Goal**: Let users search by sentiment

**Steps**:

1. Add schema field:
```python
# backend/app/schemas/memory.py
class MemorySearchRequest(BaseModel):
    sentiment: Optional[str] = None  # positive, negative, neutral
```

2. Update model:
```python
# backend/app/models/memory.py
class MemoryMetadata(Base):
    sentiment: Mapped[Optional[str]] = mapped_column(String(50))
```

3. Update service:
```python
# backend/app/services/memory_service.py
async def search(self, request: MemorySearchRequest):
    stmt = select(MemoryMetadata).where(
        and_(
            MemoryMetadata.organization_id == self.org_id,
            MemoryMetadata.sentiment == request.sentiment  # NEW
        )
    )
    return await self.session.execute(stmt)
```

4. Add test:
```python
# backend/tests/test_memory_search.py
async def test_search_by_sentiment():
    memory = await service.search_by_sentiment("positive")
    assert all(m.sentiment == "positive" for m in memory)
```

### Task 2: Improve Short-Term Memory Promotion

**Goal**: Add machine learning to decide STM → LTM promotion

**Current logic**: `count >= 3 AND spacing >= 6 hours`

**New logic**: Learn from feedback if promotion decision was right

**Files to modify**:
- `/backend/app/services/memory_promoter.py` - Decision logic
- `/backend/app/services/short_term_memory.py` - Add tracking
- `/backend/app/agents/promotion_agent.py` - ML-based scoring
- `/backend/tests/test_stm_promotion_strategies.py` - Test cases

### Task 3: Add Monitoring Dashboard

**Goal**: Show STM promotion rates, LTM growth

**Files**:
- Create `/backend/app/services/metrics_service.py`
- Add Prometheus metrics in `/backend/app/core/metrics.py`
- Create Grafana dashboard in `/observability/grafana/dashboards/`

### Task 4: Improve Documentation

**Goal**: Add detailed examples or architecture diagrams

**Files**:
- Add to `/docs/ARCHITECTURE.md` sections
- Create `/docs/TUTORIAL_*.md` for specific features
- Update `/README.md` with examples

---

## Running Tests

### All Tests
```bash
pytest tests/ -v
```

### Specific Test File
```bash
pytest tests/test_memory_service.py -v
```

### With Coverage
```bash
pytest tests/ --cov=app --cov-report=html
open htmlcov/index.html
```

### Run Only STM Tests
```bash
pytest tests/ -k stm -v
```

### Run Only Security Tests
```bash
pytest tests/ -k "rls or permission" -v
```

---

## Code Style & Standards

- **Linting**: `flake8` (PEP 8)
- **Type hints**: Required (mypy)
- **Tests**: Required for all features
- **Documentation**: Docstrings on all functions
- **Commits**: Clear commit messages

**Pre-commit check**:
```bash
flake8 app/
mypy app/ --ignore-missing-imports
pytest tests/ --cov=app -q
```

---

## Debugging Tips

### Check Memory Promotion Status
```python
# In Python REPL
from app.services.short_term_memory import ShortTermMemoryService

stm_service = ShortTermMemoryService(redis_client)
status = await stm_service.get_status(memory_id)
print(status)
# {
#   "access_count": 3,
#   "last_accessed": "...",
#   "eligible_for_promotion": True,
#   "strategy": "spacing",
#   "spacing_hours": 6.0
# }
```

### Check RLS Filtering
```python
# Test RLS in action
from app.core.rls_guard import RLSGuard

guard = RLSGuard(session)
can_access = await guard.check_access(
    user_id="user-123",
    memory_id="mem-456",
    required_scope="team"
)
```

### View Audit Trail
```sql
-- In PostgreSQL
SELECT * FROM audit_events 
WHERE resource_id = 'mem-456' 
ORDER BY created_at DESC;
```

---

## Getting Help

- **Questions**: Open a Discussion on GitHub
- **Bugs**: Open an Issue with reproduction steps
- **Features**: Open a Feature Request issue
- **Security**: Email security@sansten.ai (DO NOT open public issue)

---

## What's Next?

1. **Read** [CONTRIBUTING.md](../CONTRIBUTING.md) for process
2. **Pick an issue** with `good-first-issue` label
3. **Follow** pattern matching your task type
4. **Write tests** for everything
5. **Open a PR** with clear description
6. **Iterate** on feedback

Welcome to building the future of agent memory! 🧠
