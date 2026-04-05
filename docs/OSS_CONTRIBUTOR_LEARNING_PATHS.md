# Ninai OSS - Contributor Learning Resources

**For new open-source contributors who want to understand and contribute to Ninai Cognitive OS**

---

## 📚 Quick Links

| Resource | Purpose | Best For |
|----------|---------|----------|
| [CONTRIBUTOR_ARCHITECTURE_GUIDE.md](./CONTRIBUTOR_ARCHITECTURE_GUIDE.md) | Complete architecture & patterns | Deep understanding |
| [ARCHITECTURE_VISUAL_SUMMARY.md](./ARCHITECTURE_VISUAL_SUMMARY.md) | High-level system overview | Big picture |
| [README_NINAI.md](../README.md) | Project introduction | Getting started |
| [QUICKSTART.md](../QUICKSTART.md) | Local setup instructions | First-time setup |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution process | PR submission |

---

## 🎯 Learning Path for New Contributors

### Level 1: Understand the Problem (15 min)
**Goal**: Why does Ninai exist?

**Read**: 
- Project README (why agent memory is different from document RAG)
- "The Problem We Solved" section in ARCHITECTURE_VISUAL_SUMMARY.md

**Watch diagrams**:
1. **System Architecture Overview** - How components connect
2. **Memory Layers Diagram** - Three layers: STM, LTM, Self-Model

**Stop and ask**: "Why is STM separate from LTM?"

---

### Level 2: Understand the Flow (20 min)
**Goal**: How does a memory move through the system?

**Read**: 
- CONTRIBUTOR_ARCHITECTURE_GUIDE.md: "Data Flow" section
- CONTRIBUTOR_ARCHITECTURE_GUIDE.md: "Core Concepts"

**Watch diagrams**:
1. **Memory Lifecycle** - Birth to retrieval (5 stages)
2. **Memory Layers** - How STM promotes to LTM
3. **Data Security** - RLS filtering

**Key insight**: Memory has a **promotion journey**
- Starts in Redis (STM) → tracked for promotion eligibility
- Moves to PostgreSQL + Qdrant (LTM) when criteria met
- Retrieved safely with Row-Level Security (RLS)

---

### Level 3: Understand the Codebase (30 min)
**Goal**: Where is everything in the code?

**Read**:
- CONTRIBUTOR_ARCHITECTURE_GUIDE.md: "Key Components" & "Backend Structure"
- Browse `/backend/app/` directory structure

**Watch diagram**:
1. **Codebase Structure** - Where to find things

**Hands-on**:
```bash
# Clone and explore
git clone https://github.com/sansten/ninai.git
cd ninai
find . -name "memory_service.py"      # Core service
find . -name "short_term_memory.py"   # STM handling
find . -name "rls_guard.py"           # Security
```

---

### Level 4: Understand Agent Cognition (25 min)
**Goal**: How do agents use memory?

**Read**:
- CONTRIBUTOR_ARCHITECTURE_GUIDE.md: "Agent Cognitive Loop"

**Watch diagram**:
1. **Agent Framework - Cognitive Loop** - Plan → Execute → Evaluate → Learn

**Key insight**: Agents are **self-improving**
- Retrieve memories → plan → execute with tools → evaluate results
- Feedback improves self-model → next run is better

---

### Level 5: Your First Contribution (Varies)
**Goal**: Make a change to the codebase

**Choose a pattern**:
- **New feature**: Add feature flag (easiest)
- **Improve existing**: Add test coverage (medium)
- **New capability**: Add a service (harder)

**Read**:
- CONTRIBUTOR_ARCHITECTURE_GUIDE.md: "Contributing Patterns"
- Pick your pattern (Pattern 1-4)

**Follow the example code** provided

**Run tests**:
```bash
pytest tests/ -v
```

---

## 🔬 Deep Dive Topics

### The Memory Promotion Decision (Advanced)

**Problem**: How do we decide STM → LTM promotion?

**Solution**: Three configurable strategies:

1. **Count Strategy**
   - Promote if `access_count >= 3`
   - Ignores timing
   - Fast but noisy

2. **Spacing Strategy** ⭐ (Default)
   - Promote if accesses are 6+ hours apart
   - Filters same-session bursts
   - Better signal

3. **Hybrid Strategy**
   - Requires BOTH count AND spacing
   - Most conservative
   - Highest quality

**Files**:
- `/backend/app/core/config.py` - Configuration
- `/backend/app/services/memory_promoter.py` - Decision logic
- `/backend/app/services/short_term_memory.py` - Tracking
- `/backend/tests/test_stm_promotion_strategies.py` - Validation

**Diagram**: See "Memory Lifecycle" - Step 3 (Promotion Pipeline)

---

### Row-Level Security (RLS) (Advanced)

**Problem**: Multi-tenant system - Company A can't see Company B's data

**Solution**: PostgreSQL RLS + Application verification layer

**How it works**:
1. User makes API call with JWT token
2. Extract org_id from token
3. In PostgreSQL: `SELECT * WHERE org_id = :org_id` enforced by policy
4. Qdrant results are re-verified in PostgreSQL (belt + suspenders)

**Files**:
- `/backend/app/core/rls_guard.py` - Permission checking
- `/backend/app/models/base.py` - RLS policies defined
- `/backend/app/services/permission_checker.py` - Capability checks

**Diagram**: See "Data Security - RLS & Permission Model"

---

### Hybrid Search (Advanced)

**Problem**: Vector search (semantic) + keyword search (exact) have trade-offs

**Solution**: Use both, combine scores

**How it works**:
1. User searches "deploy Kubernetes"
2. Vector search (Qdrant): Finds semantically similar memories
3. Keyword search (PostgreSQL FTS): Finds exact matches on "deploy"
4. Hybrid ranking: Combine scores (BM25 + vector similarity)
5. RLS filter: Verify each result is accessible
6. Return best matches with explanation

**Files**:
- `/backend/app/services/retrieval_service.py` - Search orchestration
- `/backend/app/services/search_query_parser.py` - Query parsing
- Qdrant Python SDK - Vector operations
- PostgreSQL FTS - Full-text search

---

## 🧪 Testing Philosophy

**Target**: >95% code coverage

**Types**:
- **Unit tests**: Single function/method
- **Integration tests**: Service + database
- **E2E tests**: Full API request → response
- **Security tests**: RLS, permissions, audit

**Pattern**:
```python
# Test follows: Setup → Action → Assert
async def test_memory_promotion():
    # Setup
    service = MemoryPromotionService(db)
    memory = await service.store(...)
    
    # Action
    eligible = await service.check_eligibility(...)
    
    # Assert
    assert eligible == True
```

**Run tests**:
```bash
pytest tests/                    # All
pytest tests/ -k stm -v         # STM-related
pytest tests/ --cov=app -q      # With coverage
```

---

## 🛠️ Development Workflow

### Setup Local Environment
```bash
git clone https://github.com/sansten/ninai.git
cd ninai
docker-compose up -d              # Start Postgres, Redis, Qdrant
pip install -e .                  # Install in dev mode
pytest tests/ -v                  # Run tests
python -m app.main               # Start server
```

### Make a Change
```bash
git checkout -b feature/my-feature
# Edit files
# Add tests
pytest tests/ -v                  # Verify
# Commit
git push origin feature/my-feature
# Open PR on GitHub
```

### PR Review Process
1. **Code review**: Architecture, security, style
2. **Tests required**: Must pass all tests
3. **Coverage**: Must maintain >95%
4. **Documentation**: Update docs if behavior changes

---

## 📊 Architecture Decision Records

### Why Redis for STM?
- ✅ Fast ephemeral storage
- ✅ Built-in TTL expiration
- ✅ Access tracking (counters, time lists)
- ✅ No schema needed
- ❌ Data loss on restart (acceptable for ephemeral)

### Why PostgreSQL for LTM?
- ✅ ACID compliance (data safety)
- ✅ Row-Level Security (multi-tenancy)
- ✅ Audit trail (compliance)
- ✅ Complex queries (analytics)
- ❌ Slower than Redis (acceptable for archival)

### Why Qdrant for Vectors?
- ✅ Purpose-built vector database
- ✅ Support for hybrid search
- ✅ Metadata filtering
- ✅ Production-ready reliability
- ❌ Must re-verify with PostgreSQL (security belt+suspenders)

### Why Agents (Planner, Executor, Critic)?
- ✅ Separates concerns (planning vs execution)
- ✅ Enables quality checking (critic role)
- ✅ Supports learning (feedback loop)
- ✅ Mimics human cognition (interpretable)
- ❌ Added complexity vs simple function calls

---

## 🎓 Key Terminology

| Term | Meaning | Example |
|------|---------|---------|
| **STM** | Short-Term Memory | Redis cache, 7-day TTL |
| **LTM** | Long-Term Memory | PostgreSQL + Qdrant, permanent |
| **RLS** | Row-Level Security | User sees only their org's data |
| **Episode** | Conversation session | "Support chat 2025-03-01" |
| **Fact** | Verified knowledge | "API returns HTTP 200 on success" |
| **Playbook** | Reusable procedure | "Deployment checklist" |
| **Checkpoint** | State snapshot | "Execution state at step 5" |
| **Scope** | Visibility level | personal/team/dept/org |
| **Promotion** | STM → LTM migration | When accessed enough times |
| **Capability token** | Permission proof | Agent can call Tool X |

---

## ❓ Common Questions

**Q: Do I need to understand PostgreSQL RLS to contribute?**
A: No, but read [Data Security diagram](#) and understand the principle: "User can only see their org's data"

**Q: What if I break existing tests?**
A: That's OK! It means your change has impact. Review the test to understand expected behavior, then update it.

**Q: How do I debug a promotion decision?**
A: Use the Python REPL (see "Debugging Tips" in CONTRIBUTOR_ARCHITECTURE_GUIDE.md)

**Q: Do I need to write documentation for new features?**
A: Yes, update the relevant `.md` file in `/docs/` and docstrings in code

**Q: How long does it take to get a PR merged?**
A: Typically 2-5 days depending on complexity. Maintainers review ASAP.

---

## 🚀 Next Steps

1. **Read**: CONTRIBUTOR_ARCHITECTURE_GUIDE.md (30 min)
2. **Setup**: Run `docker-compose up && pytest tests/` (15 min)
3. **Explore**: Browse the codebase, read a service file (30 min)
4. **Contribute**: Pick an issue marked `good-first-issue` (varies)
5. **Iterate**: Follow feedback on your PR (varies)

**You're now ready to contribute!** 🎉

---

**Questions?** Open a GitHub Discussion or email support@sansten.ai
