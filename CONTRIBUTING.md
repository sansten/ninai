# Contributing to Ninai

Thanks for your interest in contributing to Ninai! We welcome bug fixes, feature implementations, research advancements, documentation improvements, and benchmarking work.

This document covers:
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Organization](#code-organization)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Git Workflow](#git-workflow)
- [Research Contributions](#research-contributions)
- [Code Review](#code-review)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- **Python 3.11+** (for backend)
- **Node.js 18+** (for frontend)
- **Docker & Docker Compose** (for local development stack)
- **Git** (with GitHub CLI optional but recommended)

### Quick Orientation

Before diving in, read:
1. [README.md](README.md) — Project vision and quick start
2. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System design and layers
3. [docs/multimodal_memory_architecture.md](docs/multimodal_memory_architecture.md) — Mathematical framework and research context

This will help you understand the problem we're solving and how Ninai is different from standard RAG systems.

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/ninai.git
cd ninai
git remote add upstream https://github.com/sansten/ninai.git
```

### 2. Create Virtual Environment (Backend)

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\Activate.ps1

# Install backend dependencies
pip install -e ".[dev]"
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Start Development Stack

```bash
# All services (Postgres, Qdrant, Redis, Backend, Frontend)
docker compose up -d --build

# Or minimal (just Postgres + Qdrant)
docker compose -f docker-compose.lite.yml up -d --build

# Check logs
docker compose logs -f backend
docker compose logs -f postgres
```

### 5. Verify Setup

```bash
# Backend API health check
curl http://localhost:8000/health

# Frontend
open http://localhost:3000

# Run tests
python -m pytest -xvs backend/tests/test_health.py
```

---

## Code Organization

```
backend/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── memories/          # Memory CRUD endpoints
│   │   │   ├── topics/            # Topic management
│   │   │   ├── retrieval/         # Search + scoring
│   │   │   ├── approval/          # Knowledge review workflows
│   │   │   ├── health/            # Health checks + metrics
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── services/
│   │   ├── retrieval_service.py    # GAP-3: Two-stage retrieval
│   │   ├── hierarchy_service.py    # GAP-1: Episode/semantic node lifecycle
│   │   ├── topic_structure_service.py  # GAP-2: Sparsity-semantics guidance
│   │   ├── knn_navigation_service.py   # GAP-6: Navigation graph maintenance
│   │   ├── uncertainty_gating_service.py  # GAP-4: Entropy-based expansion
│   │   ├── activation_scoring_service.py  # 8-component activation model
│   │   ├── knowledge_approval_service.py  # Review queue + workflow
│   │   └── ...
│   ├── models/
│   │   ├── memory.py              # Core memory models
│   │   ├── memory_episode.py      # GAP-1: Episode model
│   │   ├── memory_semantic_node.py # GAP-1: Semantic node model
│   │   ├── navigation_edge.py     # GAP-6: Navigation graph edges
│   │   ├── topic.py               # Topic organization
│   │   ├── approval.py            # Review workflow models
│   │   └── ...
│   ├── middleware/
│   │   ├── auth.py                # JWT + OIDC authentication
│   │   ├── tenant_context.py      # RLS org_id injection
│   │   └── audit.py               # Audit logging middleware
│   ├── core/
│   │   ├── config.py              # Settings + env vars
│   │   ├── database.py            # Postgres session factory
│   │   ├── qdrant_client.py       # Vector DB wrapper
│   │   └── security.py            # RLS policy helpers
│   └── main.py
├── alembic/
│   ├── versions/                  # Schema migrations
│   ├── env.py
│   └── script.py.mako
├── tests/
│   ├── unit/                      # Fast tests (no DB)
│   ├── integration/               # Tests requiring Postgres
│   ├── e2e/                       # End-to-end flow tests
│   ├── conftest.py                # Pytest fixtures
│   └── ...
└── scripts/
    ├── seed_data.py               # Create demo org + users
    ├── reset_db.py                # Drop and recreate schema
    └── ...

frontend/
├── src/
│   ├── components/
│   │   ├── Search/                # Query + retrieval UI
│   │   ├── ReviewQueue/           # Knowledge approval dashboard
│   │   ├── MemoryForm/            # Submit new memory
│   │   ├── ScoreBreakdown/        # Display 8-component scores
│   │   └── ...
│   ├── pages/
│   │   ├── Dashboard.tsx          # Main dashboard
│   │   ├── Memory/                # Memory detail view
│   │   ├── Admin/                 # Admin panel
│   │   └── ...
│   ├── hooks/
│   │   ├── useAuth.ts             # Auth context + helpers
│   │   ├── useMemory.ts           # Memory API calls
│   │   └── ...
│   ├── services/
│   │   └── api.ts                 # REST client wrapper
│   └── types/
│       ├── index.ts               # Shared TypeScript types
│       └── ...
└── package.json
```

---

## Making Changes

### Branch Naming

Use **Conventional Commits** style:

```bash
git checkout -b feature/adaptive-alpha-tuning      # Feature
git checkout -b fix/cross-tenant-isolation-bug     # Bug fix
git checkout -b docs/improve-setup-guide           # Documentation
git checkout -b test/add-uncertainty-gating-tests  # Tests
git checkout -b research/benchmark-locomo          # Research work
```

### Code Style

**Python:**
- `black` for formatting (line length: 100)
- `isort` for imports
- `ruff` for linting
- Type hints required for all function signatures

```bash
black backend/
isort backend/
ruff check backend/ --fix
```

**TypeScript/React:**
- `eslint` for linting
- `prettier` for formatting
- `strict: true` in tsconfig.json

```bash
cd frontend
npm run lint
npm run lint -- --fix
npm run format
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(retrieval): Implement adaptive alpha tuning for coverage-relevance trade-off

- Add AlphaService for learning α from feedback
- Integrate with RetrievalService.retrieve()
- Add A/B test harness for token savings measurement
- Update documentation with tuning guidelines

Closes #42
Refs: GAP-3, Eq. (7)
```

**Format:**
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat` — New feature or capability
- `fix` — Bug fix
- `refactor` — Code reorganization (no behavior change)
- `perf` — Performance improvement
- `test` — Test additions or fixes
- `docs` — Documentation
- `chore` — Build, CI, dependencies
- `research` — Research work, benchmarking

**Scopes** (examples):
- `retrieval` — RetrievalService changes
- `hierarchy` — HierarchyService, episode/semantic changes
- `topic-structure` — Topic rebalancing, sparsity-semantics
- `activation` — 8-component scoring
- `security` — RLS, auth, multi-tenancy
- `frontend` — React components, UI

---

## Testing

### Running Tests

```bash
# All tests (fast unit tests first)
python -m pytest backend/tests -v

# Only fast tests (no DB)
python -m pytest backend/tests/unit -v

# Integration tests (requires Postgres running)
docker compose up -d postgres
RUN_POSTGRES_TESTS=1 python -m pytest backend/tests/integration -v

# Specific test
python -m pytest backend/tests/integration/test_hierarchy.py -v

# With coverage
python -m pytest backend/tests --cov=app --cov-report=html

# Cross-tenant isolation tests
RUN_POSTGRES_TESTS=1 python -m pytest -k "cross_tenant" -v
```

### Writing Tests

**Unit Test (no database):**
```python
# backend/tests/unit/test_activation_scoring.py
import pytest
from app.services.activation_scoring_service import ActivationScoringService

@pytest.fixture
def scoring_svc():
    return ActivationScoringService()

def test_relevance_component(scoring_svc):
    """Test relevance score calculation."""
    score = scoring_svc.compute_relevance(
        query_embedding=[0.1, 0.2, ...],
        memory_embedding=[0.15, 0.18, ...]
    )
    assert 0 <= score <= 1
    assert score > 0.9  # High similarity
```

**Integration Test (with Postgres):**
```python
# backend/tests/integration/test_retrieval.py
import pytest

@pytest.mark.asyncio
async def test_retrieve_with_gating(db_session, org_id):
    """Test entropy-gated retrieval (GAP-4)."""
    retrieval_svc = RetrievalService(db_session, ollama_client)
    
    result = await retrieval_svc.retrieve_with_gating(
        query="Deploy to production",
        organization_id=org_id,
        entropy_threshold=0.1
    )
    
    assert result.representatives
    assert len(result.final_context) < 2000  # Token budget
    assert result.total_entropy_reduction > 0.5
```

**Frontend Test:**
```typescript
// frontend/src/components/Search/Search.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { Search } from './Search';

describe('Search Component', () => {
  it('should display score breakdown for each result', async () => {
    const { getByText } = render(
      <Search 
        onResultsLoaded={jest.fn()} 
      />
    );
    
    expect(screen.getByText('Relevance:')).toBeInTheDocument();
    expect(screen.getByText('Recency:')).toBeInTheDocument();
  });
});
```

### Test Requirements for PRs

- **Feature PRs:** Add tests covering happy path + edge cases
- **Bug fix PRs:** Include regression test
- **Research work:** Document experimental setup and benchmarking methodology
- **Minimum coverage:** 80% for modified files
- **All existing tests must pass**

---

## Git Workflow

### 1. Sync with upstream

```bash
git fetch upstream
git rebase upstream/main
```

### 2. Make changes

```bash
git add <files>
git commit -m "feat(scope): Description"
```

### 3. Keep branch updated

```bash
git fetch upstream
git rebase upstream/main
# If conflicts: resolve, then `git rebase --continue`
```

### 4. Push to your fork

```bash
git push origin feature/your-change
```

### 5. Open PR on GitHub

- **Title:** Follow Conventional Commits
- **Description:** Explain WHAT and WHY
- **Reference issues:** "Closes #42" or "Refs #42"
- **Link docs:** If user-facing, link to [docs/](docs/) updates

**PR Template Example:**
```
## What
Two-stage adaptive retrieval with entropy-gated expansion (GAP-3, GAP-4).

## Why
- Current flat retrieval collapses into dense regions
- Entropy-gating ensures only uncertainty-reducing evidence is included
- Saves ~40% tokens on LoCoMo benchmark (preliminary)

## Changes
- `RetrievalService.retrieve_with_gating()` with Stage I submodular greedy and Stage II entropy control
- `UncertaintyGatingService` for entropy calculation via Ollama
- 14 new integration tests
- Updated docs/RETRIEVAL.md

## Validation
- All tests passing (integration + unit)
- Manual A/B test on 100-query sample: 38% token savings, 2% quality drop
- Cross-tenant isolation tests still passing

Closes #42
```

---

## Research Contributions

If you're contributing research work (benchmarking, new algorithms, performance studies), please:

### 1. Document the Work

Create a RFC (Request for Comments) document or open a discussion:
- **Problem:** What gap are you addressing?
- **Approach:** Methods, baselines, metrics
- **Scope:** Time estimate, dependencies
- **Expected impact:** Token savings? Accuracy improvement? Latency reduction?

Examples:
- [docs/multimodal_memory_architecture.md](docs/multimodal_memory_architecture.md) — Research framework
- [GitHub Discussions](https://github.com/sansten/ninai/discussions) — Open ideas for feedback

### 2. Benchmarking

If running benchmarks:

```bash
# Create isolated branch
git checkout -b research/benchmark-locomo

# Log results with experiment metadata
python scripts/run_benchmark.py \
  --dataset LoCoMo \
  --method entropy_gating \
  --model text-embedding-3-small \
  --output results/locomo_entropy_gating_2026-02-22.json
```

**Report format:**
```json
{
  "experiment": "Entropy Gating on LoCoMo",
  "date": "2026-02-22",
  "baseline": "flat_retrieval",
  "method": "entropy_gating",
  "hyperparams": {
    "entropy_threshold": 0.1,
    "max_expansion_items": 10
  },
  "metrics": {
    "bleu": 0.58,
    "f1": 0.72,
    "rouge_l": 0.65,
    "tokens_per_query": 1240,
    "latency_p95_ms": 320
  },
  "config": {
    "dataset": "LoCoMo",
    "model": "text-embedding-3-small",
    "num_queries": 500,
    "ollama_endpoint": "http://localhost:11434"
  }
}
```

### 3. Publish Results

Share findings:
- Create **GitHub Discussions** thread with summary
- Submit **Pull Request** with benchmarking code and results
- Consider writing a **blog post** or research note

---

## Code Review

### For Reviewers

- Check alignment with [docs/multimodal_memory_architecture.md](docs/multimodal_memory_architecture.md)
- Verify cross-tenant isolation isn't broken
- Confirm all tests pass
- Look for performance regressions
- Request updates if needed

### For Authors

- Respond to feedback within 48 hours
- Request re-review after updates
- Keep PRs focused (<300 lines of code changes when possible)
- Link to related PRs or issues

### Merge Criteria

✅ All tests passing  
✅ Code review approval from maintainer  
✅ No merge conflicts  
✅ Documentation updated  
✅ Commit messages clean  

---

## Troubleshooting

### Database Issues

**Reset schema:**
```bash
docker compose down postgres
docker volume rm ninai_postgres_data
docker compose up -d postgres
python scripts/seed_data.py
```

**Check Postgres logs:**
```bash
docker compose logs -f postgres
psql postgresql://ninai:password@localhost:5432/ninai
```

### Qdrant Issues

**Verify running:**
```bash
curl http://localhost:6333/health
```

**Rebuild index:**
```bash
# Via Python
from app.services.qdrant_service import QdrantService
await qdrant_svc.rebuild_collection("memories")
```

### Frontend Build Issues

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run build
```

### Tests Timing Out

Increase timeout:
```bash
# For Postgres tests
RUN_POSTGRES_TESTS=1 pytest --timeout=30 backend/tests/
```

### Vector Similarity Search Returning Nothing

- Check embedding dimension (should be 1536 for OpenAI, 768 for Ollama)
- Verify Qdrant payload includes `organization_id`
- Ensure RLS org_id context is set before search

```python
# Debug
from app.core.database import SessionLocal
async with SessionLocal() as session:
    result = await qdrant_client.search(
        collection_name="memories",
        query_vector=[...],
        limit=10,
        query_filter=Filter(must=[
            FieldCondition(key="organization_id", match=MatchValue(value=org_id))
        ])
    )
    print(f"Found {len(result)} results")
```

---

## Getting Help

- **Questions?** → [GitHub Discussions](https://github.com/sansten/ninai/discussions)
- **Found a bug?** → [Open an Issue](https://github.com/sansten/ninai/issues)
- **Want to discuss design?** → [Create a Discussion](https://github.com/sansten/ninai/discussions)
- **Email:** opensource@sansten.com

---

## Code of Conduct

We're committed to a welcoming and inclusive community. Please review [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

---

Happy contributing! 🚀
