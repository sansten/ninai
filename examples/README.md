# Ninai2 Examples

This directory contains practical examples for using Ninai2 in various scenarios.

## Examples

### [01-session-basics.py](01-session-basics.py)
Learn the fundamentals of creating and managing Ninai2 sessions, including authentication, context management, and basic agent interaction.

**Key concepts**: Session tokens, request context, user/org scoping

### [02-multi-org-setup.py](02-multi-org-setup.py)
Set up multiple organizations with separate admin users and data isolation. Demonstrates multi-tenancy and role-based access control.

**Key concepts**: Organization creation, admin users, tenant isolation, RLS

### [03-memory-workflow.py](03-memory-workflow.py)
Create and manage agent memories across sessions, including memory tagging, retrieval, and lifecycle management.

**Key concepts**: Memory persistence, tagging/retrieval, working memory, eviction

### [04-custom-agent.py](04-custom-agent.py)
Build a custom agent with tailored instructions, function binding, and LLM router configuration.

**Key concepts**: Agent manifests, function schemas, multi-provider LLM routing

### [05-batch-operations.py](05-batch-operations.py)
Perform bulk operations: batch memory creation, updates, and async task scheduling via Celery.

**Key concepts**: Batch APIs, async tasks, error handling, transaction safety

### [06-security-audit.py](06-security-audit.py)
Query audit logs, verify compliance, export user data, and request full tenant deletion (GDPR).

**Key concepts**: Audit logs, data export, right-to-erasure, compliance

### [07-observability.py](07-observability.py)
Set up logging, metrics, and distributed tracing for production deployments. Integrates with Prometheus, Loki, and Jaeger.

**Key concepts**: Observability, distributed tracing, custom metrics

## Running Examples

All examples assume a running Ninai2 backend. Set up via Docker:

```bash
cd repos/ninai/backend
docker-compose -f docker-compose.yml up -d

# Wait for services to be ready
sleep 30
```

Then run any example:

```bash
cd examples
python 01-session-basics.py
```

## Environment Setup

Create `.env` in the `examples/` directory:

```bash
# Backend API
NINAI_BASE_URL=http://localhost:8000
NINAI_API_KEY=your-org-api-key

# Optional: Specify org/user (defaults read from API)
NINAI_ORG_ID=org-uuid
NINAI_USER_ID=user-uuid
```

## Error Handling

All examples include proper error handling patterns:
- **HTTP errors**: 4xx (client), 5xx (server)
- **Validation errors**: Check `error.detail` for specific constraint violations
- **Auth errors**: Refresh token or re-authenticate
- **Async errors**: Check Celery task status via `GET /tasks/{task_id}`

See [error-handling.md](error-handling.md) for patterns.

## Best Practices

### Connection Management
```python
# Create reusable session with connection pooling
async with aiohttp.ClientSession() as session:
    client = NinaiClient(session, api_key=key)
    # All requests share connection pool
```

### Retry Logic
```python
# Exponential backoff for transient failures
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def robust_api_call():
    ...
```

### Rate Limiting
```python
# Respect rate limits (100/min per org)
import asyncio
asyncio.Semaphore(10)  # Max 10 concurrent requests
```

## Contributing

Have a useful example? Submit a PR! Guidelines:
- Include docstring and comments
- Add error handling
- Test with both local and Docker setup
- Document required environment variables

## Troubleshooting

**Connection refused**: Ensure backend is running (`docker-compose ps`)

**Auth token expired**: Refresh via `/auth/refresh` endpoint

**Rate limited (429)**: Add exponential backoff between retries

**Memory not found**: Check org_id scope; memories are org-scoped

For more help, see [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md)
