"""Async-extract pending indicator (NINAI_ASYNC_EXTRACT).

When entity extraction is deferred to a background task, read() surfaces how many
extractions are still in flight for the tenant so a caller knows the entity graph may
not be fully back-filled yet. These cover the per-tenant Redis tracking helpers and the
response-schema back-compat.
"""

from app.v2.memory import dnc_router
from app.v2.api.schemas import V2InteractResponse


class FakeRedis:
    """Minimal async stand-in for the set ops the helpers use."""

    def __init__(self):
        self.sets: dict[str, set] = {}
        self.expires: dict[str, int] = {}

    async def sadd(self, key, *vals):
        self.sets.setdefault(key, set()).update(vals)

    async def srem(self, key, *vals):
        self.sets.get(key, set()).difference_update(vals)

    async def scard(self, key):
        return len(self.sets.get(key, set()))

    async def expire(self, key, ttl):
        self.expires[key] = ttl


def _patch_redis(monkeypatch, fake):
    async def _get_redis():
        return fake
    # helpers import get_redis lazily from the module, so patch the source attribute
    monkeypatch.setattr("app.core.redis.get_redis", _get_redis)


async def test_mark_count_clear_roundtrip(monkeypatch):
    fr = FakeRedis()
    _patch_redis(monkeypatch, fr)
    t = "tenantA"

    assert await dnc_router._count_enrich_pending(t) == 0
    await dnc_router._mark_enrich_pending(t, "u1")
    await dnc_router._mark_enrich_pending(t, "u2")
    assert await dnc_router._count_enrich_pending(t) == 2

    # re-marking the same utterance is idempotent (it's a set)
    await dnc_router._mark_enrich_pending(t, "u1")
    assert await dnc_router._count_enrich_pending(t) == 2

    await dnc_router._clear_enrich_pending(t, "u1")
    assert await dnc_router._count_enrich_pending(t) == 1
    # clearing twice does not underflow
    await dnc_router._clear_enrich_pending(t, "u1")
    assert await dnc_router._count_enrich_pending(t) == 1

    # a TTL is applied so a lost task cannot pin the hint on forever
    assert fr.expires[dnc_router._enrich_pending_key(t)] == dnc_router._ENRICH_PENDING_TTL


async def test_pending_is_tenant_scoped(monkeypatch):
    fr = FakeRedis()
    _patch_redis(monkeypatch, fr)
    await dnc_router._mark_enrich_pending("A", "u1")
    assert await dnc_router._count_enrich_pending("A") == 1
    assert await dnc_router._count_enrich_pending("B") == 0


async def test_helpers_swallow_redis_errors(monkeypatch):
    async def _boom():
        raise RuntimeError("redis down")
    monkeypatch.setattr("app.core.redis.get_redis", _boom)
    # the write/read path must never break on a Redis failure
    await dnc_router._mark_enrich_pending("A", "u1")
    await dnc_router._clear_enrich_pending("A", "u1")
    assert await dnc_router._count_enrich_pending("A") == 0


async def test_status_endpoint_reports_pending(monkeypatch):
    from app.v2.api.v2_router import v2_enrichment_status
    fr = FakeRedis()
    _patch_redis(monkeypatch, fr)
    await dnc_router._mark_enrich_pending("T", "u1")
    await dnc_router._mark_enrich_pending("T", "u2")

    resp = await v2_enrichment_status(tenant="T", current_user={"org_id": "T"})
    assert resp.tenant_id == "T"
    assert resp.pending == 2
    assert resp.enrichment_pending is True


async def test_status_endpoint_zero_when_idle(monkeypatch):
    from app.v2.api.v2_router import v2_enrichment_status
    fr = FakeRedis()
    _patch_redis(monkeypatch, fr)

    resp = await v2_enrichment_status(
        tenant="idle-tenant", current_user={"org_id": "idle-tenant"}
    )
    assert resp.pending == 0
    assert resp.enrichment_pending is False


def test_response_schema_back_compat():
    # existing callers that don't pass the new fields still construct fine, defaulting
    # to "not pending"
    r = V2InteractResponse(
        response="x", session_id="s", user_utterance_id="u", assistant_utterance_id="a",
        cited_node_ids=[], extracted_entities=[], graph_nodes_retrieved=0,
        qdrant_chunks_retrieved=0, graph_writes=0, decay_stats={}, latency_ms=1, error="",
    )
    assert r.enrichment_pending is False
    assert r.pending_enrichments == 0
