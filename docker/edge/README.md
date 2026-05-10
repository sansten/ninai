# Ninai Edge Stack (From Scratch)

This profile brings up a reproducible local-first stack with:
- PostgreSQL + pgvector
- Qdrant
- Redis (cache/broker)
- FalkorDB (graph)
- Ollama
- Ninai API

## One-time setup

```bash
cd repos/ninai/docker/edge
cp .env.edge.example .env.edge
```

## Fresh start (recommended when changing infra config)

```bash
cd repos/ninai/docker/edge
docker compose -f docker-compose.edge.yml --env-file .env.edge down -v
docker compose -f docker-compose.edge.yml --env-file .env.edge up -d
```

Or use the helper script:

```bash
cd repos/ninai/docker/edge
./setup_edge.sh
```

## Health checks

```bash
curl -f http://localhost:8000/health
curl -f http://localhost:6333/healthz
docker exec ninai-edge-redis redis-cli ping
docker exec ninai-edge-falkordb redis-cli ping
docker exec ninai-edge-postgres pg_isready -U ninai -d ninai
```

## Tuned defaults

- DB pool baseline in `.env.edge.example`:
  - `DB_POOL_SIZE=10`
  - `DB_MAX_OVERFLOW=10`
  - `DB_POOL_TIMEOUT_SECONDS=30`
  - `DB_POOL_RECYCLE_SECONDS=1800`
  - `DB_POOL_USE_LIFO=true`
- Postgres bootstrap SQL is mounted from `postgres/init/01_ninai_bootstrap.sql`
- Redis config is mounted from `redis/redis.conf`
- FalkorDB config is mounted from `falkordb/falkordb.conf`
- Qdrant config is mounted from `qdrant/config.yaml`
