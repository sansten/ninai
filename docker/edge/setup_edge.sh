#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env.edge ]]; then
  cp .env.edge.example .env.edge
  echo "Created .env.edge from template."
fi

EDGE_MODEL="$(grep -E '^OLLAMA_MODEL=' .env.edge | tail -n 1 | cut -d'=' -f2-)"
EDGE_MODEL="${EDGE_MODEL:-qwen2.5:7b}"

echo "Starting Ninai edge stack..."
docker compose -f docker-compose.edge.yml --env-file .env.edge up -d

echo "Waiting for Postgres..."
until docker exec ninai-edge-postgres pg_isready -U ninai -d ninai >/dev/null 2>&1; do
  sleep 2
done

echo "Waiting for Redis..."
until docker exec ninai-edge-redis redis-cli ping | grep -q PONG; do
  sleep 2
done

echo "Waiting for FalkorDB..."
until docker exec ninai-edge-falkordb redis-cli ping | grep -q PONG; do
  sleep 2
done

echo "Waiting for Qdrant..."
until curl -fsS http://localhost:6333/healthz >/dev/null 2>&1; do
  sleep 2
done

echo "Waiting for Ollama..."
until curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 2
done

echo "Ensuring default model is present (${EDGE_MODEL})..."
docker exec -i ninai-edge-ollama ollama pull "${EDGE_MODEL}" >/dev/null || true

echo "Edge deployment ready."
echo "API:    http://localhost:8000"
echo "Health: http://localhost:8000/health"
