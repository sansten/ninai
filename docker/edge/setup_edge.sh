#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f .env.edge ]]; then
  cp .env.edge.example .env.edge
  echo "Created .env.edge from template."
fi

echo "Starting Ninai edge stack..."
docker compose -f docker-compose.edge.yml --env-file .env.edge up -d

echo "Waiting for Ollama..."
until curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; do
  sleep 2
done

echo "Ensuring default model is present (qwen2.5:7b)..."
docker exec -i ninai-edge-ollama ollama pull qwen2.5:7b >/dev/null || true

echo "Edge deployment ready."
echo "API:    http://localhost:8000"
echo "Health: http://localhost:8000/health"
