# Demo Deployment Setup

This guide describes how to run a hosted Ninai demo environment for evaluator access.

## Goals

- Deploy a read-only or low-risk demo quickly.
- Seed realistic data for walkthroughs.
- Keep production credentials out of the demo stack.

## Recommended Platforms

- Render
- Railway
- Fly.io

## Demo Topology

- Frontend service (`frontend`)
- Backend API (`backend`)
- PostgreSQL
- Redis
- Qdrant

For local/demo parity, use:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml up -d --build
```

## Environment Requirements

Set these per environment:

```bash
APP_ENV=demo
DEBUG=false
SECRET_KEY=<strong-random-value>
POSTGRES_PASSWORD=<strong-random-value>
JWT_SECRET=<strong-random-value>
```

Optional (recommended):

```bash
AUTH_MODE=both
OIDC_ISSUER=<your-idp-issuer>
OIDC_CLIENT_ID=<your-client-id>
```

## Seeding Demo Data

`docker-compose.demo.yml` includes a one-shot `seed` service:

```bash
docker compose -f docker-compose.yml -f docker-compose.demo.yml run --rm seed
```

This seeds representative data for product walkthroughs.

## Security Checklist

- Never expose development default credentials publicly.
- Restrict demo network ingress to required ports only.
- Use HTTPS at the edge (platform-managed TLS).
- Rotate secrets before each external demo cycle.
- Disable admin-only operational endpoints unless required.

## Post-Deploy Validation

```bash
curl -f https://<demo-host>/api/v1/health
```

Open:

- `https://<demo-host>/`
- `https://<demo-host>/docs`

## Notes

- Treat hosted demos as disposable environments.
- Keep data synthetic or anonymized.
- Prefer short-lived credentials and periodic resets.
