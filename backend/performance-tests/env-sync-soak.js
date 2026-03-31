/**
 * Environment Sync Soak + Burst Test
 *
 * Goals:
 * - Simulate sustained inbound connector traffic over long windows
 * - Add periodic burst spikes to stress replay/out-of-order guards
 * - Measure latency/error profiles for inbound sync ingestion paths
 *
 * Typical use:
 *   k6 run env-sync-soak.js \
 *     -e BASE_URL=http://localhost:8000 \
 *     -e AUTH_TOKEN=your-token \
 *     -e ORG_ID=00000000-0000-0000-0000-000000e2e001
 *
 * 24h soak profile:
 *   k6 run env-sync-soak.js -e SOAK_HOURS=24
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Counter } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const AUTH_TOKEN = __ENV.AUTH_TOKEN || 'test-token';
const ORG_ID = __ENV.ORG_ID || '00000000-0000-0000-0000-000000e2e001';
const SOAK_HOURS = Number(__ENV.SOAK_HOURS || '1');

const inboundLatency = new Trend('env_sync_inbound_latency_ms');
const burstLatency = new Trend('env_sync_burst_latency_ms');
const ingestErrors = new Counter('env_sync_ingest_errors');

const soakDuration = `${Math.max(1, SOAK_HOURS)}h`;

export const options = {
  scenarios: {
    sustained_soak: {
      executor: 'constant-arrival-rate',
      rate: 15,
      timeUnit: '1s',
      duration: soakDuration,
      preAllocatedVUs: 25,
      maxVUs: 100,
      exec: 'sustainedSoak',
    },
    burst_spikes: {
      executor: 'ramping-arrival-rate',
      startRate: 5,
      timeUnit: '1s',
      preAllocatedVUs: 20,
      maxVUs: 120,
      exec: 'burstSpike',
      stages: [
        { target: 5, duration: '5m' },
        { target: 45, duration: '30s' },
        { target: 5, duration: '4m30s' },
      ],
    },
  },
  thresholds: {
    http_req_failed: ['rate<0.05'],
    http_req_duration: ['p(95)<1200', 'p(99)<2000'],
    env_sync_inbound_latency_ms: ['p(95)<1000'],
    env_sync_burst_latency_ms: ['p(95)<1500'],
    env_sync_ingest_errors: ['count<100'],
  },
};

function buildHeaders() {
  return {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${AUTH_TOKEN}`,
    'X-Org-ID': ORG_ID,
  };
}

function buildPayload(prefix) {
  const nowIso = new Date().toISOString();
  const rand = Math.floor(Math.random() * 1000000);
  return {
    event_id: `${prefix}-${__VU}-${__ITER}-${rand}`,
    id: `obj-${Math.floor(rand % 2500)}`,
    timestamp: nowIso,
    title: `env sync event ${prefix}`,
    summary: `sync heartbeat ${nowIso}`,
    severity: rand % 10 === 0 ? 'high' : 'low',
    source: 'k6-soak',
  };
}

function postInbound(payload, metric) {
  const url = `${BASE_URL}/api/v1/connectors/inbound/webhook`;
  const res = http.post(url, JSON.stringify(payload), { headers: buildHeaders() });
  metric.add(res.timings.duration);

  const ok = check(res, {
    'status is 202 or 200': (r) => r.status === 202 || r.status === 200,
  });

  if (!ok) {
    ingestErrors.add(1);
  }

  sleep(Math.random() * 0.4);
}

export function sustainedSoak() {
  postInbound(buildPayload('soak'), inboundLatency);
}

export function burstSpike() {
  postInbound(buildPayload('burst'), burstLatency);
}
