/**
 * Snapshot Operations Load Test
 * 
 * Tests backup/snapshot operations under load:
 * - Create snapshot
 * - Verify snapshot
 * - List snapshots
 * 
 * Scenarios:
 * 1. Ramp-up: 0->30 VUs over 20s
 * 2. Sustained: 30 VUs for 2 minutes
 * 3. Peak: Spike to 50 VUs for validation
 * 4. Cooldown: Scale to 0
 * 
 * Metrics:
 * - Snapshot creation latency
 * - Verify operation latency
 * - List operation latency
 * - Error rates
 * 
 * Note: Snapshot operations are heavier, so lower concurrent users
 * and expect higher latencies than queue/alert operations.
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Trend, Counter } from 'k6/metrics';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.0.0/index.js';

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const AUTH_TOKEN = __ENV.AUTH_TOKEN || 'test-token';

// Custom metrics
const createSnapshotLatency = new Trend('create_snapshot_latency');
const verifySnapshotLatency = new Trend('verify_snapshot_latency');
const listSnapshotsLatency = new Trend('list_snapshots_latency');
const snapshotErrors = new Counter('snapshot_errors');

let snapshotIds = [];

// Thresholds (higher latencies expected for snapshot operations)
export const options = {
  stages: [
    // Ramp up to 30 VUs over 20 seconds
    { duration: '20s', target: 30 },
    // Sustain for 2 minutes
    { duration: '2m', target: 30 },
    // Brief spike to 50 VUs
    { duration: '30s', target: 50 },
    // Cool down
    { duration: '20s', target: 0 },
  ],
  thresholds: {
    'create_snapshot_latency': ['p(95)<2000', 'p(99)<3000', 'avg<1000'],
    'verify_snapshot_latency': ['p(95)<1500', 'p(99)<2500', 'avg<800'],
    'list_snapshots_latency': ['p(95)<500', 'p(99)<1000', 'avg<300'],
    'snapshot_errors': ['count<10'],
    'http_req_failed': ['rate<0.1'], // Allow 10% error rate for snapshots
  },
};

export default function () {
  const headers = {
    'Authorization': `Bearer ${AUTH_TOKEN}`,
    'Content-Type': 'application/json',
  };

  group('List Snapshots', () => {
    const listResponse = http.get(
      `${BASE_URL}/api/v1/admin/ops/backups/snapshots`,
      { headers, tags: { name: 'ListSnapshots' } }
    );

    listSnapshotsLatency.add(listResponse.timings.duration);

    check(listResponse, {
      'list snapshots status is 200': (r) => r.status === 200,
      'list snapshots response time < 500ms': (r) => r.timings.duration < 500,
      'list snapshots returns array': (r) => {
        try {
          const body = JSON.parse(r.body);
          return Array.isArray(body);
        } catch {
          return false;
        }
      },
    }) || snapshotErrors.add(1);

    sleep(0.5);
  });

  group('Create Snapshot', () => {
    const createPayload = JSON.stringify({
      snapshot_name: `snapshot-${randomString(8)}-${Date.now()}`,
      snapshot_type: ['full', 'incremental'][Math.floor(Math.random() * 2)],
      retention_days: Math.floor(Math.random() * 30) + 7,
      storage_location: `s3://backups/${randomString(6)}`,
      compression_format: ['gzip', 'snappy'][Math.floor(Math.random() * 2)],
    });

    const createResponse = http.post(
      `${BASE_URL}/api/v1/admin/ops/backups/snapshots`,
      createPayload,
      { headers, tags: { name: 'CreateSnapshot' } }
    );

    createSnapshotLatency.add(createResponse.timings.duration);

    check(createResponse, {
      'create snapshot status is 201': (r) => r.status === 201,
      'create snapshot response time < 2000ms': (r) => r.timings.duration < 2000,
      'create snapshot returns ID': (r) => {
        try {
          const body = JSON.parse(r.body);
          if (body.id) {
            snapshotIds.push(body.id);
          }
          return !!body.id;
        } catch {
          return false;
        }
      },
    }) || snapshotErrors.add(1);

    sleep(1);
  });

  // Verify a random snapshot if we have any
  if (snapshotIds.length > 0) {
    group('Verify Snapshot', () => {
      const snapshotId = snapshotIds[Math.floor(Math.random() * snapshotIds.length)];
      
      const verifyPayload = JSON.stringify({
        expected_checksum: `checksum-${randomString(16)}`,
      });

      const verifyResponse = http.post(
        `${BASE_URL}/api/v1/admin/ops/backups/snapshots/${snapshotId}/verify`,
        verifyPayload,
        { headers, tags: { name: 'VerifySnapshot' } }
      );

      verifySnapshotLatency.add(verifyResponse.timings.duration);

      check(verifyResponse, {
        'verify snapshot status is 200 or 400 or 404': (r) => [200, 400, 404].includes(r.status),
        'verify snapshot response time < 1500ms': (r) => r.timings.duration < 1500,
      }) || snapshotErrors.add(1);

      sleep(1);
    });
  }
}

/**
 * Run this test with:
 * 
 * Basic run:
 *   k6 run snapshot-operations.js
 * 
 * With custom settings:
 *   k6 run snapshot-operations.js \
 *     -e BASE_URL=http://localhost:8000 \
 *     -e AUTH_TOKEN=your-token
 * 
 * Generate detailed results:
 *   k6 run --out json=snapshot-results.json snapshot-operations.js
 * 
 * Run all snapshot tests in sequence:
 *   for test in snapshot-operations.js; do
 *     k6 run $test --summary-export=${test%.js}-summary.json
 *   done
 */
