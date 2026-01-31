/**
 * Queue Operations Load Test
 * 
 * Tests the throughput and latency of queue management operations:
 * - Pause queue
 * - Resume queue  
 * - Update queue config
 * 
 * Scenarios:
 * 1. Ramp-up: 0->100 concurrent users over 30s
 * 2. Sustained: 100 concurrent users for 2 minutes
 * 3. Stress: Brief spike to 200 concurrent users
 * 
 * Metrics collected:
 * - Request latency (p50, p95, p99)
 * - Throughput (requests/sec)
 * - Error rate
 * - Resource usage baseline
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter, Gauge } from 'k6/metrics';

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const AUTH_TOKEN = __ENV.AUTH_TOKEN || 'test-token';
const ORG_ID = __ENV.ORG_ID || '00000000-0000-0000-0000-000000e2e001';
const QUEUE_NAMES = ['default', 'high', 'low'];

// Custom metrics
const queuePauseDuration = new Trend('queue_pause_duration');
const queueResumeLatency = new Trend('queue_resume_latency');
const queueUpdateLatency = new Trend('queue_update_latency');
const queueOperationErrors = new Counter('queue_operation_errors');
const activeConcurrentUsers = new Gauge('concurrent_users');

// Thresholds for pass/fail
export const options = {
  stages: [
    // Ramp up from 0 to 100 users over 30 seconds
    { duration: '30s', target: 100 },
    // Stay at 100 for 2 minutes
    { duration: '2m', target: 100 },
    // Stress test: spike to 200 concurrent for 30s
    { duration: '30s', target: 200 },
    // Back down to 0
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    'queue_pause_duration': ['p(95)<500', 'p(99)<1000', 'avg<300'],
    'queue_resume_latency': ['p(95)<500', 'p(99)<1000', 'avg<300'],
    'queue_update_latency': ['p(95)<800', 'p(99)<1500', 'avg<500'],
    'queue_operation_errors': ['count<10'],
    'http_req_duration': ['p(95)<1000'],
    'http_req_failed': ['rate<0.05'], // Allow 5% error rate max
  },
};

export default function () {
  const headers = {
    'Authorization': `Bearer ${AUTH_TOKEN}`,
    'Content-Type': 'application/json',
  };

  const queueName = QUEUE_NAMES[Math.floor(Math.random() * QUEUE_NAMES.length)];
  activeConcurrentUsers.add(1);

  group('Pause Queue Operations', () => {
    const pauseResponse = http.post(
      `${BASE_URL}/api/v1/admin/ops/queues/${queueName}/pause`,
      null,
      { headers, tags: { name: 'PauseQueue' } }
    );

    queuePauseDuration.add(pauseResponse.timings.duration);

    check(pauseResponse, {
      'pause queue status is 200': (r) => r.status === 200,
      'pause queue response time < 500ms': (r) => r.timings.duration < 500,
      'pause queue has success field': (r) => JSON.parse(r.body).success === true,
    }) || queueOperationErrors.add(1);

    sleep(0.5);
  });

  group('Resume Queue Operations', () => {
    const resumeResponse = http.post(
      `${BASE_URL}/api/v1/admin/ops/queues/${queueName}/resume`,
      null,
      { headers, tags: { name: 'ResumeQueue' } }
    );

    queueResumeLatency.add(resumeResponse.timings.duration);

    check(resumeResponse, {
      'resume queue status is 200': (r) => r.status === 200,
      'resume queue response time < 500ms': (r) => r.timings.duration < 500,
      'resume queue has success field': (r) => JSON.parse(r.body).success === true,
    }) || queueOperationErrors.add(1);

    sleep(0.5);
  });

  group('Update Queue Config', () => {
    const updatePayload = JSON.stringify({
      priority_weight: Math.random() * 2 + 0.5,
      max_retries: Math.floor(Math.random() * 5) + 1,
      concurrency: Math.floor(Math.random() * 20) + 5,
    });

    const updateResponse = http.put(
      `${BASE_URL}/api/v1/admin/ops/queues/${queueName}`,
      updatePayload,
      { headers, tags: { name: 'UpdateQueueConfig' } }
    );

    queueUpdateLatency.add(updateResponse.timings.duration);

    check(updateResponse, {
      'update queue config status is 200': (r) => r.status === 200,
      'update queue config response time < 800ms': (r) => r.timings.duration < 800,
      'update queue config has success field': (r) => JSON.parse(r.body).success === true,
    }) || queueOperationErrors.add(1);

    sleep(1);
  });

  activeConcurrentUsers.add(-1);
}

/**
 * Run this test with:
 * 
 * Basic run:
 *   k6 run queue-operations.js
 * 
 * With custom settings:
 *   k6 run queue-operations.js \
 *     -e BASE_URL=http://localhost:8000 \
 *     -e AUTH_TOKEN=your-token
 * 
 * Generate JSON report:
 *   k6 run --out json=queue-results.json queue-operations.js
 * 
 * Cloud run (k6 Cloud):
 *   k6 cloud queue-operations.js
 */
