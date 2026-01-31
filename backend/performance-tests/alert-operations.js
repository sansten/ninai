/**
 * Alert Operations Load Test
 * 
 * Tests throughput and latency of alert rule management:
 * - Create alert rules
 * - Disable alerts
 * - Auto-create SLA breach alerts
 * 
 * Scenarios:
 * 1. Ramp-up: 0->50 VUs over 20s
 * 2. Sustained: 50 VUs for 90 seconds at ~50 requests/sec
 * 3. Peak: Brief spike to 100 VUs
 * 4. Cooldown: Scale back to 0
 * 
 * Metrics collected:
 * - Alert creation latency
 * - Error rate
 * - Throughput (alerts/sec)
 * - Peak resource consumption
 */

import http from 'k6/http';
import { check, group, sleep } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString } from 'https://jslib.k6.io/k6-utils/1.0.0/index.js';

// Configuration
const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const AUTH_TOKEN = __ENV.AUTH_TOKEN || 'test-token';

// Custom metrics
const createAlertLatency = new Trend('create_alert_latency');
const disableAlertLatency = new Trend('disable_alert_latency');
const autoCreateAlertLatency = new Trend('auto_create_alert_latency');
const alertOperationErrors = new Counter('alert_operation_errors');

let alertIds = [];

// Thresholds
export const options = {
  stages: [
    // Ramp up to 50 VUs over 20 seconds
    { duration: '20s', target: 50 },
    // Sustain 50 VUs for 90 seconds (target: ~50 alerts/sec)
    { duration: '90s', target: 50 },
    // Spike to 100 VUs for 30 seconds (stress test)
    { duration: '30s', target: 100 },
    // Cool down to 0
    { duration: '20s', target: 0 },
  ],
  thresholds: {
    'create_alert_latency': ['p(95)<300', 'p(99)<500', 'avg<200'],
    'disable_alert_latency': ['p(95)<300', 'p(99)<500', 'avg<200'],
    'auto_create_alert_latency': ['p(95)<400', 'p(99)<700', 'avg<250'],
    'alert_operation_errors': ['count<15'],
    'http_req_failed': ['rate<0.05'], // Max 5% error rate
  },
};

export default function () {
  const headers = {
    'Authorization': `Bearer ${AUTH_TOKEN}`,
    'Content-Type': 'application/json',
  };

  group('Create Alert Rule', () => {
    const createPayload = JSON.stringify({
      name: `Alert-${randomString(8)}`,
      severity: ['critical', 'high', 'medium', 'low'][Math.floor(Math.random() * 4)],
      route: `test.route.${Math.floor(Math.random() * 100)}`,
      channel: ['webhook', 'email', 'slack'][Math.floor(Math.random() * 3)],
      target: `http://example.com/webhook/${Math.floor(Math.random() * 1000)}`,
      enabled: true,
    });

    const createResponse = http.post(
      `${BASE_URL}/api/v1/admin/ops/alerts`,
      createPayload,
      { headers, tags: { name: 'CreateAlert' } }
    );

    createAlertLatency.add(createResponse.timings.duration);

    check(createResponse, {
      'create alert status is 201': (r) => r.status === 201,
      'create alert response time < 300ms': (r) => r.timings.duration < 300,
      'create alert returns alert ID': (r) => {
        try {
          const body = JSON.parse(r.body);
          if (body.id) {
            alertIds.push(body.id);
          }
          return !!body.id;
        } catch {
          return false;
        }
      },
    }) || alertOperationErrors.add(1);

    sleep(0.1);
  });

  // Disable random alert if we have any
  if (alertIds.length > 0) {
    group('Disable Alert Rule', () => {
      const alertId = alertIds[Math.floor(Math.random() * alertIds.length)];
      
      const disableResponse = http.post(
        `${BASE_URL}/api/v1/admin/ops/alerts/${alertId}/disable`,
        null,
        { headers, tags: { name: 'DisableAlert' } }
      );

      disableAlertLatency.add(disableResponse.timings.duration);

      check(disableResponse, {
        'disable alert status is 200 or 404': (r) => r.status === 200 || r.status === 404,
        'disable alert response time < 300ms': (r) => r.timings.duration < 300,
      }) || alertOperationErrors.add(1);

      sleep(0.1);
    });
  }

  group('Auto-Create SLA Breach Alert', () => {
    const threshold = (Math.random() * 50).toFixed(1);
    const severity = ['critical', 'high', 'medium'][Math.floor(Math.random() * 3)];

    const autoCreateResponse = http.post(
      `${BASE_URL}/api/v1/admin/ops/alerts/auto-create?threshold=${threshold}&severity=${severity}`,
      null,
      { headers, tags: { name: 'AutoCreateAlert' } }
    );

    autoCreateAlertLatency.add(autoCreateResponse.timings.duration);

    check(autoCreateResponse, {
      'auto-create alert status is 201': (r) => r.status === 201,
      'auto-create alert response time < 400ms': (r) => r.timings.duration < 400,
      'auto-create alert returns data': (r) => {
        try {
          const body = JSON.parse(r.body);
          if (body.id) {
            alertIds.push(body.id);
          }
          return !!body.id;
        } catch {
          return false;
        }
      },
    }) || alertOperationErrors.add(1);

    sleep(0.2);
  });
}

/**
 * Run this test with:
 * 
 * Basic run:
 *   k6 run alert-operations.js
 * 
 * With custom auth token:
 *   k6 run alert-operations.js \
 *     -e BASE_URL=http://localhost:8000 \
 *     -e AUTH_TOKEN=your-token
 * 
 * Generate JSON output:
 *   k6 run --out json=alert-results.json alert-operations.js
 * 
 * Run with summary:
 *   k6 run alert-operations.js --summary-export=alert-summary.json
 */
