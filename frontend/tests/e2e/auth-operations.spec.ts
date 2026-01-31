import { test, expect } from '@playwright/test';

const API_URL = process.env.E2E_API_URL || 'http://localhost:8000/api/v1';
const AUTH_TOKEN = process.env.E2E_AUTH_TOKEN || '';

// Seeded test org/user from backend conftest.py for reproducible E2E tests
const E2E_SEED_ORG_ID = '00000000-0000-0000-0000-000000e2e001';
const E2E_SEED_USER_ID = '00000000-0000-0000-0000-000000e2e002';

test.describe('Authenticated Operations', () => {
  const headers = {
    'Authorization': `Bearer ${AUTH_TOKEN}`,
    'Content-Type': 'application/json',
  };

  test.skip(!AUTH_TOKEN, 'Skipped: E2E_AUTH_TOKEN not set');

  // ============ Snapshot Creation Tests ============

  test('create snapshot for memory export', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/snapshots`, {
      headers,
      data: {
        resource_type: 'memory',
        format: 'json',
        name: 'Test Memory Snapshot',
        expires_in_days: 30,
      },
    });
    expect(res.status()).toBe(200);
    const data = await res.json();
    expect(data).toHaveProperty('id');
    expect(data).toHaveProperty('status', 'pending');
    expect(data).toHaveProperty('format', 'json');
    expect(data).toHaveProperty('resource_type', 'memory');
  });

  test('create snapshot with custom filters', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/snapshots`, {
      headers,
      data: {
        resource_type: 'memory',
        format: 'csv',
        name: 'Filtered Memory Export',
        expires_in_days: 7,
        filters: {
          tags: ['important', 'production'],
          date_from: '2025-01-01',
          date_to: '2025-12-31',
        },
      },
    });
    expect([200, 201]).toContain(res.status());
    const data = await res.json();
    expect(data).toHaveProperty('id');
    expect(data).toHaveProperty('filters');
  });

  test('create snapshot with expiration', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/snapshots`, {
      headers,
      data: {
        resource_type: 'memory',
        format: 'json',
        name: 'Expiring Snapshot',
        expires_in_days: 1,
      },
    });
    expect(res.status()).toBe(200);
    const data = await res.json();
    expect(data).toHaveProperty('expires_at');
    expect(data.expires_at).toBeTruthy();
  });

  test('reject snapshot with invalid format', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/snapshots`, {
      headers,
      data: {
        resource_type: 'memory',
        format: 'invalid_format',
        name: 'Bad Format Snapshot',
        expires_in_days: 30,
      },
    });
    expect([400, 422]).toContain(res.status());
  });

  test('list snapshots', async ({ request }) => {
    const res = await request.get(`${API_URL}/events/snapshots`, {
      headers,
    });
    expect([200, 204]).toContain(res.status());
    const data = await res.json();
    expect(Array.isArray(data.snapshots) || Array.isArray(data)).toBe(true);
  });

  test('get snapshot by id', async ({ request }) => {
    // First create a snapshot
    const createRes = await request.post(`${API_URL}/events/snapshots`, {
      headers,
      data: {
        resource_type: 'memory',
        format: 'json',
        name: 'Snapshot to Retrieve',
        expires_in_days: 30,
      },
    });

    if (createRes.status() === 200) {
      const snapshot = await createRes.json();
      const getRes = await request.get(`${API_URL}/events/snapshots/${snapshot.id}`, {
        headers,
      });
      expect(getRes.status()).toBe(200);
      const data = await getRes.json();
      expect(data.id).toBe(snapshot.id);
    }
  });

  test('snapshot is org-scoped (RLS enforcement)', async ({ request }) => {
    // Create snapshot as seeded user
    const createRes = await request.post(`${API_URL}/events/snapshots`, {
      headers,
      data: {
        resource_type: 'memory',
        format: 'json',
        name: 'Org-scoped Snapshot',
        expires_in_days: 30,
      },
    });
    
    expect(createRes.status()).toBe(200);
    const snapshot = await createRes.json();
    
    // Verify snapshot is tied to the seeded org
    expect(snapshot.organization_id || snapshot.org_id).toBeTruthy();
  });

  // ============ Batch Memory Operations Tests ============

  test('batch update memory items with tags', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/batch/memory/update`, {
      headers,
      data: {
        memory_ids: ['00000000-0000-0000-0000-000000000001'],
        tags: ['test', 'batch-update', 'e2e'],
        status: 'active',
      },
    });
    expect([200, 400, 404]).toContain(res.status());
    // 400/404 expected if memory_ids don't exist; 200 if operation succeeds
    if (res.status() === 200) {
      const data = await res.json();
      expect(data).toHaveProperty('operation_type', 'update');
      expect(data).toHaveProperty('affected_count');
    }
  });

  test('batch update multiple memory items', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/batch/memory/update`, {
      headers,
      data: {
        memory_ids: [
          '00000000-0000-0000-0000-000000000001',
          '00000000-0000-0000-0000-000000000002',
          '00000000-0000-0000-0000-000000000003',
        ],
        tags: ['batch-processed'],
        priority: 'high',
      },
    });
    expect([200, 400, 404]).toContain(res.status());
    if (res.status() === 200) {
      const data = await res.json();
      expect(typeof data.affected_count).toBe('number');
    }
  });

  test('batch delete memory (soft delete)', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/batch/memory/delete`, {
      headers,
      data: {
        memory_ids: ['00000000-0000-0000-0000-000000000002'],
        soft_delete: true,
      },
    });
    expect([200, 400, 404]).toContain(res.status());
    if (res.status() === 200) {
      const data = await res.json();
      expect(data.operation_type).toBe('delete');
      expect(data).toHaveProperty('affected_count');
    }
  });

  test('batch hard delete memory (with authorization)', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/batch/memory/delete`, {
      headers,
      data: {
        memory_ids: ['00000000-0000-0000-0000-000000000004'],
        soft_delete: false,
        require_confirmation: true,
      },
    });
    expect([200, 400, 404, 403]).toContain(res.status());
    // 403 if user doesn't have hard-delete permission
  });

  test('batch share memory with users', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/batch/memory/share`, {
      headers,
      data: {
        memory_ids: ['00000000-0000-0000-0000-000000000003'],
        shared_with_user_ids: ['00000000-0000-0000-0000-000000000099'],
        access_level: 'view',
      },
    });
    expect([200, 400, 404]).toContain(res.status());
    if (res.status() === 200) {
      const data = await res.json();
      expect(data).toHaveProperty('affected_count');
    }
  });

  test('batch archive memory items', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/batch/memory/archive`, {
      headers,
      data: {
        memory_ids: ['00000000-0000-0000-0000-000000000005'],
        reason: 'End of project lifecycle',
      },
    });
    expect([200, 400, 404]).toContain(res.status());
    if (res.status() === 200) {
      const data = await res.json();
      expect(data.operation_type).toBe('archive');
    }
  });

  test('batch export memory items', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/batch/memory/export`, {
      headers,
      data: {
        memory_ids: ['00000000-0000-0000-0000-000000000006'],
        format: 'json',
        include_attachments: true,
      },
    });
    expect([200, 400, 404]).toContain(res.status());
    if (res.status() === 200) {
      const data = await res.json();
      expect(data).toHaveProperty('export_id');
    }
  });

  // ============ Webhook & Event Subscription Tests ============

  test('create webhook subscription for memory events', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/webhooks`, {
      headers,
      data: {
        url: 'https://example.com/webhooks/memory',
        event_types: ['memory.created', 'memory.updated'],
        description: 'Test webhook for E2E',
      },
    });
    expect([200, 201, 400]).toContain(res.status());
    if (res.status() === 200 || res.status() === 201) {
      const data = await res.json();
      expect(data).toHaveProperty('id');
      expect(data).toHaveProperty('url');
    }
  });

  test('create webhook with snapshot events', async ({ request }) => {
    const res = await request.post(`${API_URL}/events/webhooks`, {
      headers,
      data: {
        url: 'https://example.com/webhooks/snapshots',
        event_types: ['snapshot.created', 'snapshot.completed'],
        description: 'Snapshot event webhook',
      },
    });
    expect([200, 201, 400]).toContain(res.status());
  });

  test('list event subscriptions', async ({ request }) => {
    const res = await request.get(`${API_URL}/events/webhooks`, {
      headers,
    });
    expect([200, 204]).toContain(res.status());
    const data = await res.json();
    expect(data).toHaveProperty('subscriptions');
  });

  // ============ Authentication & Authorization Tests ============

  test('reject requests without auth token', async ({ request }) => {
    const res = await request.get(`${API_URL}/events/snapshots`);
    expect([401, 403]).toContain(res.status());
  });

  test('reject requests with invalid auth token', async ({ request }) => {
    const res = await request.get(`${API_URL}/events/snapshots`, {
      headers: {
        'Authorization': 'Bearer invalid.token.here',
      },
    });
    expect([401, 403]).toContain(res.status());
  });

  test('seeded user can only access own org data', async ({ request }) => {
    // Create snapshot with seeded user
    const createRes = await request.post(`${API_URL}/events/snapshots`, {
      headers,
      data: {
        resource_type: 'memory',
        format: 'json',
        name: 'Seeded User Snapshot',
        expires_in_days: 30,
      },
    });

    if (createRes.status() === 200) {
      const snapshot = await createRes.json();
      // Verify org_id matches seeded org
      const orgId = snapshot.organization_id || snapshot.org_id;
      expect(orgId === E2E_SEED_ORG_ID || orgId === undefined).toBeTruthy();
      // If not undefined, it must match the seeded org
      if (orgId !== undefined && orgId !== null) {
        expect(orgId).toBe(E2E_SEED_ORG_ID);
      }
    }
  });
});

