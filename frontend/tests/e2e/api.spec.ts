import { test, expect } from '@playwright/test';

const API_URL = process.env.E2E_API_URL || 'http://localhost:8000/api/v1';

test('events endpoint is protected (401 without auth)', async ({ request }) => {
  const res = await request.get(`${API_URL}/events/`);
  expect([401, 403]).toContain(res.status());
});

test('webhooks list endpoint is protected (401 without auth)', async ({ request }) => {
  const res = await request.get(`${API_URL}/events/webhooks`);
  expect([401, 403]).toContain(res.status());
});
