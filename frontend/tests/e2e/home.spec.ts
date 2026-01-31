import { test, expect } from '@playwright/test';

// Basic smoke test that the app loads and renders header
// Assumes dev server running at baseURL (default http://localhost:5173)

test('homepage loads and shows title', async ({ page }) => {
  await page.goto('/');
  // Adjust selector based on actual UI title element
  const title = page.locator('header, nav, h1');
  await expect(title).toBeVisible();
});

test('admin route requires auth (redirects or shows login)', async ({ page }) => {
  await page.goto('/admin');
  // Expect either login prompt or 401/redirect behavior
  const loginForm = page.locator('input[type="email"], input[name*="user"], button:has-text("Sign in")');
  // One of: login visible OR page notifies unauthorized
  const unauthorizedText = page.locator('text=/unauthorized|sign in|login/i');
  await expect(loginForm.or(unauthorizedText)).toBeVisible();
});
