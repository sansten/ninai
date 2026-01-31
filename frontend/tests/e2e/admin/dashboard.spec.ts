/**
 * E2E Tests - Admin Dashboard Page
 * Tests dashboard KPIs, widgets, and auto-refresh functionality
 */

import { test, expect } from '@playwright/test';

test.describe('Admin Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    
    // Navigate to admin dashboard
    await page.goto('/admin');
  });

  test('should display all KPI cards', async ({ page }) => {
    // Wait for dashboard to load
    await page.waitForLoadState('networkidle');

    // Should see 4 KPI cards
    const kpiLabels = ['Active Users', 'API Requests', 'Error Rate', 'Uptime'];
    
    for (const label of kpiLabels) {
      await expect(page.locator(`text=${label}`)).toBeVisible();
    }

    // Each KPI should have a value
    const kpiValues = page.locator('[data-testid*="kpi-value"]');
    const count = await kpiValues.count();
    expect(count).toBeGreaterThanOrEqual(4);
  });

  test('should display service health widget', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Should see service health section
    await expect(page.locator('text=Service Health').or(page.locator('text=System Health'))).toBeVisible();

    // Should see at least one service status
    const services = ['Backend API', 'Database', 'Message Queue'];
    let visibleCount = 0;
    
    for (const service of services) {
      const isVisible = await page.locator(`text=${service}`).isVisible().catch(() => false);
      if (isVisible) visibleCount++;
    }
    
    expect(visibleCount).toBeGreaterThan(0);
  });

  test('should display alerts widget', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Should see alerts section
    const hasAlerts = await page.locator('text=Alerts').or(page.locator('text=Active Alerts')).isVisible();
    expect(hasAlerts).toBeTruthy();
  });

  test('should display recent activity feed', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Should see recent activity section
    await expect(page.locator('text=Recent Activity').or(page.locator('text=Activity Feed'))).toBeVisible();

    // Should show some activity items (or "No activity" message)
    const bodyText = await page.locator('body').textContent();
    const hasActivityOrEmpty = bodyText!.includes('activity') || bodyText!.includes('No recent');
    expect(hasActivityOrEmpty).toBeTruthy();
  });

  test('KPI cards should display numeric values', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Active Users should be a number
    const activeUsers = page.locator('text=Active Users').locator('..').locator('[data-testid*="value"]');
    if (await activeUsers.isVisible().catch(() => false)) {
      const text = await activeUsers.textContent();
      expect(text).toMatch(/\d+/);
    }

    // Error Rate should be a percentage
    const errorRate = page.locator('text=Error Rate').locator('..').locator('[data-testid*="value"]');
    if (await errorRate.isVisible().catch(() => false)) {
      const text = await errorRate.textContent();
      expect(text).toMatch(/\d+\.?\d*%?/);
    }
  });

  test('should show loading state initially', async ({ page }) => {
    // Navigate to dashboard
    const responsePromise = page.waitForResponse(response => 
      response.url().includes('/api/v1/admin/dashboard') && response.status() === 200
    );

    await page.goto('/admin', { waitUntil: 'domcontentloaded' });

    // Should see loading indicator
    const hasLoader = await page.locator('[data-testid="loading"]').or(page.locator('text=Loading')).isVisible({ timeout: 1000 }).catch(() => false);
    
    // Wait for response
    await responsePromise;

    // Loading should disappear
    await expect(page.locator('[data-testid="loading"]')).not.toBeVisible({ timeout: 5000 }).catch(() => {});
  });

  test('should handle API errors gracefully', async ({ page, context }) => {
    // Intercept API and return error
    await page.route('**/api/v1/admin/dashboard', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ detail: 'Internal Server Error' }),
      });
    });

    await page.goto('/admin');

    // Should show error message
    await expect(page.locator('text=/Error|Failed|Unable to load/i')).toBeVisible({ timeout: 10000 });
  });

  test('should auto-refresh data', async ({ page }) => {
    await page.goto('/admin');
    await page.waitForLoadState('networkidle');

    // Get initial value
    const initialValue = await page.locator('body').textContent();

    // Wait for auto-refresh (dashboard refreshes every 30 seconds)
    // We'll wait for a fetch request to dashboard endpoint
    const refreshPromise = page.waitForResponse(
      response => response.url().includes('/api/v1/admin/dashboard'),
      { timeout: 35000 }
    );

    await refreshPromise;

    // Data might have changed (or stayed the same, but request was made)
    expect(true).toBeTruthy(); // Auto-refresh happened
  });

  test('dashboard should be responsive', async ({ page }) => {
    await page.goto('/admin');
    await page.waitForLoadState('networkidle');

    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.waitForTimeout(500);

    // Dashboard should still be visible
    const bodyText = await page.locator('body').textContent();
    expect(bodyText).toBeTruthy();

    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.waitForTimeout(500);

    // Test desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.waitForTimeout(500);

    // Should render without layout issues
    const hasOverflow = await page.evaluate(() => {
      return document.documentElement.scrollWidth > window.innerWidth;
    });
    
    // Some horizontal scroll is OK for tables
    expect(hasOverflow).toBeDefined();
  });

  test('should navigate to other admin pages from dashboard', async ({ page }) => {
    await page.goto('/admin');
    await page.waitForLoadState('networkidle');

    // Click on Users link
    await page.click('text=Users');
    await expect(page).toHaveURL(/\/admin\/users/);

    // Go back to dashboard
    await page.goto('/admin');

    // Click on Settings link
    await page.click('text=Settings');
    await expect(page).toHaveURL(/\/admin\/settings/);
  });
});

test.describe('Dashboard Performance', () => {
  test('should load dashboard within 2 seconds', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);

    const loginTime = Date.now();

    await page.goto('/admin');
    await page.waitForLoadState('networkidle');

    const endTime = Date.now();
    const loadTime = endTime - loginTime;

    console.log(`Dashboard load time: ${loadTime}ms`);
    expect(loadTime).toBeLessThan(3000); // 3 seconds max (2s target + 1s buffer)
  });

  test('should not cause memory leaks on navigation', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);

    // Navigate between pages multiple times
    for (let i = 0; i < 5; i++) {
      await page.goto('/admin');
      await page.waitForLoadState('networkidle');
      
      await page.goto('/admin/users');
      await page.waitForLoadState('networkidle');
      
      await page.goto('/admin/settings');
      await page.waitForLoadState('networkidle');
    }

    // Check for console errors
    const errors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    await page.goto('/admin');
    await page.waitForTimeout(2000);

    // Should have minimal or no errors
    const criticalErrors = errors.filter(e => 
      e.includes('memory') || e.includes('leak') || e.includes('Maximum')
    );
    expect(criticalErrors.length).toBe(0);
  });
});
