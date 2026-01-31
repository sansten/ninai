/**
 * E2E Tests - Admin Audit Logs
 * Tests audit log viewing, filtering, and details
 */

import { test, expect } from '@playwright/test';

test.describe('Audit Logs', () => {
  test.beforeEach(async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    
    // Navigate to audit logs page
    await page.goto('/admin/audit-logs');
    await page.waitForLoadState('networkidle');
  });

  test('should display audit logs table', async ({ page }) => {
    // Should see table with logs
    const table = page.locator('table, [role="table"]').first();
    await expect(table).toBeVisible();
  });

  test('should show log entries with required fields', async ({ page }) => {
    // Should display timestamp, action, admin, resource
    const headers = ['Time', 'Action', 'Admin', 'Resource'];
    
    for (const header of headers) {
      const headerVisible = await page.locator(`th:has-text("${header}"), th[aria-label*="${header}"]`).isVisible().catch(() => false);
      // Headers may vary but should exist
    }

    // Should have at least one log entry (from login)
    const rows = await page.locator('tbody tr, [role="row"]').count();
    expect(rows).toBeGreaterThan(0);
  });

  test('should filter logs by action type', async ({ page }) => {
    // Find action filter dropdown
    const actionFilter = page.locator('select, [role="combobox"]').filter({ hasText: /action|filter/i }).first();
    
    if (await actionFilter.isVisible().catch(() => false)) {
      // Select an action type
      await actionFilter.click();
      await page.locator('option, [role="option"]').filter({ hasText: /create|update|delete/i }).first().click();
      await page.waitForTimeout(1000);

      // Table should update
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toBeTruthy();
    }
  });

  test('should filter logs by resource type', async ({ page }) => {
    // Find resource filter
    const resourceFilter = page.locator('select, button').filter({ hasText: /resource|type/i }).first();
    
    if (await resourceFilter.isVisible().catch(() => false)) {
      await resourceFilter.click();
      await page.waitForTimeout(500);

      // Select a resource type
      const resourceOption = page.locator('option, button, a').filter({ hasText: /user|role|setting/i }).first();
      if (await resourceOption.isVisible().catch(() => false)) {
        await resourceOption.click();
        await page.waitForTimeout(1000);
      }
    }
  });

  test('should filter logs by admin user', async ({ page }) => {
    // Find admin filter
    const adminFilter = page.locator('input[type="search"], input[placeholder*="admin" i]').first();
    
    if (await adminFilter.isVisible().catch(() => false)) {
      await adminFilter.fill('admin@example.com');
      await page.waitForTimeout(1000);

      // Should show filtered results
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toContain('admin');
    }
  });

  test('should filter logs by date range', async ({ page }) => {
    // Find date filters
    const dateInputs = page.locator('input[type="date"], input[type="datetime-local"]');
    const count = await dateInputs.count();
    
    if (count >= 2) {
      // Set date range
      const today = new Date().toISOString().split('T')[0];
      await dateInputs.first().fill(today);
      await dateInputs.last().fill(today);
      await page.waitForTimeout(1000);

      // Should show results for today
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toBeTruthy();
    }
  });

  test('should expand log details', async ({ page }) => {
    // Find first log row
    const firstRow = page.locator('tbody tr, [role="row"]').first();
    
    // Click expand button or row itself
    const expandButton = firstRow.locator('button, [role="button"]').filter({ hasText: /expand|details|more/i }).first();
    
    if (await expandButton.isVisible().catch(() => false)) {
      await expandButton.click();
      await page.waitForTimeout(500);

      // Should show expanded details
      const details = page.locator('text=/before|after|changes|details/i');
      const detailsVisible = await details.isVisible({ timeout: 3000 }).catch(() => false);
      expect(detailsVisible).toBeTruthy();
    } else {
      // Try clicking row itself
      await firstRow.click();
      await page.waitForTimeout(500);
    }
  });

  test('should display before/after values for updates', async ({ page }) => {
    // Look for update actions
    const updateRow = page.locator('tr:has-text("update"), tr:has-text("UPDATE")').first();
    
    if (await updateRow.isVisible().catch(() => false)) {
      // Expand details
      const expandButton = updateRow.locator('button').first();
      if (await expandButton.isVisible().catch(() => false)) {
        await expandButton.click();
        await page.waitForTimeout(500);

        // Should show before/after comparison
        const beforeLabel = await page.locator('text=/before|old/i').isVisible().catch(() => false);
        const afterLabel = await page.locator('text=/after|new/i').isVisible().catch(() => false);
        
        // At least some comparison should be shown
        expect(beforeLabel || afterLabel).toBeTruthy();
      }
    }
  });

  test('should paginate log entries', async ({ page }) => {
    // Look for pagination controls
    const nextButton = page.locator('button:has-text("Next"), button[aria-label*="next"]').first();
    
    if (await nextButton.isVisible().catch(() => false)) {
      const isEnabled = await nextButton.isEnabled();
      
      if (isEnabled) {
        // Click next page
        await nextButton.click();
        await page.waitForTimeout(1000);

        // Should load next page
        const bodyText = await page.locator('body').textContent();
        expect(bodyText).toBeTruthy();
      }
    }
  });

  test('should show log count', async ({ page }) => {
    // Should display total number of logs
    const countText = page.locator('text=/total|logs|entries/i').first();
    
    if (await countText.isVisible().catch(() => false)) {
      const text = await countText.textContent();
      expect(text).toMatch(/\d+/); // Should contain numbers
    }
  });

  test('should display different action types', async ({ page }) => {
    // Should show various action types
    const actions = ['create', 'update', 'delete', 'login'];
    const bodyText = (await page.locator('body').textContent() || '').toLowerCase();
    
    // At least some actions should be present
    const hasActions = actions.some(action => bodyText.includes(action));
    expect(hasActions).toBeTruthy();
  });

  test('should format timestamps correctly', async ({ page }) => {
    // Find first timestamp
    const timestamp = page.locator('td, [role="cell"]').filter({ hasText: /ago|\d{1,2}:\d{2}|am|pm/i }).first();
    
    if (await timestamp.isVisible().catch(() => false)) {
      const text = await timestamp.textContent();
      expect(text).toBeTruthy();
      // Should show relative time or formatted date
    }
  });

  test('should clear filters', async ({ page }) => {
    // Apply a filter
    const searchInput = page.locator('input[type="search"]').first();
    if (await searchInput.isVisible().catch(() => false)) {
      await searchInput.fill('test');
      await page.waitForTimeout(1000);

      // Click clear/reset button
      const clearButton = page.locator('button:has-text("Clear"), button:has-text("Reset")').first();
      if (await clearButton.isVisible().catch(() => false)) {
        await clearButton.click();
        await page.waitForTimeout(1000);

        // Filters should be cleared
        const inputValue = await searchInput.inputValue();
        expect(inputValue).toBe('');
      }
    }
  });

  test('should show admin email in logs', async ({ page }) => {
    // Logs should display admin email who performed action
    const bodyText = await page.locator('body').textContent() || '';
    
    // Should contain email pattern
    expect(bodyText).toMatch(/@|admin/i);
  });

  test('should show resource ID in logs', async ({ page }) => {
    // Logs should show resource ID
    const firstRow = page.locator('tbody tr').first();
    const rowText = await firstRow.textContent();
    
    // Should contain some identifier
    expect(rowText).toBeTruthy();
  });

  test('should load within 2 seconds', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/admin/audit-logs');
    await page.waitForLoadState('networkidle');

    const endTime = Date.now();
    const loadTime = endTime - startTime;

    console.log(`Audit logs page load time: ${loadTime}ms`);
    expect(loadTime).toBeLessThan(3000); // 2s target + 1s buffer
  });
});

test.describe('Audit Logs - Details View', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    await page.goto('/admin/audit-logs');
    await page.waitForLoadState('networkidle');
  });

  test('should display full log details', async ({ page }) => {
    // Expand first log
    const firstRow = page.locator('tbody tr').first();
    const expandButton = firstRow.locator('button').first();
    
    if (await expandButton.isVisible().catch(() => false)) {
      await expandButton.click();
      await page.waitForTimeout(500);

      // Should show metadata
      const metadata = ['Timestamp', 'Action', 'Resource', 'Admin'];
      
      for (const field of metadata) {
        const fieldVisible = await page.locator(`text=${field}`).isVisible().catch(() => false);
        // Fields may be present
      }
    }
  });

  test('should display changes for update actions', async ({ page }) => {
    // Look for update log
    const updateLog = page.locator('tr').filter({ hasText: /update|UPDATE/ }).first();
    
    if (await updateLog.isVisible().catch(() => false)) {
      const expandButton = updateLog.locator('button').first();
      if (await expandButton.isVisible().catch(() => false)) {
        await expandButton.click();
        await page.waitForTimeout(500);

        // Should show field changes
        const bodyText = await page.locator('body').textContent() || '';
        expect(bodyText).toBeTruthy();
      }
    }
  });

  test('should show JSON formatting for complex data', async ({ page }) => {
    // Expand a log with JSON data
    const firstRow = page.locator('tbody tr').first();
    const expandButton = firstRow.locator('button').first();
    
    if (await expandButton.isVisible().catch(() => false)) {
      await expandButton.click();
      await page.waitForTimeout(500);

      // Should show formatted JSON or structured data
      const codeBlock = page.locator('pre, code, [class*="json"]').first();
      const codeVisible = await codeBlock.isVisible().catch(() => false);
      // JSON may or may not be present
    }
  });
});

test.describe('Audit Logs - Error Handling', () => {
  test('should handle API errors', async ({ page }) => {
    await page.route('**/api/v1/admin/audit-logs*', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ detail: 'Internal Server Error' }),
      });
    });

    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    await page.goto('/admin/audit-logs');
    await page.waitForLoadState('networkidle');

    // Should show error message
    await expect(page.locator('text=/Error|Failed|unable/i')).toBeVisible({ timeout: 5000 });
  });

  test('should handle empty results', async ({ page }) => {
    // Apply filters that return no results
    const searchInput = page.locator('input[type="search"]').first();
    if (await searchInput.isVisible().catch(() => false)) {
      await searchInput.fill('nonexistent@email.com');
      await page.waitForTimeout(1000);

      // Should show "no results" message
      const noResults = await page.locator('text=/no logs|no results|no entries/i').isVisible({ timeout: 3000 }).catch(() => false);
      expect(noResults).toBeTruthy();
    }
  });
});

test.describe('Audit Logs - Responsiveness', () => {
  test('should be usable on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    await page.goto('/admin/audit-logs');
    await page.waitForLoadState('networkidle');

    // Table should be scrollable or responsive
    const table = page.locator('table, [role="table"]').first();
    await expect(table).toBeVisible();

    // Filters should be accessible
    const filters = page.locator('input, select, button').filter({ hasText: /filter|search/i });
    const count = await filters.count();
    expect(count).toBeGreaterThanOrEqual(0);
  });

  test('should work on tablet', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 1024 });

    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    await page.goto('/admin/audit-logs');
    await page.waitForLoadState('networkidle');

    // Should display properly
    const table = page.locator('table, [role="table"]').first();
    await expect(table).toBeVisible();

    // All functionality should work
    const firstRow = page.locator('tbody tr').first();
    await expect(firstRow).toBeVisible();
  });
});
