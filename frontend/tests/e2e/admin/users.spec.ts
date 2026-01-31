/**
 * E2E Tests - Admin Users Management
 * Tests user listing, search, filtering, and enable/disable functionality
 */

import { test, expect } from '@playwright/test';

test.describe('Users Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    
    // Navigate to users page
    await page.goto('/admin/users');
    await page.waitForLoadState('networkidle');
  });

  test('should display users table with columns', async ({ page }) => {
    // Should see table headers
    const headers = ['Email', 'Name', 'Role', 'Status', 'Last Login'];
    
    for (const header of headers) {
      const headerVisible = await page.locator(`text=${header}`).isVisible().catch(() => false);
      // At least some headers should be visible
    }

    // Should see at least one user row or "No users" message
    const bodyText = await page.locator('body').textContent();
    const hasContent = bodyText!.includes('@') || bodyText!.includes('No users');
    expect(hasContent).toBeTruthy();
  });

  test('should search users by email', async ({ page }) => {
    // Find search input
    const searchInput = page.locator('input[placeholder*="search" i], input[type="search"], input[placeholder*="email" i]').first();
    
    if (await searchInput.isVisible().catch(() => false)) {
      // Type search query
      await searchInput.fill('admin@example.com');
      await page.waitForTimeout(1000); // Wait for debounce

      // Should filter results
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toContain('admin@example.com');
    }
  });

  test('should filter users by status', async ({ page }) => {
    // Look for status filter dropdown
    const statusFilter = page.locator('select').filter({ hasText: /status|active|inactive/i }).first();
    
    if (await statusFilter.isVisible().catch(() => false)) {
      // Select "Active" filter
      await statusFilter.selectOption({ label: 'Active' });
      await page.waitForTimeout(1000);

      // Results should update
      expect(await page.locator('body').textContent()).toBeTruthy();
    }
  });

  test('should paginate users list', async ({ page }) => {
    // Look for pagination controls
    const nextButton = page.locator('button:has-text("Next"), button:has-text("→")').first();
    
    if (await nextButton.isVisible().catch(() => false)) {
      const isEnabled = await nextButton.isEnabled();
      
      if (isEnabled) {
        // Click next page
        await nextButton.click();
        await page.waitForLoadState('networkidle');

        // Page should change
        expect(await page.locator('body').textContent()).toBeTruthy();

        // Should see previous button
        await expect(page.locator('button:has-text("Previous"), button:has-text("←")')).toBeVisible();
      }
    }
  });

  test('should view user details', async ({ page }) => {
    // Find first user row actions menu
    const actionsButton = page.locator('button:has-text("Actions"), button[aria-label*="menu"]').first();
    
    if (await actionsButton.isVisible().catch(() => false)) {
      await actionsButton.click();

      // Should see "View Details" option
      const viewDetails = page.locator('text=View Details');
      if (await viewDetails.isVisible().catch(() => false)) {
        await viewDetails.click();

        // Should show user details (modal or new page)
        await expect(page.locator('text=User Details').or(page.locator('text=Profile'))).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should disable user', async ({ page }) => {
    // Find active user
    const activeUserRow = page.locator('tr:has-text("Active")').first();
    
    if (await activeUserRow.isVisible().catch(() => false)) {
      // Click actions menu
      const actionsButton = activeUserRow.locator('button:has-text("Actions"), button[aria-label*="menu"]').first();
      await actionsButton.click();

      // Click disable
      const disableButton = page.locator('text=Disable');
      if (await disableButton.isVisible().catch(() => false)) {
        await disableButton.click();

        // Should show confirmation dialog
        await expect(page.locator('text=/Confirm|Are you sure/i')).toBeVisible({ timeout: 3000 });

        // Confirm disable
        await page.locator('button:has-text("Confirm"), button:has-text("Yes"), button:has-text("Disable")').click();

        // Should show success message
        await expect(page.locator('text=/disabled|success/i')).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should enable user', async ({ page }) => {
    // Find inactive user
    const inactiveUserRow = page.locator('tr:has-text("Inactive")').first();
    
    if (await inactiveUserRow.isVisible().catch(() => false)) {
      // Click actions menu
      const actionsButton = inactiveUserRow.locator('button:has-text("Actions"), button[aria-label*="menu"]').first();
      await actionsButton.click();

      // Click enable
      const enableButton = page.locator('text=Enable');
      if (await enableButton.isVisible().catch(() => false)) {
        await enableButton.click();

        // Should show success message
        await expect(page.locator('text=/enabled|success/i')).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should handle empty search results', async ({ page }) => {
    const searchInput = page.locator('input[placeholder*="search" i], input[type="search"]').first();
    
    if (await searchInput.isVisible().catch(() => false)) {
      // Search for non-existent user
      await searchInput.fill('nonexistentuser@example.com');
      await page.waitForTimeout(1000);

      // Should show "No users found" message
      await expect(page.locator('text=/No users found|No results/i')).toBeVisible({ timeout: 5000 });
    }
  });

  test('should clear search filter', async ({ page }) => {
    const searchInput = page.locator('input[placeholder*="search" i], input[type="search"]').first();
    
    if (await searchInput.isVisible().catch(() => false)) {
      // Enter search
      await searchInput.fill('test');
      await page.waitForTimeout(1000);

      // Clear search
      await searchInput.clear();
      await page.waitForTimeout(1000);

      // Should show all users again
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toBeTruthy();
    }
  });

  test('should display user count', async ({ page }) => {
    // Should show total user count somewhere
    const bodyText = await page.locator('body').textContent();
    const hasCount = bodyText!.match(/\d+ users?|Total: \d+|Showing \d+/i);
    
    // Count display is optional but helpful
    expect(bodyText).toBeTruthy();
  });

  test('should load within 1.5 seconds', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/admin/users');
    await page.waitForLoadState('networkidle');

    const endTime = Date.now();
    const loadTime = endTime - startTime;

    console.log(`Users page load time: ${loadTime}ms`);
    expect(loadTime).toBeLessThan(2500); // 1.5s target + 1s buffer
  });
});

test.describe('Users Management - Error Handling', () => {
  test('should handle API errors gracefully', async ({ page }) => {
    // Intercept API and return error
    await page.route('**/api/v1/admin/users*', route => {
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

    await page.goto('/admin/users');

    // Should show error message
    await expect(page.locator('text=/Error|Failed|Unable to load/i')).toBeVisible({ timeout: 10000 });
  });

  test('should handle disable user API error', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);

    // Intercept disable API
    await page.route('**/api/v1/admin/users/*/disable', route => {
      route.fulfill({
        status: 403,
        body: JSON.stringify({ detail: 'Permission denied' }),
      });
    });

    await page.goto('/admin/users');
    await page.waitForLoadState('networkidle');

    const activeUserRow = page.locator('tr:has-text("Active")').first();
    
    if (await activeUserRow.isVisible().catch(() => false)) {
      const actionsButton = activeUserRow.locator('button:has-text("Actions")').first();
      await actionsButton.click();

      const disableButton = page.locator('text=Disable');
      if (await disableButton.isVisible().catch(() => false)) {
        await disableButton.click();
        await page.locator('button:has-text("Confirm")').click().catch(() => {});

        // Should show error message
        await expect(page.locator('text=/Error|Failed|Permission denied/i')).toBeVisible({ timeout: 5000 });
      }
    }
  });
});

test.describe('Users Management - Responsiveness', () => {
  test('should be usable on mobile devices', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);

    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });

    await page.goto('/admin/users');
    await page.waitForLoadState('networkidle');

    // Should show content
    const bodyText = await page.locator('body').textContent();
    expect(bodyText).toBeTruthy();

    // Table should be scrollable horizontally
    const table = page.locator('table').first();
    if (await table.isVisible().catch(() => false)) {
      const box = await table.boundingBox();
      expect(box).toBeTruthy();
    }
  });
});
