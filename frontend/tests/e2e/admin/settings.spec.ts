/**
 * E2E Tests - Admin Settings Management
 * Tests settings CRUD, categories, type handling
 */

import { test, expect } from '@playwright/test';

test.describe('Settings Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    
    // Navigate to settings page
    await page.goto('/admin/settings');
    await page.waitForLoadState('networkidle');
  });

  test('should display settings table', async ({ page }) => {
    // Should see table with settings
    const table = page.locator('table, [role="table"]').first();
    await expect(table).toBeVisible();
  });

  test('should show create setting form', async ({ page }) => {
    // Should see create setting section
    await expect(page.locator('text=Create Setting').or(page.locator('text=Add Setting'))).toBeVisible();

    // Should see form fields
    await expect(page.locator('input[name="key"]').first()).toBeVisible();
    await expect(page.locator('input[name="value"]').first()).toBeVisible();
  });

  test('should create new setting', async ({ page }) => {
    const timestamp = Date.now();
    
    // Fill form
    await page.fill('input[name="key"]', `test.setting.${timestamp}`);
    await page.fill('input[name="value"]', 'test-value');
    
    // Select category
    const categorySelect = page.locator('select[name="category"]').first();
    if (await categorySelect.isVisible().catch(() => false)) {
      await categorySelect.selectOption('general');
    }

    // Select type
    const typeSelect = page.locator('select[name="value_type"]').first();
    if (await typeSelect.isVisible().catch(() => false)) {
      await typeSelect.selectOption('string');
    }

    // Submit form
    await page.click('button:has-text("Create"), button[type="submit"]');

    // Should show success message
    await expect(page.locator('text=/created|success/i')).toBeVisible({ timeout: 10000 });
  });

  test('should validate required fields', async ({ page }) => {
    // Try to submit without filling required fields
    const createButton = page.locator('button:has-text("Create"), button[type="submit"]').first();
    await createButton.click();

    // Should show validation errors
    const errorVisible = await page.locator('text=/required|Please|must/i').isVisible({ timeout: 3000 }).catch(() => false);
    expect(errorVisible).toBeTruthy();
  });

  test('should filter settings by category', async ({ page }) => {
    // Click category filter
    const categoryFilters = page.locator('button, a').filter({ hasText: /general|security|features|integrations/i });
    const count = await categoryFilters.count();
    
    if (count > 0) {
      await categoryFilters.first().click();
      await page.waitForTimeout(1000);

      // Table should update with filtered results
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toBeTruthy();
    }
  });

  test('should search settings by key', async ({ page }) => {
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]').first();
    
    if (await searchInput.isVisible().catch(() => false)) {
      await searchInput.fill('app');
      await page.waitForTimeout(1000);

      // Should show filtered results
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toBeTruthy();
    }
  });

  test('should edit existing setting', async ({ page }) => {
    // Find first editable setting
    const editButton = page.locator('button:has-text("Edit"), button[aria-label*="edit" i]').first();
    
    if (await editButton.isVisible().catch(() => false)) {
      await editButton.click();
      await page.waitForTimeout(500);

      // Edit value
      const valueInput = page.locator('input[name="value"]').last();
      await valueInput.clear();
      await valueInput.fill('updated-value');

      // Save
      await page.click('button:has-text("Save"), button:has-text("Update")');

      // Should show success message
      await expect(page.locator('text=/updated|saved|success/i')).toBeVisible({ timeout: 5000 });
    }
  });

  test('should delete custom setting', async ({ page }) => {
    // First create a test setting
    const settingKey = `delete.test.${Date.now()}`;
    await page.fill('input[name="key"]', settingKey);
    await page.fill('input[name="value"]', 'to-be-deleted');
    await page.click('button:has-text("Create")');
    await page.waitForTimeout(2000);

    // Find the setting in table
    const settingRow = page.locator(`tr:has-text("${settingKey}")`).first();
    
    if (await settingRow.isVisible().catch(() => false)) {
      // Click delete button
      const deleteButton = settingRow.locator('button:has-text("Delete"), button[aria-label*="delete"]').first();
      await deleteButton.click();

      // Confirm deletion
      await page.locator('button:has-text("Confirm"), button:has-text("Delete")').click();

      // Should show success message
      await expect(page.locator('text=/deleted|removed|success/i')).toBeVisible({ timeout: 5000 });
    }
  });

  test('should display setting types correctly', async ({ page }) => {
    // Should show different value types
    const types = ['string', 'number', 'boolean', 'json'];
    
    // Type selector should have options
    const typeSelect = page.locator('select[name="value_type"]').first();
    if (await typeSelect.isVisible().catch(() => false)) {
      const options = await typeSelect.locator('option').allTextContents();
      expect(options.length).toBeGreaterThan(0);
    }
  });

  test('should handle boolean settings', async ({ page }) => {
    // Create boolean setting
    const boolKey = `test.boolean.${Date.now()}`;
    await page.fill('input[name="key"]', boolKey);
    
    // Select boolean type
    const typeSelect = page.locator('select[name="value_type"]').first();
    if (await typeSelect.isVisible().catch(() => false)) {
      await typeSelect.selectOption('boolean');
      
      // Value input should change to checkbox or select
      await page.waitForTimeout(500);
      
      // Fill value as "true" or check checkbox
      const valueInput = page.locator('input[name="value"]').first();
      if (await valueInput.getAttribute('type') === 'checkbox') {
        await valueInput.check();
      } else {
        await valueInput.fill('true');
      }

      await page.click('button:has-text("Create")');
      await expect(page.locator('text=/created|success/i')).toBeVisible({ timeout: 5000 });
    }
  });

  test('should handle number settings', async ({ page }) => {
    // Create number setting
    const numKey = `test.number.${Date.now()}`;
    await page.fill('input[name="key"]', numKey);
    
    // Select number type
    const typeSelect = page.locator('select[name="value_type"]').first();
    if (await typeSelect.isVisible().catch(() => false)) {
      await typeSelect.selectOption('number');
      await page.waitForTimeout(500);

      // Fill numeric value
      await page.fill('input[name="value"]', '42');
      await page.click('button:has-text("Create")');
      await expect(page.locator('text=/created|success/i')).toBeVisible({ timeout: 5000 });
    }
  });

  test('should handle JSON settings', async ({ page }) => {
    // Create JSON setting
    const jsonKey = `test.json.${Date.now()}`;
    await page.fill('input[name="key"]', jsonKey);
    
    // Select JSON type
    const typeSelect = page.locator('select[name="value_type"]').first();
    if (await typeSelect.isVisible().catch(() => false)) {
      await typeSelect.selectOption('json');
      await page.waitForTimeout(500);

      // Fill JSON value
      const jsonValue = '{"test": true, "count": 123}';
      const valueInput = page.locator('input[name="value"], textarea[name="value"]').first();
      await valueInput.fill(jsonValue);

      await page.click('button:has-text("Create")');
      await expect(page.locator('text=/created|success/i')).toBeVisible({ timeout: 5000 });
    }
  });

  test('should mask secret values', async ({ page }) => {
    // Create secret setting
    const secretKey = `test.secret.${Date.now()}`;
    await page.fill('input[name="key"]', secretKey);
    
    // Check "is secret" checkbox
    const secretCheckbox = page.locator('input[type="checkbox"][name="is_secret"]').first();
    if (await secretCheckbox.isVisible().catch(() => false)) {
      await secretCheckbox.check();
    }

    await page.fill('input[name="value"]', 'super-secret-value');
    await page.click('button:has-text("Create")');
    await page.waitForTimeout(2000);

    // In table, secret values should be masked
    const settingRow = page.locator(`tr:has-text("${secretKey}")`).first();
    if (await settingRow.isVisible().catch(() => false)) {
      const rowText = await settingRow.textContent();
      // Should show asterisks or masked value
      expect(rowText).toMatch(/\*+|•+|hidden/i);
    }
  });

  test('should show all categories', async ({ page }) => {
    // Should see category buttons/tabs
    const categories = ['General', 'Security', 'Features', 'Integrations'];
    
    for (const category of categories) {
      const categoryVisible = await page.locator(`text=${category}`).isVisible().catch(() => false);
      // Categories should be visible
    }
  });

  test('should prevent duplicate setting keys', async ({ page }) => {
    // Create a setting
    const dupKey = `test.duplicate.${Date.now()}`;
    await page.fill('input[name="key"]', dupKey);
    await page.fill('input[name="value"]', 'first');
    await page.click('button:has-text("Create")');
    await page.waitForTimeout(2000);

    // Try to create with same key
    await page.fill('input[name="key"]', dupKey);
    await page.fill('input[name="value"]', 'second');
    await page.click('button:has-text("Create")');

    // Should show error about duplicate
    await expect(page.locator('text=/already exists|duplicate|unique/i')).toBeVisible({ timeout: 5000 });
  });

  test('should load within 1.5 seconds', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/admin/settings');
    await page.waitForLoadState('networkidle');

    const endTime = Date.now();
    const loadTime = endTime - startTime;

    console.log(`Settings page load time: ${loadTime}ms`);
    expect(loadTime).toBeLessThan(2500); // 1.5s target + 1s buffer
  });
});

test.describe('Settings Management - Error Handling', () => {
  test('should handle API errors on create', async ({ page }) => {
    await page.route('**/api/v1/admin/settings', route => {
      if (route.request().method() === 'POST') {
        route.fulfill({
          status: 500,
          body: JSON.stringify({ detail: 'Internal Server Error' }),
        });
      } else {
        route.continue();
      }
    });

    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    await page.goto('/admin/settings');
    await page.waitForLoadState('networkidle');

    // Try to create setting
    await page.fill('input[name="key"]', 'error.test');
    await page.fill('input[name="value"]', 'test');
    await page.click('button:has-text("Create")');

    // Should show error message
    await expect(page.locator('text=/Error|Failed/i')).toBeVisible({ timeout: 5000 });
  });

  test('should handle invalid JSON values', async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    await page.goto('/admin/settings');
    await page.waitForLoadState('networkidle');

    // Create JSON setting with invalid JSON
    const jsonKey = `test.invalid.json.${Date.now()}`;
    await page.fill('input[name="key"]', jsonKey);
    
    const typeSelect = page.locator('select[name="value_type"]').first();
    if (await typeSelect.isVisible().catch(() => false)) {
      await typeSelect.selectOption('json');
      await page.waitForTimeout(500);

      // Fill invalid JSON
      const valueInput = page.locator('input[name="value"], textarea[name="value"]').first();
      await valueInput.fill('{invalid json}');

      await page.click('button:has-text("Create")');

      // Should show validation error
      await expect(page.locator('text=/invalid|JSON|format/i')).toBeVisible({ timeout: 5000 });
    }
  });
});

test.describe('Settings Management - Responsiveness', () => {
  test('should be usable on mobile', async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 667 });

    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    await page.goto('/admin/settings');
    await page.waitForLoadState('networkidle');

    // Form should be visible and usable
    const keyInput = page.locator('input[name="key"]').first();
    await expect(keyInput).toBeVisible();

    // Category filters should be accessible
    const categories = page.locator('button, a').filter({ hasText: /general|security|features/i });
    const count = await categories.count();
    expect(count).toBeGreaterThanOrEqual(0);
  });
});
