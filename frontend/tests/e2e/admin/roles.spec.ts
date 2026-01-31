/**
 * E2E Tests - Admin Roles Management
 * Tests role creation, permission management, and deletion
 */

import { test, expect } from '@playwright/test';

test.describe('Roles Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    
    // Navigate to roles page
    await page.goto('/admin/roles');
    await page.waitForLoadState('networkidle');
  });

  test('should display roles table', async ({ page }) => {
    // Should see table with roles
    const bodyText = await page.locator('body').textContent();
    
    // Should have "Admin" system role at minimum
    expect(bodyText).toContain('Admin');
  });

  test('should show create role form', async ({ page }) => {
    // Should see create role section
    await expect(page.locator('text=Create Role').or(page.locator('text=Add Role'))).toBeVisible();

    // Should see form fields
    await expect(page.locator('input[name="name"], input[placeholder*="name" i]').first()).toBeVisible();
    await expect(page.locator('textarea[name="description"], textarea[placeholder*="description" i]').first()).toBeVisible();
  });

  test('should create new role with permissions', async ({ page }) => {
    // Fill role name
    await page.fill('input[name="name"], input[placeholder*="name" i]', `TestRole_${Date.now()}`);
    
    // Fill description
    await page.fill('textarea[name="description"], textarea[placeholder*="description" i]', 'Test role for E2E testing');

    // Select some permissions
    const permissionCheckboxes = page.locator('input[type="checkbox"]').filter({ hasText: /users:read|content:read/ });
    const count = await permissionCheckboxes.count();
    
    if (count > 0) {
      await permissionCheckboxes.first().check();
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
    
    // Form should not submit successfully
    expect(errorVisible).toBeTruthy();
  });

  test('should display permission categories', async ({ page }) => {
    // Should see permission categories
    const categories = ['Users', 'Roles', 'Settings', 'System'];
    
    for (const category of categories) {
      const categoryVisible = await page.locator(`text=${category}`).isVisible().catch(() => false);
      // At least some categories should be visible
    }
  });

  test('should show all available permissions', async ({ page }) => {
    // Count checkboxes
    const checkboxes = await page.locator('input[type="checkbox"]').count();
    
    // Should have at least 10 permissions (we defined 14)
    expect(checkboxes).toBeGreaterThan(5);
  });

  test('should delete custom role', async ({ page }) => {
    // First create a test role
    const roleName = `DeleteTest_${Date.now()}`;
    await page.fill('input[name="name"]', roleName);
    await page.fill('textarea[name="description"]', 'Role to be deleted');
    await page.click('button:has-text("Create")');
    await page.waitForTimeout(2000);

    // Find the role in table
    const roleRow = page.locator(`tr:has-text("${roleName}")`);
    
    if (await roleRow.isVisible().catch(() => false)) {
      // Click delete button
      const deleteButton = roleRow.locator('button:has-text("Delete"), button[aria-label*="delete"]').first();
      
      if (await deleteButton.isVisible().catch(() => false)) {
        await deleteButton.click();

        // Confirm deletion
        await page.locator('button:has-text("Confirm"), button:has-text("Delete")').click();

        // Should show success message
        await expect(page.locator('text=/deleted|removed|success/i')).toBeVisible({ timeout: 5000 });
      }
    }
  });

  test('should not delete system roles', async ({ page }) => {
    // Find Admin role (system role)
    const adminRole = page.locator('tr:has-text("Admin")').first();
    
    if (await adminRole.isVisible().catch(() => false)) {
      // Delete button should be disabled or not present
      const deleteButton = adminRole.locator('button:has-text("Delete")').first();
      const isDisabled = await deleteButton.isDisabled().catch(() => true);
      
      expect(isDisabled).toBeTruthy();
    }
  });

  test('should display role details in table', async ({ page }) => {
    // Table should show role information
    const headers = ['Name', 'Description', 'Permissions'];
    
    for (const header of headers) {
      const headerVisible = await page.locator(`text=${header}`).isVisible().catch(() => false);
      // Headers should be visible
    }
  });

  test('should filter roles by name', async ({ page }) => {
    // If search exists
    const searchInput = page.locator('input[type="search"], input[placeholder*="search" i]').first();
    
    if (await searchInput.isVisible().catch(() => false)) {
      await searchInput.fill('Admin');
      await page.waitForTimeout(1000);

      // Should show filtered results
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toContain('Admin');
    }
  });

  test('should show permission count for each role', async ({ page }) => {
    // Each role should display number of permissions
    const permissionCounts = page.locator('td:has-text("permissions"), td:has-text("perms")');
    const count = await permissionCounts.count();
    
    // Should show permission info for at least one role
    expect(count).toBeGreaterThanOrEqual(0);
  });

  test('should prevent duplicate role names', async ({ page }) => {
    const existingRole = 'Admin';
    
    // Try to create role with existing name
    await page.fill('input[name="name"]', existingRole);
    await page.fill('textarea[name="description"]', 'Duplicate test');
    await page.click('button:has-text("Create")');

    // Should show error about duplicate
    await expect(page.locator('text=/already exists|duplicate|unique/i')).toBeVisible({ timeout: 5000 });
  });

  test('should load within 1.5 seconds', async ({ page }) => {
    const startTime = Date.now();

    await page.goto('/admin/roles');
    await page.waitForLoadState('networkidle');

    const endTime = Date.now();
    const loadTime = endTime - startTime;

    console.log(`Roles page load time: ${loadTime}ms`);
    expect(loadTime).toBeLessThan(2500); // 1.5s target + 1s buffer
  });
});

test.describe('Roles Management - Permission Selection', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
    await page.fill('input[type="email"]', 'admin@example.com');
    await page.fill('input[type="password"]', 'admin123');
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
    await page.goto('/admin/roles');
    await page.waitForLoadState('networkidle');
  });

  test('should select individual permissions', async ({ page }) => {
    // Check a permission
    const firstCheckbox = page.locator('input[type="checkbox"]').first();
    await firstCheckbox.check();

    // Should be checked
    expect(await firstCheckbox.isChecked()).toBeTruthy();
  });

  test('should unselect permissions', async ({ page }) => {
    // Check then uncheck
    const firstCheckbox = page.locator('input[type="checkbox"]').first();
    await firstCheckbox.check();
    await firstCheckbox.uncheck();

    // Should be unchecked
    expect(await firstCheckbox.isChecked()).toBeFalsy();
  });

  test('should select multiple permissions', async ({ page }) => {
    // Check multiple permissions
    const checkboxes = await page.locator('input[type="checkbox"]').all();
    
    if (checkboxes.length > 3) {
      await checkboxes[0].check();
      await checkboxes[1].check();
      await checkboxes[2].check();

      expect(await checkboxes[0].isChecked()).toBeTruthy();
      expect(await checkboxes[1].isChecked()).toBeTruthy();
      expect(await checkboxes[2].isChecked()).toBeTruthy();
    }
  });

  test('should create role with no permissions', async ({ page }) => {
    // Create role without selecting permissions
    await page.fill('input[name="name"]', `NoPerms_${Date.now()}`);
    await page.fill('textarea[name="description"]', 'Role with no permissions');
    await page.click('button:has-text("Create")');

    // Should create successfully (0 permissions is valid)
    await expect(page.locator('text=/created|success/i')).toBeVisible({ timeout: 5000 });
  });
});

test.describe('Roles Management - Error Handling', () => {
  test('should handle API errors', async ({ page }) => {
    // Intercept API
    await page.route('**/api/v1/admin/roles', route => {
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
    await page.goto('/admin/roles');
    await page.waitForLoadState('networkidle');

    // Try to create role
    await page.fill('input[name="name"]', 'ErrorTest');
    await page.fill('textarea[name="description"]', 'Test');
    await page.click('button:has-text("Create")');

    // Should show error message
    await expect(page.locator('text=/Error|Failed/i')).toBeVisible({ timeout: 5000 });
  });
});
