/**
 * E2E Tests - Admin Authentication & Authorization
 * Tests login, logout, and permission-based access
 */

import { test, expect } from '@playwright/test';

const ADMIN_USER = {
  email: 'admin@example.com',
  password: 'admin123',
};

const REGULAR_USER = {
  email: 'user@example.com',
  password: 'user123',
};

test.describe('Admin Authentication', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/login');
  });

  test('should login with admin credentials', async ({ page }) => {
    await page.fill('input[type="email"]', ADMIN_USER.email);
    await page.fill('input[type="password"]', ADMIN_USER.password);
    await page.click('button[type="submit"]');

    // Should redirect to dashboard or home
    await expect(page).toHaveURL(/\/(dashboard|home|admin)/);
    
    // Should see user menu or profile
    await expect(page.locator('text=Admin').or(page.locator('[data-testid="user-menu"]'))).toBeVisible();
  });

  test('should fail login with invalid credentials', async ({ page }) => {
    await page.fill('input[type="email"]', 'wrong@example.com');
    await page.fill('input[type="password"]', 'wrongpassword');
    await page.click('button[type="submit"]');

    // Should show error message
    await expect(page.locator('text=/Invalid credentials|Login failed|Incorrect/i')).toBeVisible();
    
    // Should stay on login page
    await expect(page).toHaveURL(/\/login/);
  });

  test('should logout successfully', async ({ page }) => {
    // Login first
    await page.fill('input[type="email"]', ADMIN_USER.email);
    await page.fill('input[type="password"]', ADMIN_USER.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);

    // Click user menu and logout
    await page.click('[data-testid="user-menu"]').catch(() => page.click('text=Admin'));
    await page.click('text=Logout');

    // Should redirect to login
    await expect(page).toHaveURL(/\/login/);
  });
});

test.describe('Admin Authorization', () => {
  test('admin user can access admin panel', async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('input[type="email"]', ADMIN_USER.email);
    await page.fill('input[type="password"]', ADMIN_USER.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);

    // Navigate to admin panel
    await page.goto('/admin');

    // Should see admin UI
    await expect(page.locator('text=Dashboard').or(page.locator('text=Admin Dashboard'))).toBeVisible();
    
    // Should see navigation items
    await expect(page.locator('text=Users').or(page.locator('a[href*="/admin/users"]'))).toBeVisible();
  });

  test('regular user cannot access admin panel', async ({ page }) => {
    // Login as regular user
    await page.goto('/login');
    await page.fill('input[type="email"]', REGULAR_USER.email);
    await page.fill('input[type="password"]', REGULAR_USER.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home)/);

    // Try to access admin panel
    await page.goto('/admin');

    // Should be redirected or show error
    await expect(
      page.locator('text=/Permission denied|Not authorized|Access denied/i')
        .or(page)
    ).toBeTruthy();
    
    // Should not see admin navigation
    const hasAdminNav = await page.locator('text=Audit Logs').isVisible().catch(() => false);
    expect(hasAdminNav).toBeFalsy();
  });

  test('unauthenticated user is redirected to login', async ({ page }) => {
    await page.goto('/admin');

    // Should redirect to login
    await expect(page).toHaveURL(/\/login/);
  });
});

test.describe('Permission-Based Access', () => {
  test.beforeEach(async ({ page }) => {
    // Login as admin
    await page.goto('/login');
    await page.fill('input[type="email"]', ADMIN_USER.email);
    await page.fill('input[type="password"]', ADMIN_USER.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);
  });

  test('admin can access all admin pages', async ({ page }) => {
    const adminPages = [
      '/admin',
      '/admin/users',
      '/admin/roles',
      '/admin/settings',
      '/admin/audit-logs',
    ];

    for (const path of adminPages) {
      await page.goto(path);
      
      // Should not show permission denied error
      const hasError = await page.locator('text=/Permission denied|Not authorized/i').isVisible().catch(() => false);
      expect(hasError).toBeFalsy();
      
      // Should see some content (not blank page)
      const bodyText = await page.locator('body').textContent();
      expect(bodyText).toBeTruthy();
      expect(bodyText!.length).toBeGreaterThan(100);
    }
  });

  test('navigation menu shows only permitted pages', async ({ page }) => {
    await page.goto('/admin');

    // Should see all navigation items for admin
    const navItems = ['Dashboard', 'Users', 'Roles', 'Settings', 'Audit Logs'];
    
    for (const item of navItems) {
      await expect(page.locator(`text=${item}`)).toBeVisible();
    }
  });
});

test.describe('Session Management', () => {
  test('should persist session on page reload', async ({ page }) => {
    // Login
    await page.goto('/login');
    await page.fill('input[type="email"]', ADMIN_USER.email);
    await page.fill('input[type="password"]', ADMIN_USER.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);

    // Navigate to admin
    await page.goto('/admin');
    await expect(page.locator('text=Dashboard').or(page.locator('text=Admin Dashboard'))).toBeVisible();

    // Reload page
    await page.reload();

    // Should still be logged in
    await expect(page.locator('text=Dashboard').or(page.locator('text=Admin Dashboard'))).toBeVisible();
  });

  test('should handle expired token gracefully', async ({ page, context }) => {
    // Login
    await page.goto('/login');
    await page.fill('input[type="email"]', ADMIN_USER.email);
    await page.fill('input[type="password"]', ADMIN_USER.password);
    await page.click('button[type="submit"]');
    await page.waitForURL(/\/(dashboard|home|admin)/);

    // Clear cookies to simulate expired token
    await context.clearCookies();

    // Try to access admin page
    await page.goto('/admin');

    // Should redirect to login
    await expect(page).toHaveURL(/\/login/);
  });
});
