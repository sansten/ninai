/**
 * E2E Test Helpers
 * Reusable utilities for Playwright tests
 */

import { Page, expect } from '@playwright/test';

/**
 * Login as admin user
 */
export async function loginAsAdmin(page: Page) {
  await page.goto('/login');
  await page.fill('input[type="email"]', 'admin@example.com');
  await page.fill('input[type="password"]', 'admin123');
  await page.click('button[type="submit"]');
  await page.waitForURL(/\/(dashboard|home|admin)/);
}

/**
 * Login as regular user
 */
export async function loginAsUser(page: Page) {
  await page.goto('/login');
  await page.fill('input[type="email"]', 'user@example.com');
  await page.fill('input[type="password"]', 'user123');
  await page.click('button[type="submit"]');
  await page.waitForURL(/\/(dashboard|home)/);
}

/**
 * Logout current user
 */
export async function logout(page: Page) {
  // Look for logout button in nav or user menu
  const logoutButton = page.locator('button:has-text("Logout"), button:has-text("Sign out"), a:has-text("Logout")');
  
  if (await logoutButton.isVisible().catch(() => false)) {
    await logoutButton.click();
  } else {
    // Try clicking user menu first
    const userMenu = page.locator('[aria-label*="user menu"], button:has-text("admin@")');
    if (await userMenu.isVisible().catch(() => false)) {
      await userMenu.click();
      await page.locator('button:has-text("Logout"), a:has-text("Logout")').click();
    }
  }
  
  await page.waitForURL('/login');
}

/**
 * Wait for API request to complete
 */
export async function waitForAPIResponse(page: Page, endpoint: string, timeout = 10000) {
  return await page.waitForResponse(
    response => response.url().includes(endpoint) && response.status() < 400,
    { timeout }
  );
}

/**
 * Mock API error response
 */
export async function mockAPIError(page: Page, endpoint: string, status = 500, message = 'Internal Server Error') {
  await page.route(`**${endpoint}*`, route => {
    route.fulfill({
      status,
      contentType: 'application/json',
      body: JSON.stringify({ detail: message }),
    });
  });
}

/**
 * Mock successful API response
 */
export async function mockAPISuccess(page: Page, endpoint: string, data: any) {
  await page.route(`**${endpoint}*`, route => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(data),
    });
  });
}

/**
 * Check if element is in viewport
 */
export async function isInViewport(page: Page, selector: string): Promise<boolean> {
  return await page.evaluate((sel) => {
    const element = document.querySelector(sel);
    if (!element) return false;
    
    const rect = element.getBoundingClientRect();
    return (
      rect.top >= 0 &&
      rect.left >= 0 &&
      rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
      rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
  }, selector);
}

/**
 * Scroll element into view
 */
export async function scrollIntoView(page: Page, selector: string) {
  await page.evaluate((sel) => {
    const element = document.querySelector(sel);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, selector);
}

/**
 * Wait for loading spinner to disappear
 */
export async function waitForLoadingComplete(page: Page, timeout = 10000) {
  await page.locator('[class*="loading"], [class*="spinner"], [aria-label*="loading"]').waitFor({
    state: 'hidden',
    timeout,
  }).catch(() => {
    // Ignore if no loading indicator found
  });
}

/**
 * Take screenshot with timestamp
 */
export async function takeScreenshot(page: Page, name: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  await page.screenshot({
    path: `test-results/screenshots/${name}-${timestamp}.png`,
    fullPage: true,
  });
}

/**
 * Measure page load performance
 */
export async function measurePageLoadTime(page: Page): Promise<number> {
  return await page.evaluate(() => {
    const perfData = window.performance.timing;
    return perfData.loadEventEnd - perfData.navigationStart;
  });
}

/**
 * Check for console errors
 */
export async function expectNoConsoleErrors(page: Page) {
  const errors: string[] = [];
  
  page.on('console', msg => {
    if (msg.type() === 'error') {
      errors.push(msg.text());
    }
  });

  page.on('pageerror', error => {
    errors.push(error.message);
  });

  return {
    errors,
    check: () => {
      expect(errors).toHaveLength(0);
    },
  };
}

/**
 * Fill form field by label
 */
export async function fillByLabel(page: Page, label: string, value: string) {
  const input = page.locator(`label:has-text("${label}") + input, label:has-text("${label}") input`);
  await input.fill(value);
}

/**
 * Click button by text (case insensitive)
 */
export async function clickButton(page: Page, text: string) {
  await page.locator(`button:has-text("${text}"), button[aria-label*="${text}"]`).first().click();
}

/**
 * Wait for toast/notification message
 */
export async function waitForToast(page: Page, expectedText?: string, timeout = 5000) {
  const toast = page.locator('[role="alert"], [class*="toast"], [class*="notification"]');
  await toast.waitFor({ state: 'visible', timeout });
  
  if (expectedText) {
    await expect(toast).toContainText(expectedText);
  }
  
  return toast;
}

/**
 * Clear all filters/search inputs on page
 */
export async function clearAllFilters(page: Page) {
  // Clear search inputs
  const searchInputs = await page.locator('input[type="search"]').all();
  for (const input of searchInputs) {
    await input.clear();
  }
  
  // Click clear/reset buttons
  const clearButtons = page.locator('button:has-text("Clear"), button:has-text("Reset")');
  const count = await clearButtons.count();
  
  for (let i = 0; i < count; i++) {
    const button = clearButtons.nth(i);
    if (await button.isVisible()) {
      await button.click();
    }
  }
}

/**
 * Select option from dropdown by text
 */
export async function selectDropdownOption(page: Page, selectSelector: string, optionText: string) {
  const select = page.locator(selectSelector);
  
  if (await select.getAttribute('role') === 'combobox') {
    // Custom dropdown
    await select.click();
    await page.locator(`[role="option"]:has-text("${optionText}")`).click();
  } else {
    // Native select
    await select.selectOption({ label: optionText });
  }
}

/**
 * Check if user has permission (based on UI elements)
 */
export async function hasPermission(page: Page, feature: string): Promise<boolean> {
  // Check if feature link/button is visible
  const featureElement = page.locator(`a:has-text("${feature}"), button:has-text("${feature}")`);
  return await featureElement.isVisible().catch(() => false);
}

/**
 * Navigate to admin page
 */
export async function navigateToAdminPage(page: Page, pageName: 'dashboard' | 'users' | 'roles' | 'settings' | 'audit-logs') {
  await page.goto(`/admin/${pageName}`);
  await page.waitForLoadState('networkidle');
  await waitForLoadingComplete(page);
}

/**
 * Create test data cleanup helper
 */
export function createCleanupHelper() {
  const itemsToCleanup: Array<{ type: string; id: string }> = [];
  
  return {
    add: (type: string, id: string) => {
      itemsToCleanup.push({ type, id });
    },
    cleanup: async (page: Page) => {
      for (const item of itemsToCleanup.reverse()) {
        try {
          // Delete via API or UI
          console.log(`Cleaning up ${item.type} ${item.id}`);
        } catch (error) {
          console.error(`Failed to cleanup ${item.type} ${item.id}:`, error);
        }
      }
    },
  };
}

/**
 * Retry operation until success or max attempts
 */
export async function retryUntil<T>(
  operation: () => Promise<T>,
  condition: (result: T) => boolean,
  maxAttempts = 5,
  delayMs = 1000
): Promise<T> {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    const result = await operation();
    if (condition(result)) {
      return result;
    }
    
    if (attempt < maxAttempts) {
      await new Promise(resolve => setTimeout(resolve, delayMs));
    }
  }
  
  throw new Error(`Operation failed after ${maxAttempts} attempts`);
}
