/**
 * Auth hook — thin adapter over useAuthStore that exposes is_admin derived from roles.
 */

import { useAuthStore } from '../stores/auth';

export function useAuth() {
  const { user, logout, isAuthenticated, isLoading } = useAuthStore();

  const authUser = user
    ? {
        ...user,
        is_admin:
          user.roles.includes('org_admin') ||
          user.roles.includes('system_admin') ||
          user.roles.includes('admin'),
      }
    : null;

  return { user: authUser, logout, isAuthenticated, isLoading };
}
