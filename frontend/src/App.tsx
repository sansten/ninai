import { useEffect, lazy, Suspense, type ReactNode } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/auth';
import { apiClient } from '@/lib/api';

// Layouts (keep non-lazy as they're used on every route)
import { DashboardLayout } from '@/components/layouts/DashboardLayout';
import { AuthLayout } from '@/components/layouts/AuthLayout';

// Critical pages (loaded immediately)
import { LoginPage } from '@/pages/auth/LoginPage';
import { OidcCallbackPage } from '@/pages/auth/OidcCallbackPage';
import { DashboardPage } from '@/pages/dashboard/DashboardPage';

// Lazy-loaded pages (code splitting for better performance)
const MemoriesPage = lazy(() => import('@/pages/memories/MemoriesPage').then(m => ({ default: m.MemoriesPage })));
const MemoryDetailPage = lazy(() => import('@/pages/memories/MemoryDetailPage').then(m => ({ default: m.MemoryDetailPage })));
const TeamsPage = lazy(() => import('@/pages/teams/TeamsPage').then(m => ({ default: m.TeamsPage })));
const UsersPage = lazy(() => import('@/pages/users/UsersPage').then(m => ({ default: m.UsersPage })));
const AuditPage = lazy(() => import('@/pages/audit/AuditPage').then(m => ({ default: m.AuditPage })));
const SettingsPage = lazy(() => import('@/pages/settings/SettingsPage').then(m => ({ default: m.SettingsPage })));
const KnowledgeReviewPage = lazy(() => import('@/pages/review/KnowledgeReviewPage').then(m => ({ default: m.KnowledgeReviewPage })));

// Admin pages (lazy loaded)
const AdminDashboard = lazy(() => import('@/pages/admin/Dashboard'));
const AdminUsers = lazy(() => import('@/pages/admin/Users'));
const AdminRoles = lazy(() => import('@/pages/admin/Roles'));
const AdminSettings = lazy(() => import('@/pages/admin/Settings'));
const AdminAuditLogs = lazy(() => import('@/pages/admin/AuditLogs'));

/**
 * Loading fallback component
 */
function LoadingFallback() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
    </div>
  );
}

/**
 * Protected route wrapper
 */
function ProtectedRoute({ children }: { children: ReactNode }) {
  const { isAuthenticated, isLoading } = useAuthStore();

  useEffect(() => {
    let cancelled = false;

    async function hydrateRolesIfMissing() {
      const { accessToken, user, currentOrg, setUser } = useAuthStore.getState();
      if (!accessToken || !user || !currentOrg) return;
      if (Array.isArray(user.roles) && user.roles.length > 0) return;

      try {
        const res = await apiClient.get('/auth/me');
        const me = res.data;

        if (!cancelled) {
          setUser({
            id: me.id,
            email: me.email,
            display_name: me.full_name,
            avatar_url: me.avatar_url,
            is_active: me.is_active,
            created_at: me.created_at,
            roles: Array.isArray(me.roles) ? me.roles : [],
          });
        }
      } catch {
        // Non-fatal; user can still use the app. Admin tab may remain hidden.
      }
    }

    if (isAuthenticated && !isLoading) {
      hydrateRolesIfMissing();
    }

    return () => {
      cancelled = true;
    };
  }, [isAuthenticated, isLoading]);

  if (isLoading) {
    return <LoadingFallback />;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}

/**
 * Main App Component
 */
export default function App() {
  return (
    <Routes>
      {/* Auth routes */}
      <Route element={<AuthLayout />}>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/auth/oidc/callback" element={<OidcCallbackPage />} />
      </Route>

      {/* Protected dashboard routes */}
      <Route
        element={
          <ProtectedRoute>
            <DashboardLayout />
          </ProtectedRoute>
        }
      >
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="/dashboard" element={<DashboardPage />} />
        <Route path="/memories" element={<Suspense fallback={<LoadingFallback />}><MemoriesPage /></Suspense>} />
        <Route path="/memories/:id" element={<Suspense fallback={<LoadingFallback />}><MemoryDetailPage /></Suspense>} />
        <Route path="/teams" element={<Suspense fallback={<LoadingFallback />}><TeamsPage /></Suspense>} />
        <Route path="/users" element={<Suspense fallback={<LoadingFallback />}><UsersPage /></Suspense>} />
        <Route path="/audit" element={<Suspense fallback={<LoadingFallback />}><AuditPage /></Suspense>} />
        <Route path="/review" element={<Suspense fallback={<LoadingFallback />}><KnowledgeReviewPage /></Suspense>} />
        <Route path="/settings" element={<Suspense fallback={<LoadingFallback />}><SettingsPage /></Suspense>} />
        
        {/* Admin routes - lazy loaded */}
        <Route path="/admin" element={<Navigate to="/admin/dashboard" replace />} />
        <Route path="/admin/dashboard" element={<Suspense fallback={<LoadingFallback />}><AdminDashboard /></Suspense>} />
        <Route path="/admin/users" element={<Suspense fallback={<LoadingFallback />}><AdminUsers /></Suspense>} />
        <Route path="/admin/roles" element={<Suspense fallback={<LoadingFallback />}><AdminRoles /></Suspense>} />
        <Route path="/admin/settings" element={<Suspense fallback={<LoadingFallback />}><AdminSettings /></Suspense>} />
        <Route path="/admin/audit-logs" element={<Suspense fallback={<LoadingFallback />}><AdminAuditLogs /></Suspense>} />
      </Route>

      {/* Catch-all redirect */}
      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
}
