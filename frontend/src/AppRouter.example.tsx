/**
 * React App Router with Code Splitting
 * 
 * This is an example of how to structure your App.tsx
 * for optimal code splitting and lazy loading.
 */

import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';

// Components
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import MainLayout from '@/components/layouts/MainLayout';
import AdminLayout from '@/components/layouts/AdminLayout';
import ProtectedRoute from '@/components/auth/ProtectedRoute';

// ==================== EAGER LOADED (CRITICAL) ====================

// Home & Auth - loaded immediately
import Home from '@/pages/Home';
import Login from '@/pages/Login';
import Register from '@/pages/Register';
import NotFound from '@/pages/NotFound';

// ==================== LAZY LOADED (NON-CRITICAL) ====================

// Admin routes - lazy loaded
const AdminDashboard = lazy(() => 
  import('@/pages/admin/Dashboard').then(m => ({ default: m.Dashboard }))
);
const AdminRoles = lazy(() => 
  import('@/pages/admin/Roles').then(m => ({ default: m.RolesList }))
);
const AdminRoleDetail = lazy(() => 
  import('@/pages/admin/RoleDetail').then(m => ({ default: m.RoleDetail }))
);
const AdminSettings = lazy(() => 
  import('@/pages/admin/Settings').then(m => ({ default: m.SettingsList }))
);
const AdminAuditLogs = lazy(() => 
  import('@/pages/admin/AuditLogs').then(m => ({ default: m.AuditLogs }))
);
const AdminUsers = lazy(() => 
  import('@/pages/admin/Users').then(m => ({ default: m.UsersList }))
);
const AdminUserDetail = lazy(() => 
  import('@/pages/admin/UserDetail').then(m => ({ default: m.UserDetail }))
);

// Memory routes - lazy loaded
const MemoryList = lazy(() => 
  import('@/pages/memories/List').then(m => ({ default: m.MemoryList }))
);
const MemoryDetail = lazy(() => 
  import('@/pages/memories/Detail').then(m => ({ default: m.MemoryDetail }))
);
const MemoryCreate = lazy(() => 
  import('@/pages/memories/Create').then(m => ({ default: m.MemoryCreate }))
);
const MemoryEdit = lazy(() => 
  import('@/pages/memories/Edit').then(m => ({ default: m.MemoryEdit }))
);

// Other feature routes
const Dashboard = lazy(() => 
  import('@/pages/Dashboard').then(m => ({ default: m.Dashboard }))
);
const Profile = lazy(() => 
  import('@/pages/Profile').then(m => ({ default: m.Profile }))
);
const Settings = lazy(() => 
  import('@/pages/Settings').then(m => ({ default: m.Settings }))
);

// ==================== SUSPENSE WRAPPER ====================

function SuspenseWrapper({ children }: { children: React.ReactNode }) {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      {children}
    </Suspense>
  );
}

// ==================== APP ROUTES ====================

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* PUBLIC ROUTES */}
        <Route path="/" element={<Home />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />

        {/* PROTECTED ROUTES */}
        <Route 
          element={
            <ProtectedRoute>
              <MainLayout />
            </ProtectedRoute>
          }
        >
          <Route 
            path="/dashboard" 
            element={
              <SuspenseWrapper>
                <Dashboard />
              </SuspenseWrapper>
            } 
          />
          
          <Route 
            path="/memories" 
            element={
              <SuspenseWrapper>
                <MemoryList />
              </SuspenseWrapper>
            } 
          />
          <Route 
            path="/memories/:id" 
            element={
              <SuspenseWrapper>
                <MemoryDetail />
              </SuspenseWrapper>
            } 
          />
          <Route 
            path="/memories/new" 
            element={
              <SuspenseWrapper>
                <MemoryCreate />
              </SuspenseWrapper>
            } 
          />
          <Route 
            path="/memories/:id/edit" 
            element={
              <SuspenseWrapper>
                <MemoryEdit />
              </SuspenseWrapper>
            } 
          />
          
          <Route 
            path="/profile" 
            element={
              <SuspenseWrapper>
                <Profile />
              </SuspenseWrapper>
            } 
          />
          <Route 
            path="/settings" 
            element={
              <SuspenseWrapper>
                <Settings />
              </SuspenseWrapper>
            } 
          />
        </Route>

        {/* ADMIN ROUTES - PROTECTED */}
        <Route 
          element={
            <ProtectedRoute requiredRole="admin">
              <AdminLayout />
            </ProtectedRoute>
          }
        >
          <Route 
            path="/admin" 
            element={
              <SuspenseWrapper>
                <AdminDashboard />
              </SuspenseWrapper>
            } 
          />
          
          <Route 
            path="/admin/roles" 
            element={
              <SuspenseWrapper>
                <AdminRoles />
              </SuspenseWrapper>
            } 
          />
          <Route 
            path="/admin/roles/:id" 
            element={
              <SuspenseWrapper>
                <AdminRoleDetail />
              </SuspenseWrapper>
            } 
          />
          
          <Route 
            path="/admin/settings" 
            element={
              <SuspenseWrapper>
                <AdminSettings />
              </SuspenseWrapper>
            } 
          />
          
          <Route 
            path="/admin/audit-logs" 
            element={
              <SuspenseWrapper>
                <AdminAuditLogs />
              </SuspenseWrapper>
            } 
          />
          
          <Route 
            path="/admin/users" 
            element={
              <SuspenseWrapper>
                <AdminUsers />
              </SuspenseWrapper>
            } 
          />
          <Route 
            path="/admin/users/:id" 
            element={
              <SuspenseWrapper>
                <AdminUserDetail />
              </SuspenseWrapper>
            } 
          />
        </Route>

        {/* 404 FALLBACK */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}

/**
 * Benefits of this approach:
 * 
 * 1. CRITICAL PATH OPTIMIZATION
 *    - Only Home, Login, Register loaded on initial page load
 *    - ~50% reduction in initial bundle size
 * 
 * 2. FEATURE-BASED CHUNKING
 *    - Admin features in separate chunk (loaded on /admin routes)
 *    - Memory features in separate chunk (loaded on /memories routes)
 *    - Each chunk loaded only when needed
 * 
 * 3. PREDICTABLE LOADING STATES
 *    - SuspenseWrapper shows LoadingSpinner while route component loads
 *    - User sees consistent loading UI
 * 
 * 4. BETTER CACHING
 *    - Vite/webpack auto-generates deterministic chunk hashes
 *    - Chunks that don't change keep same filename
 *    - Browser caches unchanged chunks across deployments
 * 
 * 5. MAINTAINABILITY
 *    - Clear route structure
 *    - Easy to add new routes
 *    - Obvious which components are critical vs non-critical
 */
