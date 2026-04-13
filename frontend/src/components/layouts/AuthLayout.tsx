/**
 * Auth Layout
 * ===========
 * 
 * Layout for authentication pages (login, register, etc.)
 */

import { Outlet, Navigate } from 'react-router-dom';
import { useAuthStore } from '@/stores/auth';

export function AuthLayout() {
  const { isAuthenticated, isLoading } = useAuthStore();

  // Show loading while checking auth
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  // Redirect to dashboard if already authenticated
  if (isAuthenticated) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="min-h-screen flex flex-col justify-center py-12 sm:px-6 lg:px-8 bg-gray-50">
      {/* Logo / Brand */}
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <h2 className="ninai mt-6 text-center">
          NINAI
        </h2>
        <p className="cos text-center">
          Cognitive Operating System
        </p>
      </div>

      {/* Auth form container */}
      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow-lg sm:rounded-xl sm:px-10 border border-gray-100">
          <Outlet />
        </div>
      </div>

      {/* Footer */}
      <div className="mt-8 text-center text-sm text-gray-500">
        <p>&copy; {new Date().getFullYear()} <span className="font-semibold text-black">NINAI</span>. All rights reserved.</p>
      </div>
    </div>
  );
}
