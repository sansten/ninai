/**
 * Admin UI - Main Layout Component
 * Provides the base layout for all admin pages with sidebar, header, and breadcrumbs
 */

import React, { useState } from 'react';
import { Navigate, Outlet, useNavigate, useLocation } from 'react-router-dom';
import {
  Menu,
  ChevronDown,
  LogOut,
  Settings,
  User as UserIcon,
  BarChart3,
  Users,
  Shield,
  Sliders,
  FileText,
  AlertTriangle,
} from 'lucide-react';
import { useAuth } from '../../hooks/useAuth';
import { useAdmin } from '../../hooks/useAdmin';
import { cn } from '../../lib/utils';

interface NavItem {
  icon: React.ReactNode;
  label: string;
  path: string;
  permission: string;
}

const AdminLayout: React.FC = () => {
  const { user, logout } = useAuth();
  const { adminUser } = useAdmin();
  const navigate = useNavigate();
  const location = useLocation();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [userMenuOpen, setUserMenuOpen] = useState(false);

  // Redirect if not admin
  if (!user?.is_admin) {
    return <Navigate to="/login" replace />;
  }

  const navItems: NavItem[] = [
    {
      icon: <BarChart3 className="w-5 h-5" />,
      label: 'Dashboard',
      path: '/admin',
      permission: 'system:read',
    },
    {
      icon: <Users className="w-5 h-5" />,
      label: 'Users',
      path: '/admin/users',
      permission: 'users:read',
    },
    {
      icon: <Shield className="w-5 h-5" />,
      label: 'Roles',
      path: '/admin/roles',
      permission: 'roles:read',
    },
    {
      icon: <Sliders className="w-5 h-5" />,
      label: 'Settings',
      path: '/admin/settings',
      permission: 'settings:read',
    },
    {
      icon: <FileText className="w-5 h-5" />,
      label: 'Audit Logs',
      path: '/admin/audit-logs',
      permission: 'audit:read',
    },
  ];

  const handleLogout = async () => {
    await logout();
    navigate('/login');
  };

  const hasPermission = (permission: string): boolean => {
    // Check if adminUser has the required permission
    if (!adminUser?.permissions) return false;
    return adminUser.permissions.includes(permission);
  };

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <div
        className={cn(
          'bg-gray-900 text-white transition-all duration-300 flex flex-col',
          sidebarOpen ? 'w-64' : 'w-20'
        )}
      >
        {/* Logo */}
        <div className="flex items-center justify-between h-16 px-4 border-b border-gray-800">
          {sidebarOpen && <h1 className="text-xl font-bold">NINAI Admin</h1>}
          <button
            onClick={() => setSidebarOpen(!sidebarOpen)}
            className="p-2 hover:bg-gray-800 rounded-lg"
          >
            <Menu className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-2 py-4 space-y-2">
          {navItems.map((item) => (
            hasPermission(item.permission) && (
              <button
                key={item.path}
                onClick={() => navigate(item.path)}
                className={cn(
                  'w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors',
                  'hover:bg-gray-800 text-gray-300 hover:text-white',
                  location.pathname === item.path && 'bg-blue-600 text-white'
                )}
              >
                {item.icon}
                {sidebarOpen && <span>{item.label}</span>}
              </button>
            )
          ))}
        </nav>

        {/* Version Info */}
        {sidebarOpen && (
          <div className="px-4 py-4 border-t border-gray-800 text-xs text-gray-500">
            <div>NINAI v1.0</div>
            <div>Admin UI</div>
          </div>
        )}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-8 py-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Admin Dashboard</h2>
          </div>

          {/* User Menu */}
          <div className="relative">
            <button
              onClick={() => setUserMenuOpen(!userMenuOpen)}
              className="flex items-center space-x-2 px-4 py-2 rounded-lg hover:bg-gray-100"
            >
              <UserIcon className="w-5 h-5" />
              <span className="text-sm font-medium">{user?.full_name}</span>
              <ChevronDown className="w-4 h-4" />
            </button>

            {userMenuOpen && (
              <div className="absolute right-0 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
                <div className="px-4 py-2 border-b border-gray-200">
                  <p className="text-sm font-medium text-gray-900">{user?.email}</p>
                </div>
                <button
                  onClick={() => navigate('/admin/settings')}
                  className="w-full text-left px-4 py-2 hover:bg-gray-100 flex items-center space-x-2"
                >
                  <Settings className="w-4 h-4" />
                  <span className="text-sm">Settings</span>
                </button>
                <button
                  onClick={handleLogout}
                  className="w-full text-left px-4 py-2 hover:bg-gray-100 text-red-600 flex items-center space-x-2"
                >
                  <LogOut className="w-4 h-4" />
                  <span className="text-sm">Logout</span>
                </button>
              </div>
            )}
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-auto">
          <div className="p-8">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
};

export default AdminLayout;
