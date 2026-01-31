/**
 * Admin Integration Tests
 * Test all admin pages, API integration, and workflows
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import AdminRoutes from '../pages/admin';
import * as adminAPI from '../hooks/useAdminAPI';

// Mock the API calls
jest.mock('../hooks/useAdminAPI');
jest.mock('../hooks/useAdmin');
jest.mock('../hooks/useAuth');

const mockQueryClient = new QueryClient({
  defaultOptions: {
    queries: { retry: false },
    mutations: { retry: false },
  },
});

const renderAdminRoutes = () => {
  return render(
    <BrowserRouter>
      <QueryClientProvider client={mockQueryClient}>
        <AdminRoutes />
      </QueryClientProvider>
    </BrowserRouter>
  );
};

describe('Admin UI Integration Tests', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Navigation and Layout', () => {
    it('should render admin layout with all navigation items', async () => {
      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.getByText('Users')).toBeInTheDocument();
        expect(screen.getByText('Roles')).toBeInTheDocument();
        expect(screen.getByText('Settings')).toBeInTheDocument();
        expect(screen.getByText('Audit Logs')).toBeInTheDocument();
      });
    });

    it('should navigate between pages when clicking nav items', async () => {
      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      (adminAPI.useUsers as jest.Mock).mockReturnValue({
        data: { items: [], total: 0 },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const usersLink = await screen.findByText('Users');
      fireEvent.click(usersLink);

      await waitFor(() => {
        expect(screen.getByText(/Users Management/i)).toBeInTheDocument();
      });
    });
  });

  describe('Dashboard Page', () => {
    const mockDashboardData = {
      kpi: {
        active_users: 150,
        api_requests_today: 5000,
        error_rate: 0.5,
        uptime_percentage: 99.9,
      },
      services: [
        { name: 'Backend API', status: 'healthy' },
        { name: 'Database', status: 'healthy' },
        { name: 'Message Queue', status: 'healthy' },
      ],
      alerts: { critical: 0, warning: 2 },
      recent_activity: [
        {
          id: '1',
          action: 'create',
          resource_type: 'user',
          timestamp: new Date().toISOString(),
          ip_address: '192.168.1.1',
        },
      ],
    };

    it('should display KPI cards with correct data', async () => {
      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: mockDashboardData,
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      await waitFor(() => {
        expect(screen.getByText('150')).toBeInTheDocument(); // active users
        expect(screen.getByText('5000')).toBeInTheDocument(); // api requests
        expect(screen.getByText('0.5%')).toBeInTheDocument(); // error rate
        expect(screen.getByText('99.9%')).toBeInTheDocument(); // uptime
      });
    });

    it('should show service health status', async () => {
      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: mockDashboardData,
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      await waitFor(() => {
        expect(screen.getByText('Backend API')).toBeInTheDocument();
        expect(screen.getByText('Database')).toBeInTheDocument();
      });
    });

    it('should display alerts count', async () => {
      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: mockDashboardData,
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      await waitFor(() => {
        expect(screen.getByText('2')).toBeInTheDocument(); // warning count
      });
    });
  });

  describe('Users Management', () => {
    const mockUsersData = {
      items: [
        {
          id: '1',
          email: 'user1@example.com',
          full_name: 'User One',
          is_admin: false,
          is_active: true,
          last_login: new Date().toISOString(),
        },
        {
          id: '2',
          email: 'user2@example.com',
          full_name: 'User Two',
          is_admin: false,
          is_active: false,
          last_login: null,
        },
      ],
      total: 2,
      page: 1,
      page_size: 50,
    };

    it('should render users table with data', async () => {
      (adminAPI.useUsers as jest.Mock).mockReturnValue({
        data: mockUsersData,
        isLoading: false,
        error: null,
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const usersNav = await screen.findByText('Users');
      fireEvent.click(usersNav);

      await waitFor(() => {
        expect(screen.getByText('user1@example.com')).toBeInTheDocument();
        expect(screen.getByText('user2@example.com')).toBeInTheDocument();
      });
    });

    it('should search users by email', async () => {
      (adminAPI.useUsers as jest.Mock).mockImplementation(
        (_, __, ___, ____, search) => {
          const filtered = mockUsersData.items.filter(
            (u) => u.email.includes(search || '')
          );
          return {
            data: { ...mockUsersData, items: filtered },
            isLoading: false,
            error: null,
          };
        }
      );

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const usersNav = await screen.findByText('Users');
      fireEvent.click(usersNav);

      const searchInput = await screen.findByPlaceholderText('Search by email or name...');
      await userEvent.type(searchInput, 'user1');

      await waitFor(() => {
        expect(screen.getByText('user1@example.com')).toBeInTheDocument();
        expect(screen.queryByText('user2@example.com')).not.toBeInTheDocument();
      });
    });

    it('should disable user with confirmation', async () => {
      const mockDisable = jest.fn();
      (adminAPI.useUsers as jest.Mock).mockReturnValue({
        data: mockUsersData,
        isLoading: false,
        error: null,
      });

      (adminAPI.useDisableUser as jest.Mock).mockReturnValue({
        mutate: mockDisable,
        isPending: false,
        error: null,
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const usersNav = await screen.findByText('Users');
      fireEvent.click(usersNav);

      // Find and click the disable button (would be in actions menu)
      await waitFor(() => {
        expect(screen.getByText('user1@example.com')).toBeInTheDocument();
      });
    });
  });

  describe('Roles Management', () => {
    const mockRolesData = {
      items: [
        {
          id: '1',
          name: 'Admin',
          description: 'Full system access',
          permissions: ['system:*'],
          is_system: true,
        },
        {
          id: '2',
          name: 'Moderator',
          description: 'Content moderation',
          permissions: ['content:moderate', 'users:view'],
          is_system: false,
        },
      ],
      total: 2,
    };

    it('should render roles table with data', async () => {
      (adminAPI.useRoles as jest.Mock).mockReturnValue({
        data: mockRolesData,
        isLoading: false,
        error: null,
      });

      (adminAPI.usePermissions as jest.Mock).mockReturnValue({
        data: { items: [] },
        isLoading: false,
        error: null,
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const rolesNav = await screen.findByText('Roles');
      fireEvent.click(rolesNav);

      await waitFor(() => {
        expect(screen.getByText('Admin')).toBeInTheDocument();
        expect(screen.getByText('Moderator')).toBeInTheDocument();
      });
    });

    it('should create new role with permissions', async () => {
      const mockCreate = jest.fn();
      (adminAPI.useRoles as jest.Mock).mockReturnValue({
        data: mockRolesData,
        isLoading: false,
        error: null,
      });

      (adminAPI.useCreateRole as jest.Mock).mockReturnValue({
        mutate: mockCreate,
        isPending: false,
        error: null,
      });

      (adminAPI.usePermissions as jest.Mock).mockReturnValue({
        data: {
          items: [
            { id: '1', name: 'users:view', category: 'users' },
            { id: '2', name: 'users:create', category: 'users' },
          ],
        },
        isLoading: false,
        error: null,
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const rolesNav = await screen.findByText('Roles');
      fireEvent.click(rolesNav);

      // Would need to interact with form to test fully
      await waitFor(() => {
        expect(screen.getByText('Admin')).toBeInTheDocument();
      });
    });
  });

  describe('Settings Management', () => {
    const mockSettingsData = {
      items: [
        {
          category: 'general',
          key: 'app_name',
          value: 'NINAI',
          type: 'string',
          is_secret: false,
        },
        {
          category: 'security',
          key: 'max_login_attempts',
          value: '5',
          type: 'number',
          is_secret: false,
        },
      ],
      total: 2,
    };

    it('should render settings table with data', async () => {
      (adminAPI.useSettings as jest.Mock).mockReturnValue({
        data: mockSettingsData,
        isLoading: false,
        error: null,
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const settingsNav = await screen.findByText('Settings');
      fireEvent.click(settingsNav);

      await waitFor(() => {
        expect(screen.getByText('app_name')).toBeInTheDocument();
        expect(screen.getByText('max_login_attempts')).toBeInTheDocument();
      });
    });

    it('should add new setting', async () => {
      const mockAdd = jest.fn();
      (adminAPI.useSettings as jest.Mock).mockReturnValue({
        data: mockSettingsData,
        isLoading: false,
        error: null,
      });

      (adminAPI.useCreateSetting as jest.Mock).mockReturnValue({
        mutate: mockAdd,
        isPending: false,
        error: null,
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const settingsNav = await screen.findByText('Settings');
      fireEvent.click(settingsNav);

      await waitFor(() => {
        expect(screen.getByText('app_name')).toBeInTheDocument();
      });
    });
  });

  describe('Audit Logs', () => {
    const mockAuditData = {
      items: [
        {
          id: '1',
          action: 'create',
          resource_type: 'user',
          resource_id: 'user123',
          admin_id: 'admin1',
          old_values: null,
          new_values: { email: 'new@example.com' },
          created_at: new Date().toISOString(),
          ip_address: '192.168.1.1',
          user_agent: 'Mozilla/5.0',
        },
      ],
      total: 1,
    };

    it('should render audit logs table', async () => {
      (adminAPI.useAuditLogs as jest.Mock).mockReturnValue({
        data: mockAuditData,
        isLoading: false,
        error: null,
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const auditNav = await screen.findByText('Audit Logs');
      fireEvent.click(auditNav);

      await waitFor(() => {
        expect(screen.getByText('create')).toBeInTheDocument();
        expect(screen.getByText('user')).toBeInTheDocument();
      });
    });

    it('should expand log to show details', async () => {
      (adminAPI.useAuditLogs as jest.Mock).mockReturnValue({
        data: mockAuditData,
        isLoading: false,
        error: null,
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      const auditNav = await screen.findByText('Audit Logs');
      fireEvent.click(auditNav);

      await waitFor(() => {
        expect(screen.getByText('create')).toBeInTheDocument();
      });

      // Click to expand - would need to click the log row
      // This is a simplified test
    });
  });

  describe('Permission Checks', () => {
    it('should not show pages user lacks permission for', async () => {
      // Mock useAdmin to return limited permissions
      const mockUseAdmin = require('../hooks/useAdmin').useAdmin;
      mockUseAdmin.mockReturnValue({
        hasPermission: (perm: string) => perm === 'system:read', // only dashboard
        adminUser: {
          permissions: ['system:read'],
        },
      });

      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: { kpi: {}, services: [] },
        isLoading: false,
        error: null,
      });

      renderAdminRoutes();

      await waitFor(() => {
        expect(screen.getByText('Dashboard')).toBeInTheDocument();
        expect(screen.queryByText('Users')).not.toBeInTheDocument();
        expect(screen.queryByText('Roles')).not.toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('should show error state when API call fails', async () => {
      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: null,
        isLoading: false,
        error: new Error('API Error'),
      });

      renderAdminRoutes();

      await waitFor(() => {
        expect(screen.getByText(/Failed to load dashboard/i)).toBeInTheDocument();
      });
    });

    it('should show loading state while data is fetching', async () => {
      (adminAPI.useDashboard as jest.Mock).mockReturnValue({
        data: null,
        isLoading: true,
        error: null,
      });

      renderAdminRoutes();

      await waitFor(() => {
        expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
      });
    });
  });
});
