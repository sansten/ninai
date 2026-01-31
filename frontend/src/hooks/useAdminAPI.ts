/**
 * Admin API Hooks - Data Fetching & Management
 * React Query hooks for admin operations
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '../services/api';
import {
  AdminRole, AdminSetting, AdminAuditLog, User as AdminUser,
  DashboardData, AdminIPWhitelist, Permission
} from '../types/admin';

// ==================== DASHBOARD HOOKS ====================

export const useDashboard = () => {
  return useQuery({
    queryKey: ['admin', 'dashboard'],
    queryFn: async () => {
      const response = await apiClient.get<DashboardData>('/admin/dashboard');
      return response.data;
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });
};

// ==================== ROLES HOOKS ====================

export const useRoles = (page = 1, limit = 50) => {
  return useQuery({
    queryKey: ['admin', 'roles', page, limit],
    queryFn: async () => {
      const response = await apiClient.get<AdminRole[]>('/admin/roles', {
        params: { skip: (page - 1) * limit, limit },
      });
      return response.data;
    },
  });
};

export const useRole = (roleId: string) => {
  return useQuery({
    queryKey: ['admin', 'roles', roleId],
    queryFn: async () => {
      const response = await apiClient.get<AdminRole>(`/admin/roles/${roleId}`);
      return response.data;
    },
    enabled: !!roleId,
  });
};

export const useCreateRole = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: Partial<AdminRole>) => {
      const response = await apiClient.post<AdminRole>('/admin/roles', data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'roles'] });
    },
  });
};

export const useUpdateRole = (roleId: string) => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: Partial<AdminRole>) => {
      const response = await apiClient.put<AdminRole>(`/admin/roles/${roleId}`, data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'roles'] });
      queryClient.invalidateQueries({ queryKey: ['admin', 'roles', roleId] });
    },
  });
};

export const useDeleteRole = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (roleId: string) => {
      await apiClient.delete(`/admin/roles/${roleId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'roles'] });
    },
  });
};

// ==================== SETTINGS HOOKS ====================

export const useSettings = (category?: string, page = 1, limit = 50) => {
  return useQuery({
    queryKey: ['admin', 'settings', category, page, limit],
    queryFn: async () => {
      const response = await apiClient.get<{ items: AdminSetting[]; total: number }>('/admin/settings', {
        params: { category, skip: (page - 1) * limit, limit },
      });
      return response.data;
    },
  });
};

export const useSetting = (settingId: string) => {
  return useQuery({
    queryKey: ['admin', 'settings', settingId],
    queryFn: async () => {
      const response = await apiClient.get<AdminSetting>(`/admin/settings/${settingId}`);
      return response.data;
    },
    enabled: !!settingId,
  });
};

export const useCreateSetting = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: Partial<AdminSetting>) => {
      const response = await apiClient.post<AdminSetting>('/admin/settings', data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'settings'] });
    },
  });
};

export const useUpdateSetting = (settingId: string) => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: Partial<AdminSetting>) => {
      const response = await apiClient.put<AdminSetting>(`/admin/settings/${settingId}`, data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'settings'] });
      queryClient.invalidateQueries({ queryKey: ['admin', 'settings', settingId] });
    },
  });
};

export const useDeleteSetting = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (settingId: string) => {
      await apiClient.delete(`/admin/settings/${settingId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'settings'] });
    },
  });
};

// ==================== USERS HOOKS ====================

export const useUsers = (search?: string, roleId?: string, page = 1, limit = 50) => {
  return useQuery({
    queryKey: ['admin', 'users', search, roleId, page, limit],
    queryFn: async () => {
      const response = await apiClient.get<{ items: AdminUser[]; total: number }>('/admin/users', {
        params: { search, role_id: roleId, skip: (page - 1) * limit, limit },
      });
      return response.data;
    },
  });
};

export const useUser = (userId: string) => {
  return useQuery({
    queryKey: ['admin', 'users', userId],
    queryFn: async () => {
      const response = await apiClient.get<AdminUser>(`/admin/users/${userId}`);
      return response.data;
    },
    enabled: !!userId,
  });
};

export const useDisableUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (userId: string) => {
      const response = await apiClient.post<AdminUser>(`/admin/users/${userId}/disable`, {});
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'users'] });
    },
  });
};

export const useEnableUser = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (userId: string) => {
      const response = await apiClient.post<AdminUser>(`/admin/users/${userId}/enable`, {});
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'users'] });
    },
  });
};

// ==================== AUDIT LOGS HOOKS ====================

export const useAuditLogs = (
  adminId?: string,
  action?: string,
  resourceType?: string,
  page = 1,
  limit = 50
) => {
  return useQuery({
    queryKey: ['admin', 'audit-logs', adminId, action, resourceType, page, limit],
    queryFn: async () => {
      const response = await apiClient.get<{ items: AdminAuditLog[]; total: number }>(
        '/admin/audit-logs',
        {
          params: {
            admin_id: adminId,
            action,
            resource_type: resourceType,
            skip: (page - 1) * limit,
            limit,
          },
        }
      );
      return response.data;
    },
  });
};

export const useAuditLog = (logId: string) => {
  return useQuery({
    queryKey: ['admin', 'audit-logs', logId],
    queryFn: async () => {
      const response = await apiClient.get<AdminAuditLog>(`/admin/audit-logs/${logId}`);
      return response.data;
    },
    enabled: !!logId,
  });
};

// ==================== PERMISSIONS HOOKS ====================

export const usePermissions = () => {
  return useQuery({
    queryKey: ['admin', 'permissions'],
    queryFn: async () => {
      const response = await apiClient.get<{ permissions: Permission[] }>('/admin/permissions');
      return response.data.permissions;
    },
  });
};

// ==================== IP WHITELIST HOOKS ====================

export const useIPWhitelist = () => {
  return useQuery({
    queryKey: ['admin', 'ip-whitelist'],
    queryFn: async () => {
      const response = await apiClient.get<AdminIPWhitelist[]>('/admin/ip-whitelist');
      return response.data;
    },
  });
};

export const useAddIPWhitelist = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (data: { ip_address: string; description?: string }) => {
      const response = await apiClient.post<AdminIPWhitelist>('/admin/ip-whitelist', data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'ip-whitelist'] });
    },
  });
};

export const useRemoveIPWhitelist = () => {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: async (ipAddress: string) => {
      await apiClient.delete(`/admin/ip-whitelist/${ipAddress}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin', 'ip-whitelist'] });
    },
  });
};
