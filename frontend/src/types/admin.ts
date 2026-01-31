/**
 * Admin Types & Interfaces
 */

export interface AdminRole {
  id: string;
  name: string;
  description?: string;
  permissions: string[];
  is_system: boolean;
  created_at: string;
  updated_at: string;
  created_by?: string;
}

export interface AdminSetting {
  id: string;
  category: string;
  key: string;
  value?: any;
  type?: string;
  description?: string;
  is_secret: boolean;
  updated_at: string;
  updated_by?: string;
}

export interface AdminAuditLog {
  id: string;
  admin_id: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  old_values?: Record<string, any>;
  new_values?: Record<string, any>;
  ip_address?: string;
  user_agent?: string;
  created_at: string;
}

export interface User {
  id: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_admin: boolean;
  admin_role_id?: string;
  admin_notes?: string;
  last_login?: string;
  created_at: string;
  updated_at: string;
  last_admin_action_at?: string;
  last_admin_action_by?: string;
}

export interface AdminIPWhitelist {
  id: string;
  ip_address: string;
  description?: string;
  created_by?: string;
  created_at: string;
}

export interface Permission {
  permission: string;
  description: string;
  category: string;
}

export interface DashboardKPI {
  label: string;
  value: string;
  unit?: string;
  trend?: 'up' | 'down' | 'stable';
  change_percent?: number;
}

export interface ServiceHealthStatus {
  name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  message?: string;
  last_check: string;
}

export interface DashboardActivity {
  id: string;
  admin_id: string;
  action: string;
  resource_type: string;
  resource_id?: string;
  timestamp: string;
  description: string;
  status: string;
}

export interface DashboardKPIGroup {
  users: DashboardKPI[];
  memories: DashboardKPI[];
  system: DashboardKPI[];
}

export interface DashboardData {
  timestamp: string;
  kpis: DashboardKPIGroup;
  services: ServiceHealthStatus[];
  recent_activities: DashboardActivity[];
  alerts?: Array<Record<string, any>>;
}
