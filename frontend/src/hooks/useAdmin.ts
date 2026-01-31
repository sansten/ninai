/**
 * Admin API Hooks - useAdmin
 * Provides access to admin context and user data
 */

import { createContext, useContext } from 'react';
import { User } from '../types/auth';

interface AdminContextType {
  adminUser: User | null;
  permissions: Set<string>;
  hasPermission: (permission: string) => boolean;
  hasAnyPermission: (permissions: string[]) => boolean;
  hasAllPermissions: (permissions: string[]) => boolean;
}

const AdminContext = createContext<AdminContextType | undefined>(undefined);

export const useAdmin = (): AdminContextType => {
  const context = useContext(AdminContext);
  if (!context) {
    throw new Error('useAdmin must be used within AdminProvider');
  }
  return context;
};

export { AdminContext };
