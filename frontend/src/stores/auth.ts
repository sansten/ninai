/**
 * Auth Store
 * ==========
 * 
 * Zustand store for authentication state management.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';

const memoryStorage = (() => {
  const store: Record<string, string> = {};
  return {
    getItem: (name: string) => store[name] ?? null,
    setItem: (name: string, value: string) => {
      store[name] = value;
    },
    removeItem: (name: string) => {
      delete store[name];
    },
  };
})();

function getSafeStorage(): Storage | typeof memoryStorage {
  try {
    if (typeof localStorage !== 'undefined') {
      const testKey = '__ninai_storage_test__';
      localStorage.setItem(testKey, '1');
      localStorage.removeItem(testKey);
      return localStorage;
    }
  } catch {
    // Fallback to in-memory storage
  }
  return memoryStorage;
}

/**
 * User type
 */
export interface User {
  id: string;
  email: string;
  display_name: string;
  avatar_url?: string;
  is_active: boolean;
  created_at: string;
  roles: string[];
}

/**
 * Organization type
 */
export interface Organization {
  id: string;
  name: string;
  slug: string;
  tier?: string;
  description?: string;
  settings?: Record<string, unknown>;
  is_active?: boolean;
  parent_org_id?: string;
  created_at?: string;
  updated_at?: string;
}

/**
 * Auth state interface
 */
interface AuthState {
  // State
  user: User | null;
  accessToken: string | null;
  refreshToken: string | null;
  currentOrg: Organization | null;
  availableOrgs: Organization[];
  isAuthenticated: boolean;
  isLoading: boolean;
  
  // Actions
  setUser: (user: User) => void;
  setTokens: (accessToken: string, refreshToken: string) => void;
  setCurrentOrg: (org: Organization) => void;
  setAvailableOrgs: (orgs: Organization[]) => void;
  setLoading: (loading: boolean) => void;
  login: (user: User, accessToken: string, refreshToken: string, org: Organization) => void;
  logout: () => void;
  switchOrg: (org: Organization) => void;
}

/**
 * Auth store with persistence
 */
export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      // Initial state
      user: null,
      accessToken: null,
      refreshToken: null,
      currentOrg: null,
      availableOrgs: [],
      isAuthenticated: false,
      isLoading: true,

      // Actions
      setUser: (user) => set({ user }),
      
      setTokens: (accessToken, refreshToken) => 
        set({ accessToken, refreshToken, isAuthenticated: true }),
      
      setCurrentOrg: (org) => set({ currentOrg: org }),
      
      setAvailableOrgs: (orgs) => set({ availableOrgs: orgs }),
      
      setLoading: (loading) => set({ isLoading: loading }),
      
      login: (user, accessToken, refreshToken, org) => 
        set({
          user,
          accessToken,
          refreshToken,
          currentOrg: org,
          isAuthenticated: true,
          isLoading: false,
        }),
      
      logout: () => 
        set({
          user: null,
          accessToken: null,
          refreshToken: null,
          currentOrg: null,
          availableOrgs: [],
          isAuthenticated: false,
          isLoading: false,
        }),
      
      switchOrg: (org) => set({ currentOrg: org }),
    }),
    {
      name: 'ninai-auth',
      storage: createJSONStorage(() => getSafeStorage()),
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        user: state.user,
        currentOrg: state.currentOrg,
        availableOrgs: state.availableOrgs,
      }),
      onRehydrateStorage: () => (state) => {
        // Set isAuthenticated based on stored token
        if (state) {
          const hasSession = !!state.accessToken && !!state.currentOrg && !!state.user;
          state.isAuthenticated = hasSession;
          state.isLoading = false;

          if (!hasSession) {
            state.user = null;
            state.accessToken = null;
            state.refreshToken = null;
            state.currentOrg = null;
            state.availableOrgs = [];
          }
        }
      },
    }
  )
);

/**
 * Get current user or throw
 */
export function useCurrentUser(): User {
  const user = useAuthStore((state) => state.user);
  if (!user) {
    throw new Error('User not authenticated');
  }
  return user;
}

export function useHasRole(role: string): boolean {
  const roles = useAuthStore((state) => state.user?.roles ?? []);
  return roles.includes(role);
}

export function useIsSystemAdmin(): boolean {
  return useHasRole('system_admin');
}

export function useIsAdmin(): boolean {
  const roles = useAuthStore((state) => state.user?.roles ?? []);
  return roles.includes('org_admin') || roles.includes('system_admin');
}

export function useIsKnowledgeReviewer(): boolean {
  const roles = useAuthStore((state) => state.user?.roles ?? []);
  return roles.includes('knowledge_reviewer') || roles.includes('org_admin') || roles.includes('system_admin');
}

export function useCanViewAudit(): boolean {
  const roles = useAuthStore((state) => state.user?.roles ?? []);
  return roles.includes('org_admin') || roles.includes('security_admin') || roles.includes('system_admin');
}

/**
 * Get current org or throw
 */
export function useCurrentOrg(): Organization {
  const org = useAuthStore((state) => state.currentOrg);
  if (!org) {
    throw new Error('No organization selected');
  }
  return org;
}
