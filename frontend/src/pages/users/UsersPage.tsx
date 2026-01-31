/**
 * Users Page
 * ==========
 * 
 * User listing and management.
 */

import { useMemo, useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import {
  MagnifyingGlassIcon,
  PlusIcon,
  UsersIcon,
  ShieldCheckIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';
import { Navigate } from 'react-router-dom';
import { apiClient, getErrorMessage, PaginatedResponse } from '@/lib/api';
import { useCurrentOrg, useIsAdmin } from '@/stores/auth';
import type { User, UserRole } from '@/types/api';

type Role = {
  id: string;
  name: string;
  description: string | null;
  permissions: string[];
  is_system_role: boolean;
};

type UserWithRolesResponse = User & {
  roles: UserRole[];
};

/**
 * User Row Component
 */
interface UserRowProps {
  user: User;
  onToggleActive: (user: User) => void;
  onManageRoles: (user: User) => void;
  onToggleReviewer: (user: User) => void;
  reviewerButtonDisabled: boolean;
  isReviewer: boolean;
}

function UserRow({ user, onToggleActive, onManageRoles, onToggleReviewer, reviewerButtonDisabled, isReviewer }: UserRowProps) {
  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap">
        <div className="flex items-center">
          <div className="h-10 w-10 flex-shrink-0">
            {user.avatar_url ? (
              <img
                className="h-10 w-10 rounded-full"
                src={user.avatar_url}
                alt=""
              />
            ) : (
              <div className="h-10 w-10 rounded-full bg-primary-100 flex items-center justify-center">
                <span className="text-primary-600 font-medium">
                  {user.display_name?.charAt(0).toUpperCase() || 'U'}
                </span>
              </div>
            )}
          </div>
          <div className="ml-4">
            <div className="text-sm font-medium text-gray-900">
              {user.display_name}
            </div>
            <div className="text-sm text-gray-500">{user.email}</div>
          </div>
        </div>
      </td>
      <td className="px-6 py-4 whitespace-nowrap">
        <span className={user.is_active ? 'badge-success' : 'badge-gray'}>
          {user.is_active ? 'Active' : 'Inactive'}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {new Date(user.created_at).toLocaleDateString()}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
        <button
          onClick={() => onToggleReviewer(user)}
          disabled={reviewerButtonDisabled}
          className={
            reviewerButtonDisabled
              ? 'text-gray-400 cursor-not-allowed mr-4'
              : 'text-primary-600 hover:text-primary-900 mr-4'
          }
          title={
            reviewerButtonDisabled
              ? 'Loading role catalog…'
              : isReviewer
                ? 'Remove knowledge_reviewer'
                : 'Assign knowledge_reviewer'
          }
        >
          <ShieldCheckIcon className="h-5 w-5 inline" />
          <span className="ml-1">{isReviewer ? 'Remove Reviewer' : 'Assign Reviewer'}</span>
        </button>
        <button
          onClick={() => onManageRoles(user)}
          className="text-primary-600 hover:text-primary-900 mr-4"
        >
          <ShieldCheckIcon className="h-5 w-5 inline" />
          <span className="ml-1">Roles</span>
        </button>
        <button
          onClick={() => onToggleActive(user)}
          className={user.is_active ? 'text-red-600 hover:text-red-900' : 'text-green-600 hover:text-green-900'}
        >
          {user.is_active ? (
            <>
              <XCircleIcon className="h-5 w-5 inline" />
              <span className="ml-1">Deactivate</span>
            </>
          ) : (
            <>
              <ShieldCheckIcon className="h-5 w-5 inline" />
              <span className="ml-1">Activate</span>
            </>
          )}
        </button>
      </td>
    </tr>
  );
}

/**
 * Users Page Component
 */
export function UsersPage() {
  const org = useCurrentOrg();
  const queryClient = useQueryClient();
  const isAdmin = useIsAdmin();
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [rolesUser, setRolesUser] = useState<User | null>(null);

  const canLoad = Boolean(isAdmin && org?.id);

  // Fetch users
  const { data, isLoading } = useQuery<PaginatedResponse<User>>({
    queryKey: ['users', org?.id ?? 'no-org', search, page],
    enabled: canLoad,
    queryFn: async () => {
      const response = await apiClient.get('/users', {
        params: { search, page, page_size: 20 },
      });
      return response.data;
    },
  });

  const rolesCatalogQuery = useQuery<Role[]>({
    queryKey: ['users', 'roles-catalog'],
    enabled: canLoad,
    queryFn: async () => {
      const res = await apiClient.get('/users/roles');
      return res.data;
    },
  });

  const userRolesQuery = useQuery<UserWithRolesResponse>({
    queryKey: ['users', rolesUser?.id, 'roles'],
    enabled: canLoad && !!rolesUser?.id,
    queryFn: async () => {
      const res = await apiClient.get(`/users/${rolesUser!.id}`);
      return res.data;
    },
  });

  const assignedRoleIds = useMemo(() => {
    const roles = userRolesQuery.data?.roles ?? [];
    return new Set(roles.filter((r) => r.is_active).map((r) => r.role_id));
  }, [userRolesQuery.data]);

  const knowledgeReviewerRoleId = useMemo(() => {
    return rolesCatalogQuery.data?.find((r) => r.name === 'knowledge_reviewer')?.id ?? null;
  }, [rolesCatalogQuery.data]);

  // Toggle active mutation
  const toggleMutation = useMutation({
    mutationFn: async ({ userId, isActive }: { userId: string; isActive: boolean }) => {
      if (isActive) {
        await apiClient.delete(`/users/${userId}`);
      } else {
        await apiClient.patch(`/users/${userId}`, { is_active: true });
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      toast.success('User status updated');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const handleToggleActive = (user: User) => {
    const action = user.is_active ? 'deactivate' : 'activate';
    if (confirm(`Are you sure you want to ${action} "${user.display_name}"?`)) {
      toggleMutation.mutate({ userId: user.id, isActive: user.is_active });
    }
  };

  const handleManageRoles = (_user: User) => {
    setRolesUser(_user);
  };

  const quickAssignReviewerMutation = useMutation({
    mutationFn: async ({ userId, roleId }: { userId: string; roleId: string }) => {
      await apiClient.post(`/users/${userId}/roles`, {
        role_id: roleId,
        scope_type: 'organization',
        scope_id: null,
        granted_reason: 'Assigned for knowledge review',
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      toast.success('Reviewer role assigned');
    },
    onError: (err: unknown) => toast.error(getErrorMessage(err)),
  });

  const handleAssignReviewer = (user: User) => {
    if (!knowledgeReviewerRoleId) {
      toast.error('Roles catalog not loaded (missing knowledge_reviewer)');
      return;
    }
    quickAssignReviewerMutation.mutate({ userId: user.id, roleId: knowledgeReviewerRoleId });
  };

  const quickRevokeReviewerMutation = useMutation({
    mutationFn: async ({ userId, assignmentId }: { userId: string; assignmentId: string }) => {
      await apiClient.delete(`/users/${userId}/roles/${assignmentId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
      toast.success('Reviewer role removed');
    },
    onError: (err: unknown) => toast.error(getErrorMessage(err)),
  });

  const handleToggleReviewer = (user: User) => {
    const roleNames = user.role_names ?? [];
    const isReviewer = roleNames.includes('knowledge_reviewer');
    if (!knowledgeReviewerRoleId) {
      toast.error('Roles catalog not loaded (missing knowledge_reviewer)');
      return;
    }

    if (!isReviewer) {
      handleAssignReviewer(user);
      return;
    }

    const assignmentId = user.role_assignment_ids?.knowledge_reviewer;
    if (!assignmentId) {
      toast('Open Roles to revoke (missing assignment id)');
      setRolesUser(user);
      return;
    }
    quickRevokeReviewerMutation.mutate({ userId: user.id, assignmentId });
  };

  const assignRoleMutation = useMutation({
    mutationFn: async (roleId: string) => {
      if (!rolesUser) return;
      await apiClient.post(`/users/${rolesUser.id}/roles`, {
        role_id: roleId,
        scope_type: 'organization',
        scope_id: null,
        granted_reason: 'Assigned for knowledge review',
      });
    },
    onSuccess: () => {
      userRolesQuery.refetch();
      toast.success('Role assigned');
    },
    onError: (err: unknown) => toast.error(getErrorMessage(err)),
  });

  const revokeRoleMutation = useMutation({
    mutationFn: async (roleId: string) => {
      if (!rolesUser) return;
      const assignment = (userRolesQuery.data?.roles ?? []).find((r) => r.role_id === roleId && r.is_active);
      if (!assignment) return;
      await apiClient.delete(`/users/${rolesUser.id}/roles/${assignment.id}`);
    },
    onSuccess: () => {
      userRolesQuery.refetch();
      toast.success('Role revoked');
    },
    onError: (err: unknown) => toast.error(getErrorMessage(err)),
  });

  if (!isAdmin) {
    return <Navigate to="/dashboard" replace />;
  }

  if (!org?.id) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Users</h1>
          <p className="text-gray-500 mt-1">
            Manage users and their permissions
          </p>
        </div>
        <button className="btn-primary">
          <PlusIcon className="h-5 w-5 mr-2" />
          Invite User
        </button>
      </div>

      {/* Search */}
      <div className="card">
        <div className="relative">
          <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
          <input
            type="text"
            placeholder="Search users..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            className="input pl-10"
          />
        </div>
      </div>

      {/* Users Table */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      ) : data?.items?.length ? (
        <div className="card p-0 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  User
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Joined
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {data.items.map((user) => (
                <UserRow
                  key={user.id}
                  user={user}
                  onToggleActive={handleToggleActive}
                  onManageRoles={handleManageRoles}
                  onToggleReviewer={handleToggleReviewer}
                  reviewerButtonDisabled={
                    !knowledgeReviewerRoleId ||
                    quickAssignReviewerMutation.isPending ||
                    quickRevokeReviewerMutation.isPending
                  }
                  isReviewer={(user.role_names ?? []).includes('knowledge_reviewer')}
                />
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          {data.pages > 1 && (
            <div className="px-6 py-4 border-t border-gray-200 flex items-center justify-between">
              <p className="text-sm text-gray-500">
                Showing {(page - 1) * 20 + 1} to {Math.min(page * 20, data.total)} of {data.total} users
              </p>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage(page - 1)}
                  disabled={page === 1}
                  className="btn-secondary"
                >
                  Previous
                </button>
                <button
                  onClick={() => setPage(page + 1)}
                  disabled={page >= data.pages}
                  className="btn-secondary"
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </div>
      ) : (
        <div className="text-center py-12">
          <UsersIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-semibold text-gray-900">No users found</h3>
          <p className="mt-1 text-sm text-gray-500">
            {search ? 'Try a different search term.' : 'Invite users to get started.'}
          </p>
        </div>
      )}

      {/* Roles Modal */}
      {rolesUser && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-2xl rounded-lg bg-white shadow-xl">
            <div className="flex items-center justify-between border-b px-6 py-4">
              <div>
                <h2 className="text-lg font-semibold text-gray-900">Manage roles</h2>
                <p className="text-sm text-gray-500 mt-1">{rolesUser.email}</p>
              </div>
              <button className="btn-secondary" onClick={() => setRolesUser(null)}>
                Close
              </button>
            </div>

            <div className="px-6 py-4 space-y-4">
              <div className="card bg-amber-50 border border-amber-200">
                <p className="text-sm text-amber-900 font-medium">Reviewer flow</p>
                <p className="mt-1 text-sm text-amber-800">
                  Assign <span className="font-medium">knowledge_reviewer</span> to allow non-admin users to access the Review Queue
                  and approve/reject knowledge submissions.
                </p>
              </div>

              {(rolesCatalogQuery.isLoading || userRolesQuery.isLoading) && (
                <div className="text-sm text-gray-500">Loading roles…</div>
              )}

              {(rolesCatalogQuery.isError || userRolesQuery.isError) && (
                <div className="text-sm text-red-600">Failed to load roles</div>
              )}

              {rolesCatalogQuery.data && (
                <div className="space-y-2">
                  {rolesCatalogQuery.data.map((role) => {
                    const checked = assignedRoleIds.has(role.id);
                    const busy = assignRoleMutation.isPending || revokeRoleMutation.isPending;
                    return (
                      <label key={role.id} className="flex items-start gap-3 rounded-lg border p-3">
                        <input
                          type="checkbox"
                          className="mt-1"
                          disabled={busy}
                          checked={checked}
                          onChange={() => {
                            if (checked) revokeRoleMutation.mutate(role.id);
                            else assignRoleMutation.mutate(role.id);
                          }}
                        />
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-gray-900">{role.name}</span>
                            {role.name === 'knowledge_reviewer' && <span className="badge-primary">Recommended</span>}
                            {role.is_system_role && <span className="badge-gray">System</span>}
                          </div>
                          {role.description && <p className="text-sm text-gray-600 mt-1">{role.description}</p>}
                        </div>
                      </label>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
