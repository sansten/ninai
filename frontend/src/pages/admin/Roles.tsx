/**
 * Roles Management Page
 * Create, edit, and manage admin roles
 */

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAdmin } from '../../hooks/useAdmin';
import { useRoles, useCreateRole, useDeleteRole, usePermissions } from '../../hooks/useAdminAPI';
import { Plus, Loader, AlertCircle, Trash2, Edit, Check } from 'lucide-react';
import { cn } from '../../lib/utils';

const Roles: React.FC = () => {
  const navigate = useNavigate();
  const { hasPermission } = useAdmin();
  const [page, setPage] = useState(1);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    permissions: [] as string[],
  });

  const { data: rolesData, isLoading: rolesLoading } = useRoles(page, 50);
  const { data: permissions } = usePermissions();
  const createRoleMutation = useCreateRole();
  const deleteRoleMutation = useDeleteRole();

  if (!hasPermission('roles:read')) {
    return (
      <div className="p-6 bg-red-50 rounded-lg border border-red-200">
        <p className="text-red-800">You don't have permission to view roles.</p>
      </div>
    );
  }

  const handleCreateRole = async () => {
    if (!formData.name) {
      alert('Role name is required');
      return;
    }

    try {
      await createRoleMutation.mutateAsync({
        name: formData.name,
        description: formData.description || undefined,
        permissions: formData.permissions,
      });

      setFormData({ name: '', description: '', permissions: [] });
      setShowCreateForm(false);
    } catch (error) {
      alert(`Error creating role: ${error}`);
    }
  };

  const handleDeleteRole = async (roleId: string) => {
    if (window.confirm('Are you sure you want to delete this role?')) {
      try {
        await deleteRoleMutation.mutateAsync(roleId);
      } catch (error) {
        alert(`Error deleting role: ${error}`);
      }
    }
  };

  const togglePermission = (permission: string) => {
    setFormData((prev) => ({
      ...prev,
      permissions: prev.permissions.includes(permission)
        ? prev.permissions.filter((p) => p !== permission)
        : [...prev.permissions, permission],
    }));
  };

  const permissionsByCategory = permissions
    ? Object.entries(
        permissions.reduce(
          (acc, perm) => {
            if (!acc[perm.category]) acc[perm.category] = [];
            acc[perm.category].push(perm);
            return acc;
          },
          {} as Record<string, typeof permissions>
        )
      )
    : [];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Roles</h1>
          <p className="text-gray-600 mt-1">Create and manage admin roles</p>
        </div>
        {hasPermission('roles:write') && (
          <button
            onClick={() => setShowCreateForm(!showCreateForm)}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            <Plus className="w-5 h-5" />
            <span>Create Role</span>
          </button>
        )}
      </div>

      {/* Create Form */}
      {showCreateForm && hasPermission('roles:write') && (
        <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-6">Create New Role</h2>
          
          {/* Form Fields */}
          <div className="space-y-4 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                Role Name *
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData((prev) => ({ ...prev, name: e.target.value }))}
                placeholder="e.g., Editor, Operator"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-900 mb-2">
                Description
              </label>
              <textarea
                value={formData.description}
                onChange={(e) => setFormData((prev) => ({ ...prev, description: e.target.value }))}
                placeholder="Brief description of this role..."
                rows={3}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>

            {/* Permissions */}
            <div>
              <label className="block text-sm font-medium text-gray-900 mb-4">Permissions</label>
              <div className="space-y-4">
                {permissionsByCategory.map(([category, perms]) => (
                  <div key={category} className="bg-gray-50 rounded-lg p-4">
                    <h3 className="font-medium text-gray-900 mb-3 capitalize">{category}</h3>
                    <div className="space-y-2">
                      {perms.map((perm) => (
                        <label key={perm.permission} className="flex items-center">
                          <input
                            type="checkbox"
                            checked={formData.permissions.includes(perm.permission)}
                            onChange={() => togglePermission(perm.permission)}
                            className="rounded border-gray-300"
                          />
                          <span className="ml-3 text-sm text-gray-700">{perm.description}</span>
                        </label>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Form Actions */}
          <div className="flex space-x-3">
            <button
              onClick={handleCreateRole}
              disabled={createRoleMutation.isPending || !formData.name}
              className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
            >
              {createRoleMutation.isPending && <Loader className="w-4 h-4 animate-spin" />}
              <span>Create Role</span>
            </button>
            <button
              onClick={() => setShowCreateForm(false)}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Roles Table */}
      <div className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden">
        {rolesLoading ? (
          <div className="flex items-center justify-center h-96">
            <Loader className="w-8 h-8 animate-spin text-blue-600" />
          </div>
        ) : !rolesData || rolesData.length === 0 ? (
          <div className="p-12 text-center">
            <p className="text-gray-600">No roles found</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b border-gray-200">
                <tr>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Name</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Description</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Permissions</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Type</th>
                  <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Actions</th>
                </tr>
              </thead>
              <tbody>
                {rolesData.map((role) => (
                  <tr key={role.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="px-6 py-4 font-medium text-gray-900">{role.name}</td>
                    <td className="px-6 py-4 text-sm text-gray-600">
                      {role.description || '-'}
                    </td>
                    <td className="px-6 py-4 text-sm">
                      <div className="flex flex-wrap gap-1">
                        {role.permissions.slice(0, 2).map((perm) => (
                          <span
                            key={perm}
                            className="inline-block px-2 py-1 rounded text-xs bg-blue-100 text-blue-800"
                          >
                            {perm}
                          </span>
                        ))}
                        {role.permissions.length > 2 && (
                          <span className="inline-block px-2 py-1 text-xs text-gray-600">
                            +{role.permissions.length - 2} more
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-6 py-4 text-sm">
                      {role.is_system ? (
                        <span className="inline-block px-2 py-1 rounded text-xs bg-gray-100 text-gray-800">
                          System
                        </span>
                      ) : (
                        <span className="inline-block px-2 py-1 rounded text-xs bg-green-100 text-green-800">
                          Custom
                        </span>
                      )}
                    </td>
                    <td className="px-6 py-4 text-sm space-x-2">
                      {!role.is_system && hasPermission('roles:write') && (
                        <button className="p-2 hover:bg-gray-100 rounded-lg text-gray-600">
                          <Edit className="w-4 h-4" />
                        </button>
                      )}
                      {!role.is_system && hasPermission('roles:delete') && (
                        <button
                          onClick={() => handleDeleteRole(role.id)}
                          className="p-2 hover:bg-red-50 rounded-lg text-red-600"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default Roles;
