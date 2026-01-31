/**
 * Audit Logs Page
 * View and filter admin action audit trail
 */

import React, { useState } from 'react';
import { useAdmin } from '../../hooks/useAdmin';
import { useAuditLogs } from '../../hooks/useAdminAPI';
import { Search, Filter, Loader, AlertCircle, Eye, ChevronDown } from 'lucide-react';
import { cn } from '../../lib/utils';

const AuditLogs: React.FC = () => {
  const { hasPermission } = useAdmin();
  const [page, setPage] = useState(1);
  const [showFilters, setShowFilters] = useState(false);
  const [filters, setFilters] = useState({
    action: '',
    resourceType: '',
    adminId: '',
  });
  const [expandedLogId, setExpandedLogId] = useState<string | null>(null);

  const { data, isLoading, error } = useAuditLogs(
    filters.adminId || undefined,
    filters.action || undefined,
    filters.resourceType || undefined,
    page,
    50
  );

  if (!hasPermission('audit:read')) {
    return (
      <div className="p-6 bg-red-50 rounded-lg border border-red-200">
        <p className="text-red-800">You don't have permission to view audit logs.</p>
      </div>
    );
  }

  const actionOptions = [
    'create', 'read', 'update', 'delete', 'disable', 'enable',
  ];
  const resourceTypes = [
    'user', 'role', 'setting', 'session', 'ip_whitelist', 'audit_log',
  ];

  const handleFilterChange = (key: string, value: string) => {
    setFilters((prev) => ({ ...prev, [key]: value }));
    setPage(1);
  };

  const clearFilters = () => {
    setFilters({ action: '', resourceType: '', adminId: '' });
    setPage(1);
  };

  const renderChangeDetails = (oldValues?: any, newValues?: any) => {
    if (!oldValues && !newValues) return 'No changes recorded';

    return (
      <div className="space-y-2 text-sm">
        {oldValues && Object.keys(oldValues).length > 0 && (
          <div>
            <p className="font-semibold text-gray-700">Before:</p>
            <div className="bg-red-50 p-2 rounded border border-red-200 font-mono text-xs">
              {JSON.stringify(oldValues, null, 2)}
            </div>
          </div>
        )}
        {newValues && Object.keys(newValues).length > 0 && (
          <div>
            <p className="font-semibold text-gray-700">After:</p>
            <div className="bg-green-50 p-2 rounded border border-green-200 font-mono text-xs">
              {JSON.stringify(newValues, null, 2)}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Audit Logs</h1>
        <p className="text-gray-600 mt-1">Complete audit trail of admin actions</p>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-lg shadow border border-gray-200 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-semibold text-gray-900">Filters</h2>
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center space-x-1 text-blue-600 hover:text-blue-700"
          >
            <Filter className="w-4 h-4" />
            <span className="text-sm">{showFilters ? 'Hide' : 'Show'}</span>
            <ChevronDown className={cn('w-4 h-4 transition-transform', showFilters && 'rotate-180')} />
          </button>
        </div>

        {showFilters && (
          <>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">Action</label>
                <select
                  value={filters.action}
                  onChange={(e) => handleFilterChange('action', e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">All Actions</option>
                  {actionOptions.map((action) => (
                    <option key={action} value={action}>
                      {action.charAt(0).toUpperCase() + action.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">Resource Type</label>
                <select
                  value={filters.resourceType}
                  onChange={(e) => handleFilterChange('resourceType', e.target.value)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="">All Resources</option>
                  {resourceTypes.map((type) => (
                    <option key={type} value={type}>
                      {type.charAt(0).toUpperCase() + type.slice(1).replace(/_/g, ' ')}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-900 mb-2">Admin ID</label>
                <input
                  type="text"
                  value={filters.adminId}
                  onChange={(e) => handleFilterChange('adminId', e.target.value)}
                  placeholder="Filter by admin..."
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>

            <div className="flex space-x-2">
              {(filters.action || filters.resourceType || filters.adminId) && (
                <button
                  onClick={clearFilters}
                  className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50"
                >
                  Clear Filters
                </button>
              )}
            </div>
          </>
        )}
      </div>

      {/* Audit Logs Table */}
      <div className="bg-white rounded-lg shadow border border-gray-200 overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center h-96">
            <Loader className="w-8 h-8 animate-spin text-blue-600" />
          </div>
        ) : error ? (
          <div className="p-6 bg-red-50">
            <div className="flex items-center space-x-2">
              <AlertCircle className="w-5 h-5 text-red-600" />
              <p className="text-red-800">Failed to load audit logs</p>
            </div>
          </div>
        ) : !data || data.items.length === 0 ? (
          <div className="p-12 text-center">
            <p className="text-gray-600">No audit logs found</p>
          </div>
        ) : (
          <div className="space-y-0">
            {data.items.map((log) => (
              <div key={log.id} className="border-b border-gray-100 last:border-b-0">
                <button
                  onClick={() => setExpandedLogId(expandedLogId === log.id ? null : log.id)}
                  className="w-full px-6 py-4 text-left hover:bg-gray-50 transition-colors"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-3">
                        <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800">
                          {log.action}
                        </span>
                        <span className="text-sm text-gray-600">
                          {log.resource_type}
                          {log.resource_id && ` (${log.resource_id.substring(0, 8)}...)`}
                        </span>
                        <span className="text-sm text-gray-500">
                          {new Date(log.created_at).toLocaleString()}
                        </span>
                      </div>
                      {log.ip_address && (
                        <p className="text-xs text-gray-500 mt-1">{log.ip_address}</p>
                      )}
                    </div>
                    <Eye className={cn(
                      'w-5 h-5 text-gray-400 transition-transform',
                      expandedLogId === log.id && 'rotate-180'
                    )} />
                  </div>
                </button>

                {expandedLogId === log.id && (
                  <div className="px-6 py-4 bg-gray-50 border-t border-gray-100">
                    <div className="space-y-4">
                      <div>
                        <p className="text-sm font-semibold text-gray-900 mb-2">Admin ID</p>
                        <p className="text-sm text-gray-600 font-mono">{log.admin_id}</p>
                      </div>

                      {log.resource_id && (
                        <div>
                          <p className="text-sm font-semibold text-gray-900 mb-2">Resource ID</p>
                          <p className="text-sm text-gray-600 font-mono">{log.resource_id}</p>
                        </div>
                      )}

                      <div>
                        <p className="text-sm font-semibold text-gray-900 mb-2">Changes</p>
                        {renderChangeDetails(log.old_values, log.new_values)}
                      </div>

                      {log.user_agent && (
                        <div>
                          <p className="text-sm font-semibold text-gray-900 mb-2">User Agent</p>
                          <p className="text-xs text-gray-600 break-words">{log.user_agent}</p>
                        </div>
                      )}

                      <div>
                        <p className="text-sm font-semibold text-gray-900 mb-2">Timestamp</p>
                        <p className="text-sm text-gray-600">
                          {new Date(log.created_at).toLocaleString()}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Pagination */}
      {data && data.total > 50 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-600">
            Showing {(page - 1) * 50 + 1} to {Math.min(page * 50, data.total)} of {data.total} logs
          </p>
          <div className="flex space-x-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
            >
              Previous
            </button>
            <button
              onClick={() => setPage(page + 1)}
              disabled={page * 50 >= data.total}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default AuditLogs;
