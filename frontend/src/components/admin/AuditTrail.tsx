/**
 * Audit Trail Component
 * =====================
 * 
 * Display admin actions with filtering and details.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { FunnelIcon, InformationCircleIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { apiClient, getErrorMessage } from '@/lib/api';

interface AuditEvent {
  id: string;
  timestamp: string;
  category: string;
  user_email?: string;
  before?: Record<string, unknown>;
  after?: Record<string, unknown>;
  status?: 'success' | 'failed';
}

interface AuditResponse {
  events: AuditEvent[];
  total: number;
}

const ACTION_CATEGORIES = [
  { value: 'policy', label: 'Policy Changes' },
  { value: 'admission', label: 'Admission Control' },
  { value: 'backup', label: 'Backups & Restore' },
  { value: 'token', label: 'Token Management' },
  { value: 'alert', label: 'Alerts' },
  { value: 'queue', label: 'Queue Management' },
  { value: 'security', label: 'Security' },
  { value: 'config', label: 'Configuration' },
];

export function AuditTrail() {
  const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const auditQuery = useQuery<AuditResponse>({
    queryKey: ['admin', 'audit', selectedCategory],
    queryFn: async () => {
      const res = await apiClient.get('/admin/audit', {
        params: {
          ...(selectedCategory ? { category: selectedCategory } : {}),
          limit: 50,
        },
      });
      return res.data;
    },
  });

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex gap-3">
        <InformationCircleIcon className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-blue-900 text-sm">Admin Audit Trail</h3>
          <p className="text-blue-800 text-xs mt-1">All admin actions are logged here with timestamps, user identity, and before/after values. Use category filters to find specific event types. Click events to see detailed change information.</p>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">Audit Trail</h2>
            <p className="text-sm text-gray-500 mt-1">All admin actions with who, when, and what changed.</p>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2 mb-4 flex-wrap">
          <FunnelIcon className="w-4 h-4 text-gray-400" />
          <button
            onClick={() => setSelectedCategory(null)}
            className={clsx(
              'px-3 py-1 rounded-full text-sm font-medium transition-colors',
              selectedCategory === null ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            )}
          >
            All Actions
          </button>
          {ACTION_CATEGORIES.map((cat) => (
            <button
              key={cat.value}
              onClick={() => setSelectedCategory(selectedCategory === cat.value ? null : cat.value)}
              className={clsx(
                'px-3 py-1 rounded-full text-sm font-medium transition-colors',
                selectedCategory === cat.value ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
              )}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Events List */}
        {auditQuery.isLoading ? (
          <p className="text-sm text-gray-500">Loading audit events…</p>
        ) : auditQuery.isError ? (
          <div className="p-3 bg-red-50 border border-red-200 rounded text-sm text-red-600">
            Failed to load audit trail: {getErrorMessage(auditQuery.error)}
          </div>
        ) : (
          <div className="space-y-2">
            {auditQuery.data?.events && auditQuery.data.events.length > 0 ? (
              auditQuery.data.events.map((event) => (
                <div
                  key={event.id}
                  className="border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                >
                  <button
                    onClick={() => setExpandedId(expandedId === event.id ? null : event.id)}
                    className="w-full p-4 text-left flex items-center justify-between"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={clsx(
                          'px-2 py-1 rounded text-xs font-medium',
                          event.status === 'success' || !event.status
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        )}>
                          {event.status?.toUpperCase() || 'SUCCESS'}
                        </span>
                        <span className="font-medium text-gray-900">{event.category}</span>
                        {event.resource_type && (
                          <span className="text-xs text-gray-500">{event.resource_type}</span>
                        )}
                      </div>
                      <div className="mt-1 flex items-center gap-4">
                        <p className="text-sm text-gray-600">{event.user_email || event.user_id}</p>
                        <p className="text-xs text-gray-500">{new Date(event.timestamp).toLocaleString()}</p>
                      </div>
                    </div>
                    <div className="ml-4">
                      <svg
                        className={clsx(
                          'w-5 h-5 text-gray-400 transition-transform',
                          expandedId === event.id && 'transform rotate-180'
                        )}
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                      </svg>
                    </div>
                  </button>

                  {/* Expanded Details */}
                  {expandedId === event.id && (
                    <div className="px-4 pb-4 border-t border-gray-200 space-y-3">
                      {event.details && (
                        <div>
                          <p className="text-xs font-medium text-gray-500 uppercase">Details</p>
                          <p className="text-sm text-gray-700 mt-1">{event.details}</p>
                        </div>
                      )}

                      {(event.before || event.after) && (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          {event.before && (
                            <div>
                              <p className="text-xs font-medium text-gray-500 uppercase mb-2">Before</p>
                              <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto max-h-40">
                                {JSON.stringify(event.before, null, 2)}
                              </pre>
                            </div>
                          )}
                          {event.after && (
                            <div>
                              <p className="text-xs font-medium text-gray-500 uppercase mb-2">After</p>
                              <pre className="text-xs bg-gray-100 p-2 rounded overflow-x-auto max-h-40">
                                {JSON.stringify(event.after, null, 2)}
                              </pre>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))
            ) : (
              <p className="text-sm text-gray-500 text-center py-8">No audit events found.</p>
            )}
          </div>
        )}

        {auditQuery.data && (
          <div className="mt-4 text-xs text-gray-500">
            Showing {auditQuery.data.items?.length ?? 0} of {auditQuery.data.total} events
          </div>
        )}
      </div>
    </div>
  );
}
