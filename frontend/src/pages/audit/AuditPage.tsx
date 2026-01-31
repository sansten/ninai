/**
 * Audit Page
 * ==========
 * 
 * Audit event viewer and access log analysis.
 */

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  FunnelIcon,
  ShieldExclamationIcon,
  ClipboardDocumentListIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { apiClient, PaginatedResponse } from '@/lib/api';
import { useCanViewAudit, useCurrentOrg } from '@/stores/auth';
import type { AuditEvent, AccessLog } from '@/types/api';

// Event type colors
function getEventTypeColor(eventType: string): string {
  if (eventType.startsWith('auth.')) return 'badge-primary';
  if (eventType.startsWith('memory.')) return 'badge-success';
  if (eventType.startsWith('permission.')) return 'badge-warning';
  if (eventType.startsWith('admin.')) return 'badge-danger';
  return 'badge-gray';
}

/**
 * Audit Event Row Component
 */
interface EventRowProps {
  event: AuditEvent;
}

function EventRow({ event }: EventRowProps) {
  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap">
        <span className={getEventTypeColor(event.event_type)}>
          {event.event_type}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {event.action}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {event.resource_type}
        {event.resource_id && (
          <span className="text-gray-400 ml-1">
            ({event.resource_id.slice(0, 8)}...)
          </span>
        )}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        <span className="capitalize">{event.actor_type}</span>
        {event.actor_id && (
          <span className="text-gray-400 ml-1">
            ({event.actor_id.slice(0, 8)}...)
          </span>
        )}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
        {new Date(event.timestamp).toLocaleString()}
      </td>
    </tr>
  );
}

/**
 * Access Log Row Component
 */
interface AccessLogRowProps {
  log: AccessLog;
}

function AccessLogRow({ log }: AccessLogRowProps) {
  return (
    <tr className="hover:bg-gray-50">
      <td className="px-6 py-4 whitespace-nowrap">
        <span className={log.access_granted ? 'badge-success' : 'badge-danger'}>
          {log.access_granted ? 'Granted' : 'Denied'}
        </span>
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {log.access_type}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {log.memory_id.slice(0, 8)}...
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
        {log.user_id ? `User: ${log.user_id.slice(0, 8)}...` : log.agent_id ? `Agent: ${log.agent_id.slice(0, 8)}...` : 'Unknown'}
      </td>
      <td className="px-6 py-4 text-sm text-gray-500 max-w-xs truncate">
        {log.denial_reason || log.permission_path || '-'}
      </td>
      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
        {new Date(log.accessed_at).toLocaleString()}
      </td>
    </tr>
  );
}

/**
 * Audit Page Component
 */
export function AuditPage() {
  const org = useCurrentOrg();
  const canViewAudit = useCanViewAudit();
  const [activeTab, setActiveTab] = useState<'events' | 'access'>('events');
  const [eventType, setEventType] = useState('');
  const [showFilters, setShowFilters] = useState(false);
  const [page, setPage] = useState(1);

  // Fetch audit events
  const { data: eventsData, isLoading: eventsLoading } = useQuery<PaginatedResponse<AuditEvent>>({
    queryKey: ['audit-events', org.id, eventType, page],
    queryFn: async () => {
      const response = await apiClient.get('/audit/events', {
        params: { event_type: eventType || undefined, page, page_size: 50 },
      });
      return response.data;
    },
    enabled: canViewAudit && activeTab === 'events',
  });

  // Fetch access logs
  const { data: accessData, isLoading: accessLoading } = useQuery<PaginatedResponse<AccessLog>>({
    queryKey: ['access-logs', org.id, page],
    queryFn: async () => {
      const response = await apiClient.get('/audit/access-logs', {
        params: { page, page_size: 50 },
      });
      return response.data;
    },
    enabled: canViewAudit && activeTab === 'access',
  });

  // Fetch recent denials
  const { data: denials } = useQuery<AccessLog[]>({
    queryKey: ['recent-denials', org.id],
    queryFn: async () => {
      const response = await apiClient.get('/audit/stats/denials', {
        params: { limit: 10 },
      });
      return response.data;
    },
    enabled: canViewAudit,
  });

  if (!canViewAudit) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Audit Log</h1>
          <p className="text-gray-500 mt-1">
            Monitor system activity and access patterns
          </p>
        </div>
        <div className="card bg-blue-50 border border-blue-200">
          <h2 className="text-lg font-semibold text-blue-900 mb-1">Permission required</h2>
          <p className="text-sm text-blue-700">
            Audit logs are only available to <span className="font-medium">org_admin</span>, <span className="font-medium">security_admin</span>, or <span className="font-medium">system_admin</span>.
          </p>
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'events', name: 'Audit Events', icon: ClipboardDocumentListIcon },
    { id: 'access', name: 'Access Logs', icon: ShieldExclamationIcon },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Audit Log</h1>
        <p className="text-gray-500 mt-1">
          Monitor system activity and access patterns
        </p>
      </div>

      {/* Denial Alert */}
      {denials && denials.length > 0 && (
        <div className="card bg-red-50 border-red-200">
          <div className="flex items-center gap-3">
            <ShieldExclamationIcon className="h-6 w-6 text-red-600" />
            <div>
              <p className="font-medium text-red-800">
                {denials.length} recent access denial{denials.length > 1 ? 's' : ''}
              </p>
              <p className="text-sm text-red-600">
                Review denied access attempts for potential security issues
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setActiveTab(tab.id as 'events' | 'access');
                setPage(1);
              }}
              className={clsx(
                'flex items-center gap-2 py-4 px-1 border-b-2 font-medium text-sm',
                activeTab === tab.id
                  ? 'border-primary-500 text-primary-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              )}
            >
              <tab.icon className="h-5 w-5" />
              {tab.name}
            </button>
          ))}
        </nav>
      </div>

      {/* Filters */}
      {activeTab === 'events' && (
        <div className="card">
          <div className="flex items-center gap-4">
            <div className="flex-1">
              <select
                value={eventType}
                onChange={(e) => {
                  setEventType(e.target.value);
                  setPage(1);
                }}
                className="input"
              >
                <option value="">All event types</option>
                <option value="auth.login">auth.login</option>
                <option value="auth.logout">auth.logout</option>
                <option value="memory.create">memory.create</option>
                <option value="memory.read">memory.read</option>
                <option value="memory.update">memory.update</option>
                <option value="memory.delete">memory.delete</option>
                <option value="permission.granted">permission.granted</option>
                <option value="permission.denied">permission.denied</option>
              </select>
            </div>
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={clsx('btn-secondary', showFilters && 'bg-gray-300')}
            >
              <FunnelIcon className="h-5 w-5" />
            </button>
          </div>
        </div>
      )}

      {/* Events Table */}
      {activeTab === 'events' && (
        eventsLoading ? (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
          </div>
        ) : eventsData?.items?.length ? (
          <div className="card p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Action</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Resource</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actor</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {eventsData.items.map((event) => (
                    <EventRow key={event.id} event={event} />
                  ))}
                </tbody>
              </table>
            </div>

            {eventsData.pages > 1 && (
              <div className="px-6 py-4 border-t flex items-center justify-between">
                <p className="text-sm text-gray-500">
                  Page {page} of {eventsData.pages}
                </p>
                <div className="flex gap-2">
                  <button onClick={() => setPage(page - 1)} disabled={page === 1} className="btn-secondary">
                    Previous
                  </button>
                  <button onClick={() => setPage(page + 1)} disabled={page >= eventsData.pages} className="btn-secondary">
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12">
            <ClipboardDocumentListIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-semibold text-gray-900">No events</h3>
          </div>
        )
      )}

      {/* Access Logs Table */}
      {activeTab === 'access' && (
        accessLoading ? (
          <div className="flex justify-center py-12">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
          </div>
        ) : accessData?.items?.length ? (
          <div className="card p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Memory</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Accessor</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Details</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Time</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {accessData.items.map((log) => (
                    <AccessLogRow key={log.id} log={log} />
                  ))}
                </tbody>
              </table>
            </div>

            {accessData.pages > 1 && (
              <div className="px-6 py-4 border-t flex items-center justify-between">
                <p className="text-sm text-gray-500">
                  Page {page} of {accessData.pages}
                </p>
                <div className="flex gap-2">
                  <button onClick={() => setPage(page - 1)} disabled={page === 1} className="btn-secondary">
                    Previous
                  </button>
                  <button onClick={() => setPage(page + 1)} disabled={page >= accessData.pages} className="btn-secondary">
                    Next
                  </button>
                </div>
              </div>
            )}
          </div>
        ) : (
          <div className="text-center py-12">
            <ShieldExclamationIcon className="mx-auto h-12 w-12 text-gray-400" />
            <h3 className="mt-2 text-sm font-semibold text-gray-900">No access logs</h3>
          </div>
        )
      )}
    </div>
  );
}
