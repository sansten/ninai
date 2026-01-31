/**
 * Dashboard Page
 * ==============
 * 
 * Main dashboard with overview statistics and recent activity.
 */

import { useQuery } from '@tanstack/react-query';
import {
  CircleStackIcon,
  UserGroupIcon,
  UsersIcon,
  ShieldExclamationIcon,
} from '@heroicons/react/24/outline';
import { apiClient } from '@/lib/api';
import { useCanViewAudit, useCurrentOrg, useIsAdmin } from '@/stores/auth';
import type { AuditStats } from '@/types/api';

/**
 * Stats Card Component
 */
interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ElementType;
  color: 'primary' | 'green' | 'yellow' | 'red';
  description?: string;
}

function StatCard({ title, value, icon: Icon, color, description }: StatCardProps) {
  const colorClasses = {
    primary: 'bg-primary-100 text-primary-600',
    green: 'bg-green-100 text-green-600',
    yellow: 'bg-yellow-100 text-yellow-600',
    red: 'bg-red-100 text-red-600',
  };

  return (
    <div className="card">
      <div className="flex items-center">
        <div className={`p-3 rounded-lg ${colorClasses[color]}`}>
          <Icon className="h-6 w-6" />
        </div>
        <div className="ml-4 flex-1">
          <p className="text-sm font-medium text-gray-500">{title}</p>
          <p className="text-2xl font-bold text-gray-900">{value}</p>
          {description && (
            <p className="text-xs text-gray-500 mt-1">{description}</p>
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Recent Activity Item
 */
interface ActivityItemProps {
  event_type: string;
  event_category: string;
  resource_type?: string;
  timestamp: string;
}

function ActivityItem({ event_type, event_category, resource_type, timestamp }: ActivityItemProps) {
  return (
    <div className="flex items-center py-3">
      <div className="flex-1 min-w-0">
        <p className="text-sm font-medium text-gray-900 truncate">
          {event_type}
        </p>
        <p className="text-sm text-gray-500">
          {event_category}{resource_type && ` on ${resource_type}`}
        </p>
      </div>
      <div className="text-xs text-gray-400">
        {new Date(timestamp).toLocaleString()}
      </div>
    </div>
  );
}

/**
 * Dashboard Page Component
 */
export function DashboardPage() {
  const org = useCurrentOrg();
  const canViewAudit = useCanViewAudit();
  const isAdmin = useIsAdmin();

  // Fetch audit stats
  const { data: stats, isLoading: statsLoading, error: statsError } = useQuery<AuditStats>({
    queryKey: ['audit-stats', org.id],
    queryFn: async () => {
      const response = await apiClient.get('/audit/stats');
      return response.data;
    },
    enabled: canViewAudit,
  });

  // Fetch recent events
  const { data: recentEvents, isLoading: eventsLoading, error: eventsError } = useQuery({
    queryKey: ['recent-events', org.id],
    queryFn: async () => {
      const response = await apiClient.get('/audit/events', {
        params: { page_size: 10 },
      });
      return response.data.items;
    },
    enabled: canViewAudit,
  });

  // Show error state if queries fail
  if (canViewAudit && (statsError || eventsError)) {
    return (
      <div className="space-y-8">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-500 mt-1">
            Welcome to {org.name}'s Memory OS
          </p>
        </div>
        <div className="card bg-red-50 border border-red-200">
          <h2 className="text-lg font-semibold text-red-900 mb-2">Error Loading Dashboard</h2>
          <p className="text-red-700 mb-2">{statsError?.message || eventsError?.message}</p>
          <details className="text-red-600 text-sm">
            <summary>Details</summary>
            <pre className="mt-2 p-2 bg-white rounded text-xs overflow-auto">
              Stats Error: {JSON.stringify(statsError, null, 2)}
              {'\n\n'}
              Events Error: {JSON.stringify(eventsError, null, 2)}
            </pre>
          </details>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-500 mt-1">
          Welcome to {org.name}'s Memory OS
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Total Events"
          value={!canViewAudit ? '-' : statsLoading ? '...' : (stats?.total_events ?? 0)}
          icon={CircleStackIcon}
          color="primary"
          description="Last 7 days"
        />
        <StatCard
          title="Memory Operations"
          value={!canViewAudit ? '-' : statsLoading ? '...' : (stats?.events_by_type?.['memory.create'] ?? 0) + (stats?.events_by_type?.['memory.read'] ?? 0)}
          icon={UserGroupIcon}
          color="green"
          description="Creates + Reads"
        />
        <StatCard
          title="Active Users"
          value={!canViewAudit ? '-' : statsLoading ? '...' : (stats?.top_actors?.length ?? 0)}
          icon={UsersIcon}
          color="yellow"
          description="With recent activity"
        />
        <StatCard
          title="Access Denials"
          value={!canViewAudit ? '-' : statsLoading ? '...' : (stats?.recent_denials ?? 0)}
          icon={ShieldExclamationIcon}
          color="red"
          description="Review required"
        />
      </div>

      {!canViewAudit && (
        <div className="card bg-blue-50 border border-blue-200">
          <h2 className="text-lg font-semibold text-blue-900 mb-1">Limited dashboard</h2>
          <p className="text-sm text-blue-700">
            Audit statistics and recent activity are only available to <span className="font-medium">org_admin</span>, <span className="font-medium">security_admin</span>, or <span className="font-medium">system_admin</span>.
          </p>
        </div>
      )}

      {/* Content Grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Recent Activity */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Recent Activity
          </h2>
          {!canViewAudit ? (
            <p className="text-gray-500 text-center py-8">Not available for your role</p>
          ) : eventsLoading ? (
            <div className="flex justify-center py-8">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary-600" />
            </div>
          ) : recentEvents?.length > 0 ? (
            <div className="divide-y divide-gray-100">
              {recentEvents.map((event: ActivityItemProps & { id: string }) => (
                <ActivityItem key={event.id} {...event} />
              ))}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-8">No recent activity</p>
          )}
        </div>

        {/* Quick Actions */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Quick Actions
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <a
              href="/memories"
              className="flex flex-col items-center p-4 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
            >
              <CircleStackIcon className="h-8 w-8 text-primary-600" />
              <span className="mt-2 text-sm font-medium text-gray-900">
                Browse Memories
              </span>
            </a>
            {isAdmin && (
              <a
                href="/teams"
                className="flex flex-col items-center p-4 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
              >
                <UserGroupIcon className="h-8 w-8 text-primary-600" />
                <span className="mt-2 text-sm font-medium text-gray-900">
                  Manage Teams
                </span>
              </a>
            )}
            {isAdmin && (
              <a
                href="/users"
                className="flex flex-col items-center p-4 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
              >
                <UsersIcon className="h-8 w-8 text-primary-600" />
                <span className="mt-2 text-sm font-medium text-gray-900">
                  User Management
                </span>
              </a>
            )}
            {canViewAudit && (
              <a
                href="/audit"
                className="flex flex-col items-center p-4 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors"
              >
                <ShieldExclamationIcon className="h-8 w-8 text-primary-600" />
                <span className="mt-2 text-sm font-medium text-gray-900">
                  Audit Logs
                </span>
              </a>
            )}
          </div>
        </div>
      </div>

      {/* Events by Type */}
      {canViewAudit && stats?.events_by_type && Object.keys(stats.events_by_type).length > 0 && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Events by Type
          </h2>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-4">
            {Object.entries(stats.events_by_type).map(([type, count]) => (
              <div
                key={type}
                className="p-4 bg-gray-50 rounded-lg"
              >
                <p className="text-sm text-gray-500">{type}</p>
                <p className="text-xl font-bold text-gray-900">{count}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
