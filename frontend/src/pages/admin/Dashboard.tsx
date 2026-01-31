/**
 * Admin Dashboard Page
 * Displays system KPIs, service health, alerts, and recent activity
 */

import React from 'react';
import { useAdmin } from '../../hooks/useAdmin';
import { useDashboard } from '../../hooks/useAdminAPI';
import {
  BarChart3, AlertTriangle, TrendingUp, Activity, Loader, AlertCircle
} from 'lucide-react';
import { cn } from '../../lib/utils';

const Dashboard: React.FC = () => {
  const { hasPermission } = useAdmin();
  const { data: dashboard, isLoading, error } = useDashboard();

  if (!hasPermission('system:read')) {
    return (
      <div className="p-6 bg-red-50 rounded-lg border border-red-200">
        <p className="text-red-800">You don't have permission to view the dashboard.</p>
      </div>
    );
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-96">
        <Loader className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-50 rounded-lg border border-red-200">
        <div className="flex items-center space-x-2">
          <AlertCircle className="w-5 h-5 text-red-600" />
          <p className="text-red-800">Failed to load dashboard data</p>
        </div>
      </div>
    );
  }

  if (!dashboard) {
    return null;
  }

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="text-gray-600 mt-1">System overview and key metrics</p>
      </div>

      {/* KPI Cards - Users */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">User Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {dashboard.kpis.users.map((kpi, idx) => (
            <KPICard key={idx} kpi={kpi} />
          ))}
        </div>
      </div>

      {/* KPI Cards - Memory */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Memory Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {dashboard.kpis.memories.map((kpi, idx) => (
            <KPICard key={idx} kpi={kpi} />
          ))}
        </div>
      </div>

      {/* KPI Cards - System */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">System Metrics</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {dashboard.kpis.system.map((kpi, idx) => (
            <KPICard key={idx} kpi={kpi} />
          ))}
        </div>
      </div>

      {/* Service Health & Alerts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Service Health */}
        <div className="bg-white rounded-lg shadow border border-gray-200">
          <div className="p-6 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Service Health</h2>
          </div>
          <div className="p-6 space-y-4">
            {dashboard.services.map((service, idx) => (
              <div key={idx} className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{service.name}</p>
                  {service.message && (
                    <p className="text-sm text-gray-600 mt-1">{service.message}</p>
                  )}
                </div>
                <div
                  className={cn(
                    'px-3 py-1 rounded-full text-xs font-semibold',
                    service.status === 'healthy' && 'bg-green-100 text-green-800',
                    service.status === 'degraded' && 'bg-yellow-100 text-yellow-800',
                    service.status === 'unhealthy' && 'bg-red-100 text-red-800'
                  )}
                >
                  {service.status.charAt(0).toUpperCase() + service.status.slice(1)}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Alerts */}
        <div className="bg-white rounded-lg shadow border border-gray-200">
          <div className="p-6 border-b border-gray-200 flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900">Active Alerts</h2>
            <div className="text-2xl font-bold text-yellow-600">
              {dashboard.alerts?.length || 0}
            </div>
          </div>
          <div className="p-6">
            {(!dashboard.alerts || dashboard.alerts.length === 0) ? (
              <p className="text-gray-600 text-center py-8">No active alerts</p>
            ) : (
              <div className="flex items-center space-x-3 p-4 bg-yellow-50 rounded-lg border border-yellow-200">
                <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0" />
                <p className="text-yellow-800">
                  {dashboard.alerts.length} alert{dashboard.alerts.length !== 1 ? 's' : ''} require attention
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white rounded-lg shadow border border-gray-200">
        <div className="p-6 border-b border-gray-200 flex items-center space-x-2">
          <Activity className="w-5 h-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-900">Recent Activity</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200">
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Action</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Resource</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Timestamp</th>
                <th className="px-6 py-3 text-left text-sm font-semibold text-gray-900">Status</th>
              </tr>
            </thead>
            <tbody>
              {dashboard.recent_activities.map((activity) => (
                <tr
                  key={activity.id}
                  className="border-b border-gray-100 hover:bg-gray-50"
                >
                  <td className="px-6 py-4 text-sm">
                    <span className="inline-block px-3 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-800">
                      {activity.action}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900">
                    {activity.resource_type}
                    {activity.resource_id && ` (${activity.resource_id.substring(0, 8)}...)`}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-600">
                    {new Date(activity.timestamp).toLocaleString()}
                  </td>
                  <td className="px-6 py-4 text-sm">
                    <span
                      className={cn(
                        'inline-block px-2 py-1 rounded text-xs font-semibold',
                        activity.status === 'success' && 'bg-green-100 text-green-800',
                        activity.status === 'failed' && 'bg-red-100 text-red-800',
                        activity.status !== 'success' && activity.status !== 'failed' && 'bg-gray-100 text-gray-800'
                      )}
                    >
                      {activity.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

/**
 * KPI Card Component
 */
const KPICard: React.FC<{ kpi: any }> = ({ kpi }) => (
  <div className="bg-white rounded-lg shadow p-6 border border-gray-200">
    <div className="flex items-center justify-between">
      <div>
        <p className="text-gray-600 text-sm font-medium">{kpi.label}</p>
        <p className="text-2xl font-bold text-gray-900 mt-1">{kpi.value}</p>
        {kpi.unit && (
          <p className="text-gray-500 text-xs mt-1">{kpi.unit}</p>
        )}
      </div>
      {kpi.trend && (
        <div
          className={cn(
            'w-12 h-12 rounded-lg flex items-center justify-center',
            kpi.trend === 'up' && 'bg-green-100',
            kpi.trend === 'down' && 'bg-red-100',
            kpi.trend === 'stable' && 'bg-gray-100'
          )}
        >
          <TrendingUp
            className={cn(
              'w-6 h-6',
              kpi.trend === 'up' && 'text-green-600',
              kpi.trend === 'down' && 'text-red-600 rotate-180',
              kpi.trend === 'stable' && 'text-gray-600'
            )}
          />
        </div>
      )}
    </div>
    {kpi.change_percent !== undefined && (
      <p
        className={cn(
          'text-xs font-medium mt-3',
          kpi.change_percent > 0 && 'text-green-600',
          kpi.change_percent < 0 && 'text-red-600',
          kpi.change_percent === 0 && 'text-gray-600'
        )}
      >
        {kpi.change_percent > 0 ? '+' : ''}{kpi.change_percent}% from last period
      </p>
    )}
  </div>
);
