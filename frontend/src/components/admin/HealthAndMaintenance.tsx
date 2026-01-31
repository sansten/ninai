/**
 * Health & Maintenance Settings
 * =============================
 * 
 * Readiness overrides, startup gates, feature flags.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { ExclamationTriangleIcon, CheckCircleIcon, XCircleIcon, InformationCircleIcon } from '@heroicons/react/24/solid';
import clsx from 'clsx';
import { apiClient, getErrorMessage } from '@/lib/api';

interface HealthConfig {
  maintenance_mode: boolean;
  maintenance_message?: string;
  readiness_bypass: boolean;
  feature_flags: Record<string, boolean>;
  dependencies: Record<string, { status: string; latency_ms: number }>;
}

export function HealthAndMaintenance() {
  const qc = useQueryClient();
  const [maintenanceMessage, setMaintenanceMessage] = useState('');

  const healthQuery = useQuery<HealthConfig>({
    queryKey: ['admin', 'health'],
    queryFn: async () => {
      const res = await apiClient.get('/admin/health');
      return res.data;
    },
  });

  const toggleMaintenanceMode = useMutation({
    mutationFn: async (enabled: boolean) => {
      const res = await apiClient.post('/admin/health/maintenance', {
        enabled,
        message: maintenanceMessage,
      });
      return res.data;
    },
    onSuccess: () => {
      toast.success('Maintenance mode updated');
      qc.invalidateQueries({ queryKey: ['admin', 'health'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const toggleReadinessBypass = useMutation({
    mutationFn: async (enabled: boolean) => {
      const confirmed = window.confirm(
        enabled
          ? 'Bypass readiness checks? This allows traffic to unhealthy dependencies.'
          : 'Re-enable readiness checks?'
      );
      if (!confirmed) throw new Error('Cancelled');
      const res = await apiClient.post('/admin/health/readiness-bypass', { enabled });
      return res.data;
    },
    onSuccess: () => {
      toast.success('Readiness bypass updated');
      qc.invalidateQueries({ queryKey: ['admin', 'health'] });
    },
    onError: (err) => {
      if (getErrorMessage(err) !== 'Cancelled') {
        toast.error(getErrorMessage(err));
      }
    },
  });

  const toggleFeatureFlag = useMutation({
    mutationFn: async ({ flag, enabled }: { flag: string; enabled: boolean }) => {
      const res = await apiClient.post(`/admin/health/feature-flags/${flag}`, { enabled });
      return res.data;
    },
    onSuccess: () => {
      toast.success('Feature flag updated');
      qc.invalidateQueries({ queryKey: ['admin', 'health'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  if (healthQuery.isLoading) return <p className="text-sm text-gray-500">Loading health configuration…</p>;
  if (healthQuery.isError) return <p className="text-sm text-red-600">Failed to load health config</p>;

  const config = healthQuery.data!;

  return (
    <div className="space-y-6">
      <div className="bg-green-50 border border-green-200 rounded-lg p-3 flex gap-3">
        <InformationCircleIcon className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-green-900 text-sm">Health & Maintenance</h3>
          <p className="text-green-800 text-xs mt-1">Enable maintenance mode to stop accepting new work, monitor dependency health, use feature flags for gradual rollouts, and manage emergency overrides for critical situations.</p>
        </div>
      </div>

      {/* Maintenance Mode */}
      <div className="card border-2 border-amber-200 bg-amber-50">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h3 className="font-semibold text-gray-900">Maintenance Mode</h3>
            <p className="text-sm text-gray-600 mt-1">
              Temporarily take system offline for maintenance while preserving health checks.
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={config.maintenance_mode}
              onChange={(e) => toggleMaintenanceMode.mutate(e.target.checked)}
              className="sr-only peer"
              disabled={toggleMaintenanceMode.isPending}
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>

        {config.maintenance_mode_enabled && (
          <div className="space-y-3">
            <textarea
              placeholder="Maintenance message (shown to users)"
              className="input w-full"
              value={maintenanceMessage}
              onChange={(e) => setMaintenanceMessage(e.target.value)}
              rows={3}
            />
            <p className="text-xs text-amber-700">
              Current message: {config.maintenance_message || 'No message set'}
            </p>
          </div>
        )}
      </div>

      {/* Dependency Health Checks */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">Dependency Health</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(config.dependencies || {}).map(([name, check]) => (
            <div key={name} className="p-4 border border-gray-200 rounded-lg">
              <div className="flex items-center justify-between mb-2">
                <p className="font-medium text-gray-900">{name}</p>
                <span className={clsx(
                  'px-2 py-1 rounded text-xs font-medium',
                  check.status === 'healthy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                )}>
                  {check.status}
                </span>
              </div>
              <p className="text-sm text-gray-600">Latency: {check.latency_ms}ms</p>
            </div>
          ))}
        </div>
      </div>

      {/* Readiness Checks Bypass */}
      <div className="card border-2 border-red-200 bg-red-50">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="font-semibold text-gray-900">Readiness Check Bypass</h3>
            <p className="text-sm text-red-700 mt-1">
              ⚠️ Allow traffic even if dependencies are unhealthy. Use only during incidents.
            </p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={config.readiness_bypass}
              onChange={(e) => toggleReadinessBypass.mutate(e.target.checked)}
              className="sr-only peer"
              disabled={toggleReadinessBypass.isPending}
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-red-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-red-600"></div>
          </label>
        </div>
      </div>

      {/* Feature Flags */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">Optional Subsystems</h3>
        <div className="space-y-3">
          {Object.entries(config.feature_flags || {}).map(([flag, enabled]) => (
            <div key={flag} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
              <div>
                <p className="font-medium text-gray-900">{flag}</p>
                <p className="text-xs text-gray-500">
                  {flag === 'redis_cache' && 'In-memory caching layer'}
                  {flag === 'qdrant_search' && 'Vector search engine'}
                  {flag === 'async_workers' && 'Background task workers'}
                </p>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={enabled}
                  onChange={(e) => toggleFeatureFlag.mutate({ flag, enabled: e.target.checked })}
                  className="sr-only peer"
                  disabled={toggleFeatureFlag.isPending}
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
              </label>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
