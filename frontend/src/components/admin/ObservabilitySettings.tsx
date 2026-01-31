/**
 * Observability & Logging Settings
 * ================================
 * 
 * Configure logging levels, sampling, tracing, webhooks.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { XMarkIcon, InformationCircleIcon } from '@heroicons/react/24/solid';
import { apiClient, getErrorMessage } from '@/lib/api';

interface LogConfig {
  service: string;
  module?: string;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR' | 'CRITICAL';
}

interface WebhookConfig {
  id: string;
  channel: 'email' | 'slack' | 'webhook' | string;
  event_types?: string[];
  enabled: boolean;
  target?: string;
  route?: string;
}

interface ObservabilityResponse {
  log_config: {
    services: LogConfig[];
  };
  tracing: {
    enabled: boolean;
    sample_rate: number;
    trace_provider: string;
  };
  metrics: {
    enabled: boolean;
    scrape_interval_seconds: number;
    prometheus_enabled: boolean;
  };
  webhooks: WebhookConfig[];
}

export function ObservabilitySettings() {
  const qc = useQueryClient();
  const [editingLog, setEditingLog] = useState<string | null>(null);
  const [editLogFormData, setEditLogFormData] = useState<Partial<LogConfig>>({});
  const [showAddWebhook, setShowAddWebhook] = useState(false);
  const [newWebhook, setNewWebhook] = useState<Partial<WebhookConfig>>({
    channel: 'webhook',
    event_types: [],
    enabled: true,
  });

  const configQuery = useQuery<ObservabilityResponse>({
    queryKey: ['admin', 'observability'],
    queryFn: async () => {
      const res = await apiClient.get('/admin/observability');
      return res.data;
    },
  });

  const updateLogConfig = useMutation({
    mutationFn: async (config: LogConfig) => {
      const res = await apiClient.put('/admin/observability/log-config', config);
      return res.data;
    },
    onSuccess: () => {
      toast.success('Log config updated');
      setEditingLog(null);
      qc.invalidateQueries({ queryKey: ['admin', 'observability'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const toggleMetrics = useMutation({
    mutationFn: async (enabled: boolean) => {
      const res = await apiClient.post('/admin/observability/metrics', { enabled });
      return res.data;
    },
    onSuccess: () => {
      toast.success('Metrics config updated');
      qc.invalidateQueries({ queryKey: ['admin', 'observability'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const createWebhook = useMutation({
    mutationFn: async (webhook: Partial<WebhookConfig>) => {
      const res = await apiClient.post('/admin/observability/webhooks', webhook);
      return res.data;
    },
    onSuccess: () => {
      toast.success('Webhook created');
      setNewWebhook({ channel: 'webhook', event_types: [], enabled: true });
      setShowAddWebhook(false);
      qc.invalidateQueries({ queryKey: ['admin', 'observability'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const deleteWebhook = useMutation({
    mutationFn: async (id: string) => {
      await apiClient.delete(`/admin/observability/webhooks/${id}`);
    },
    onSuccess: () => {
      toast.success('Webhook deleted');
      qc.invalidateQueries({ queryKey: ['admin', 'observability'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  if (configQuery.isLoading) return <p className="text-sm text-gray-500">Loading observability config…</p>;
  if (configQuery.isError) return <p className="text-sm text-red-600">Failed to load configuration</p>;

  return (
    <div className="space-y-6">
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-3 flex gap-3">
        <InformationCircleIcon className="w-5 h-5 text-purple-600 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-purple-900 text-sm">Observability & Logging</h3>
          <p className="text-purple-800 text-xs mt-1">Control logging levels per service, configure distributed tracing, manage sampling rates, and set up webhooks for monitoring events. Webhooks can send data to HTTP endpoints or Slack channels.</p>
        </div>
      </div>

      {/* Logging Configuration */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">Logging Configuration</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="py-2 px-4 text-left font-medium text-gray-700">Service</th>
                <th className="py-2 px-4 text-left font-medium text-gray-700">Module</th>
                <th className="py-2 px-4 text-left font-medium text-gray-700">Log Level</th>
                <th className="py-2 px-4 text-left font-medium text-gray-700">Sampling</th>
                <th className="py-2 px-4 text-left font-medium text-gray-700">Tracing</th>
                <th className="py-2 px-4 text-left font-medium text-gray-700">Action</th>
              </tr>
            </thead>
            <tbody>
              {configQuery.data?.log_config?.services?.map((config) => (
                <tr key={`${config.service}-${config.module}`} className="border-b hover:bg-gray-50">
                  <td className="py-2 px-4 font-medium text-gray-900">{config.service}</td>
                  <td className="py-2 px-4 text-gray-700">{config.module || '—'}</td>
                  <td className="py-2 px-4 text-gray-700">{config.level}</td>
                  <td className="py-2 px-4 text-gray-700">100%</td>
                  <td className="py-2 px-4">
                    <span className={clsx(
                      'px-2 py-1 rounded text-xs font-medium',
                      'bg-blue-100 text-blue-800'
                    )}>
                      On
                    </span>
                  </td>
                  <td className="py-2 px-4">
                    <button
                      onClick={() => {
                        setEditingLog(`${config.service}-${config.module}`);
                        setEditLogFormData(config);
                      }}
                      className="text-primary-600 hover:text-primary-800 text-xs font-medium"
                    >
                      Edit
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Edit Log Config Form */}
        {editingLog && (
          <div className="mt-4 p-4 bg-gray-50 border border-gray-200 rounded-lg space-y-4">
            <div className="flex items-center justify-between">
              <h4 className="font-medium text-gray-900">Edit Logging Configuration</h4>
              <button
                onClick={() => setEditingLog(null)}
                className="text-gray-600 hover:text-gray-800 text-sm"
              >
                ✕
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label">Service</label>
                <input
                  type="text"
                  value={editLogFormData.service || ''}
                  onChange={(e) => setEditLogFormData({ ...editLogFormData, service: e.target.value })}
                  className="input w-full"
                  disabled
                />
              </div>

              <div>
                <label className="label">Module</label>
                <input
                  type="text"
                  value={editLogFormData.module || ''}
                  onChange={(e) => setEditLogFormData({ ...editLogFormData, module: e.target.value })}
                  className="input w-full"
                  disabled
                />
              </div>

              <div>
                <label className="label">Log Level</label>
                <select
                  value={editLogFormData.level || 'INFO'}
                  onChange={(e) => setEditLogFormData({ ...editLogFormData, level: e.target.value as any })}
                  className="input w-full"
                >
                  <option value="DEBUG">DEBUG</option>
                  <option value="INFO">INFO</option>
                  <option value="WARNING">WARNING</option>
                  <option value="ERROR">ERROR</option>
                  <option value="CRITICAL">CRITICAL</option>
                </select>
              </div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => {
                  if (!editLogFormData.service) {
                    toast.error('Select a service to update');
                    return;
                  }
                  updateLogConfig.mutate(editLogFormData as LogConfig);
                }}
                className="btn-primary text-sm"
              >
                Save
              </button>
              <button
                onClick={() => setEditingLog(null)}
                className="btn-secondary text-sm"
              >
                Cancel
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Tracing Sample Rate */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">Tracing Configuration</h3>
        <div>
          <label className="label">Global Trace Sample Rate</label>
          <div className="flex items-center gap-4">
            <input
              type="range"
              min="0"
              max="100"
              step="5"
              value={Math.round(((configQuery.data?.tracing?.sample_rate ?? 0) as number) * 100)}
              onChange={(e) => {
                // Could add mutation here
              }}
              className="flex-1"
            />
            <span className="text-sm font-medium text-gray-900 w-12">
              {Math.round(((configQuery.data?.tracing?.sample_rate ?? 0) as number) * 100)}%
            </span>
          </div>
          <p className="text-xs text-gray-500 mt-2">
            Sample rate for distributed tracing (increases overhead). Recommended: 10-20%.
          </p>
        </div>
      </div>

      {/* Metrics Configuration */}
      <div className="card">
        <h3 className="font-semibold text-gray-900 mb-4">Metrics Scraping</h3>
        <div className="flex items-center justify-between">
          <div>
            <p className="font-medium text-gray-900">Prometheus Metrics Endpoint</p>
            <p className="text-sm text-gray-500 mt-1">Expose /metrics for Prometheus scraping</p>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={configQuery.data?.metrics?.prometheus_enabled ?? false}
              onChange={(e) => toggleMetrics.mutate(e.target.checked)}
              className="sr-only peer"
            />
            <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
          </label>
        </div>
      </div>

      {/* Webhooks */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-900">Alert Webhooks</h3>
          <button
            onClick={() => setShowAddWebhook(!showAddWebhook)}
            className="btn-secondary text-sm"
          >
            {showAddWebhook ? 'Cancel' : 'Add Webhook'}
          </button>
        </div>

        {showAddWebhook && (
          <div className="mb-4 p-4 border border-gray-200 rounded-lg bg-gray-50 space-y-4">
            <div>
              <label className="label">Name</label>
              <input
                type="text"
                placeholder="e.g., 'Slack Alerts'"
                className="input w-full"
                value={newWebhook.name || ''}
                onChange={(e) => setNewWebhook({ ...newWebhook, name: e.target.value })}
              />
            </div>
            <div>
              <label className="label">Channel</label>
              <select
                className="input w-full"
                value={newWebhook.channel || 'webhook'}
                onChange={(e) => setNewWebhook({ ...newWebhook, channel: e.target.value as any })}
              >
                <option value="webhook">Webhook (HTTP POST)</option>
                <option value="slack">Slack</option>
                <option value="email">Email</option>
              </select>
            </div>
            <div>
              <label className="label">URL/Address</label>
              <input
                type="text"
                placeholder="https://hooks.slack.com/... or email@example.com"
                className="input w-full"
                value={newWebhook.url || ''}
                onChange={(e) => setNewWebhook({ ...newWebhook, url: e.target.value })}
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => createWebhook.mutate(newWebhook)}
                disabled={createWebhook.isPending}
                className="btn-primary"
              >
                Create
              </button>
            </div>
          </div>
        )}

        <div className="space-y-2">
          {configQuery.data?.webhooks && configQuery.data.webhooks.length > 0 ? (
            configQuery.data.webhooks.map((hook) => (
              <div key={hook.id} className="flex items-center justify-between p-3 border border-gray-200 rounded-lg">
                <div>
                  <p className="font-medium text-gray-900">{hook.target || hook.route || 'Webhook'}</p>
                  <p className="text-xs text-gray-500">{hook.channel}</p>
                </div>
                <button
                  onClick={() => deleteWebhook.mutate(hook.id)}
                  className="text-red-600 hover:text-red-800 text-sm font-medium"
                >
                  Delete
                </button>
              </div>
            ))
          ) : (
            <p className="text-sm text-gray-500 text-center py-4">No webhooks configured</p>
          )}
        </div>
      </div>
    </div>
  );
}

import clsx from 'clsx';
