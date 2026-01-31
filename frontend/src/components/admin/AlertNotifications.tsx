/**
 * Alert Notifications Component
 * ============================
 * 
 * Create and manage admin alerts that send notifications via email or Slack.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { TrashIcon, InformationCircleIcon, BellIcon } from '@heroicons/react/24/solid';
import clsx from 'clsx';
import { apiClient, getErrorMessage } from '@/lib/api';

interface AlertNotification {
  id: string;
  name: string;
  enabled: boolean;
  channel: 'email' | 'slack';
  recipients: string[]; // email addresses or Slack webhook URLs
  conditions: {
    event_type: string;
    severity?: 'info' | 'warning' | 'error' | 'critical';
    threshold?: number;
  };
  created_at: string;
}

interface AlertNotificationResponse {
  notifications: AlertNotification[];
}

const EVENT_TYPES = [
  { value: 'policy_rollout', label: 'Policy Rollout' },
  { value: 'queue_drain', label: 'Queue Drained' },
  { value: 'maintenance_mode', label: 'Maintenance Mode Enabled' },
  { value: 'dependency_unhealthy', label: 'Dependency Unhealthy' },
  { value: 'feature_flag_change', label: 'Feature Flag Changed' },
  { value: 'backup_failed', label: 'Backup Failed' },
  { value: 'token_revoked', label: 'Token Revoked' },
  { value: 'high_error_rate', label: 'High Error Rate' },
  { value: 'quota_exceeded', label: 'Quota Exceeded' },
];

const SEVERITY_LEVELS = ['info', 'warning', 'error', 'critical'] as const;

export function AlertNotifications() {
  const qc = useQueryClient();
  const [showAddAlert, setShowAddAlert] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    channel: 'email' as 'email' | 'slack',
    recipients: '',
    event_type: 'policy_rollout',
    severity: 'error' as typeof SEVERITY_LEVELS[number],
  });

  const alertsQuery = useQuery<AlertNotificationResponse>({
    queryKey: ['admin', 'alert-notifications'],
    queryFn: async () => {
      const res = await apiClient.get('/admin/alerts/notifications');
      return res.data;
    },
  });

  const createAlert = useMutation({
    mutationFn: async (data: typeof formData) => {
      const res = await apiClient.post('/admin/alerts/notifications', {
        name: data.name,
        enabled: true,
        channel: data.channel,
        recipients: data.recipients.split(',').map(r => r.trim()),
        conditions: {
          event_type: data.event_type,
          severity: data.severity,
        },
      });
      return res.data;
    },
    onSuccess: () => {
      toast.success('Alert notification created');
      setFormData({
        name: '',
        channel: 'email',
        recipients: '',
        event_type: 'policy_rollout',
        severity: 'error',
      });
      setShowAddAlert(false);
      qc.invalidateQueries({ queryKey: ['admin', 'alert-notifications'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const toggleAlert = useMutation({
    mutationFn: async ({ id, enabled }: { id: string; enabled: boolean }) => {
      const res = await apiClient.put(`/admin/alerts/notifications/${id}`, { enabled });
      return res.data;
    },
    onSuccess: () => {
      toast.success('Alert updated');
      qc.invalidateQueries({ queryKey: ['admin', 'alert-notifications'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const deleteAlert = useMutation({
    mutationFn: async (id: string) => {
      const confirmed = window.confirm('Delete this alert notification?');
      if (!confirmed) throw new Error('Cancelled');
      const res = await apiClient.delete(`/admin/alerts/notifications/${id}`);
      return res.data;
    },
    onSuccess: () => {
      toast.success('Alert deleted');
      qc.invalidateQueries({ queryKey: ['admin', 'alert-notifications'] });
    },
    onError: (err) => {
      if (getErrorMessage(err) !== 'Cancelled') {
        toast.error(getErrorMessage(err));
      }
    },
  });

  if (alertsQuery.isLoading) return <p className="text-sm text-gray-500">Loading alerts…</p>;
  if (alertsQuery.isError) return <p className="text-sm text-red-600">Failed to load alerts</p>;

  return (
    <div className="space-y-6">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-3 flex gap-3">
        <InformationCircleIcon className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-blue-900 text-sm">Alert Notifications</h3>
          <p className="text-blue-800 text-xs mt-1">Set up automated alerts that notify you via email or Slack when important events occur. Configure recipients and conditions for each alert.</p>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-900 flex items-center gap-2">
            <BellIcon className="w-5 h-5 text-blue-600" />
            Active Alerts
          </h3>
          <button
            onClick={() => setShowAddAlert(!showAddAlert)}
            className="btn-secondary text-sm"
          >
            {showAddAlert ? 'Cancel' : '+ Add Alert'}
          </button>
        </div>

        {/* Add New Alert Form */}
        {showAddAlert && (
          <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg space-y-4">
            <div>
              <label className="label">Alert Name</label>
              <input
                type="text"
                placeholder="e.g., Policy Rollout Notifications"
                className="input w-full"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              />
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label">Notification Channel</label>
                <select
                  className="input w-full"
                  value={formData.channel}
                  onChange={(e) => setFormData({ ...formData, channel: e.target.value as 'email' | 'slack' })}
                >
                  <option value="email">📧 Email</option>
                  <option value="slack">💬 Slack</option>
                </select>
              </div>

              <div>
                <label className="label">
                  {formData.channel === 'email' ? 'Email Addresses' : 'Slack Webhooks'}
                </label>
                <input
                  type="text"
                  placeholder={formData.channel === 'email' ? 'user@example.com, admin@example.com' : 'https://hooks.slack.com/...'}
                  className="input w-full text-xs"
                  value={formData.recipients}
                  onChange={(e) => setFormData({ ...formData, recipients: e.target.value })}
                />
                <p className="text-xs text-gray-500 mt-1">
                  {formData.channel === 'email' ? 'Separate multiple emails with commas' : 'Enter Slack webhook URL'}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label">Event Type</label>
                <select
                  className="input w-full"
                  value={formData.event_type}
                  onChange={(e) => setFormData({ ...formData, event_type: e.target.value })}
                >
                  {EVENT_TYPES.map((event) => (
                    <option key={event.value} value={event.value}>
                      {event.label}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="label">Minimum Severity</label>
                <select
                  className="input w-full"
                  value={formData.severity}
                  onChange={(e) => setFormData({ ...formData, severity: e.target.value as any })}
                >
                  {SEVERITY_LEVELS.map((level) => (
                    <option key={level} value={level}>
                      {level.charAt(0).toUpperCase() + level.slice(1)}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <button
              onClick={() => createAlert.mutate(formData)}
              disabled={createAlert.isPending || !formData.name || !formData.recipients}
              className="btn-primary w-full text-sm"
            >
              {createAlert.isPending ? 'Creating…' : 'Create Alert'}
            </button>
          </div>
        )}

        {/* Alerts List */}
        <div className="space-y-3">
          {alertsQuery.data?.notifications && alertsQuery.data.notifications.length > 0 ? (
            alertsQuery.data.notifications.map((alert) => (
              <div
                key={alert.id}
                className="p-4 border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={alert.enabled}
                          onChange={(e) => toggleAlert.mutate({ id: alert.id, enabled: e.target.checked })}
                          className="sr-only peer"
                        />
                        <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-primary-600"></div>
                      </label>
                      <h4 className="font-medium text-gray-900">{alert.name}</h4>
                      <span className={clsx(
                        'px-2 py-1 rounded text-xs font-medium',
                        alert.channel === 'email' ? 'bg-blue-100 text-blue-800' : 'bg-purple-100 text-purple-800'
                      )}>
                        {alert.channel === 'email' ? '📧 Email' : '💬 Slack'}
                      </span>
                    </div>

                    <div className="space-y-1 text-sm text-gray-600">
                      <p>
                        <span className="font-medium">Event:</span>{' '}
                        {EVENT_TYPES.find(e => e.value === alert.conditions.event_type)?.label || alert.conditions.event_type}
                      </p>
                      <p>
                        <span className="font-medium">Severity:</span>{' '}
                        <span className={clsx(
                          'px-2 py-0.5 rounded text-xs font-medium',
                          alert.conditions.severity === 'critical' ? 'bg-red-100 text-red-800' :
                          alert.conditions.severity === 'error' ? 'bg-orange-100 text-orange-800' :
                          alert.conditions.severity === 'warning' ? 'bg-yellow-100 text-yellow-800' :
                          'bg-blue-100 text-blue-800'
                        )}>
                          {alert.conditions.severity}
                        </span>
                      </p>
                      <p>
                        <span className="font-medium">Recipients:</span> {alert.recipients.join(', ')}
                      </p>
                    </div>
                  </div>

                  <button
                    onClick={() => deleteAlert.mutate(alert.id)}
                    className="text-red-600 hover:text-red-800 transition-colors"
                  >
                    <TrashIcon className="w-5 h-5" />
                  </button>
                </div>
              </div>
            ))
          ) : (
            <p className="text-sm text-gray-500 text-center py-6">
              No alerts configured. Create one to get started!
            </p>
          )}
        </div>
      </div>

      {/* Quick Setup Guide */}
      <div className="card bg-gradient-to-br from-blue-50 to-indigo-50 border border-blue-200">
        <h3 className="font-semibold text-gray-900 mb-3">Quick Setup Guide</h3>
        <div className="space-y-3 text-sm text-gray-700">
          <div>
            <p className="font-medium text-blue-900">📧 Email Alerts</p>
            <p>Enter comma-separated email addresses to send notifications directly to inboxes.</p>
          </div>
          <div>
            <p className="font-medium text-blue-900">💬 Slack Alerts</p>
            <p>
              Create a Slack Incoming Webhook in your workspace settings, then paste the webhook URL here.{' '}
              <a href="https://api.slack.com/messaging/webhooks" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:underline">
                Learn more
              </a>
            </p>
          </div>
          <div>
            <p className="font-medium text-blue-900">⚙️ Conditions</p>
            <p>Alerts will only send when the specified event occurs with the selected severity level or higher.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
