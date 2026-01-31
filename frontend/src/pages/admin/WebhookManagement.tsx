/**
 * Webhook Management Page
 * Configure and monitor webhook subscriptions
 */

import React, { useState, useEffect } from 'react';
import { Plus, Trash2, Eye, RefreshCw, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import { useAdmin } from '../../hooks/useAdmin';

interface WebhookSubscription {
  id: string;
  url: string;
  is_active: boolean;
  event_types: string[];
  description: string | null;
  created_at: string;
}

interface WebhookDelivery {
  id: string;
  subscription_id: string;
  event_type: string;
  status: 'pending' | 'delivered' | 'failed';
  attempts: number;
  next_attempt_at: string;
  delivered_at: string | null;
  last_http_status: number | null;
  last_error: string | null;
  created_at: string;
}

interface CreateWebhookForm {
  url: string;
  eventTypes: string[];
  description: string;
}

const AVAILABLE_EVENTS = [
  'memory.created',
  'memory.updated',
  'memory.deleted',
  'memory.shared',
  'user.created',
  'user.updated',
  'user.deleted',
  'capability.issued',
  'capability.revoked',
  'audit.event',
];

const WebhookManagement: React.FC = () => {
  const { hasPermission } = useAdmin();
  const [webhooks, setWebhooks] = useState<WebhookSubscription[]>([]);
  const [deliveries, setDeliveries] = useState<Record<string, WebhookDelivery[]>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [testingId, setTestingId] = useState<string | null>(null);

  const [form, setForm] = useState<CreateWebhookForm>({
    url: '',
    eventTypes: [],
    description: '',
  });

  // Fetch webhooks
  useEffect(() => {
    fetchWebhooks();
  }, []);

  const fetchWebhooks = async () => {
    try {
      setLoading(true);
      const res = await fetch('/api/v1/webhooks', {
        credentials: 'include',
      });
      if (!res.ok) throw new Error('Failed to fetch webhooks');
      const data = await res.json();
      setWebhooks(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load webhooks');
    } finally {
      setLoading(false);
    }
  };

  const fetchDeliveries = async (webhookId: string) => {
    try {
      const res = await fetch(`/api/v1/webhooks/${webhookId}/deliveries?limit=20`, {
        credentials: 'include',
      });
      if (!res.ok) throw new Error('Failed to fetch deliveries');
      const data = await res.json();
      setDeliveries((prev) => ({ ...prev, [webhookId]: data.deliveries }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load deliveries');
    }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.url) {
      setError('URL is required');
      return;
    }

    try {
      const res = await fetch('/api/v1/webhooks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({
          url: form.url,
          event_types: form.eventTypes.length > 0 ? form.eventTypes : null,
          description: form.description || null,
        }),
      });

      if (!res.ok) throw new Error('Failed to create webhook');

      setForm({ url: '', eventTypes: [], description: '' });
      setShowCreateForm(false);
      await fetchWebhooks();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create webhook');
    }
  };

  const handleDelete = async (webhookId: string) => {
    if (!confirm('Delete this webhook?')) return;

    try {
      const res = await fetch(`/api/v1/webhooks/${webhookId}`, {
        method: 'DELETE',
        credentials: 'include',
      });

      if (!res.ok) throw new Error('Failed to delete webhook');
      await fetchWebhooks();
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete webhook');
    }
  };

  const handleToggleExpand = async (webhookId: string) => {
    if (expandedId === webhookId) {
      setExpandedId(null);
    } else {
      setExpandedId(webhookId);
      await fetchDeliveries(webhookId);
    }
  };

  const handleTestDelivery = async (webhookId: string) => {
    setTestingId(webhookId);
    try {
      // TODO: Call webhook test endpoint when available
      setError('Test delivery not yet implemented');
    } finally {
      setTestingId(null);
    }
  };

  if (!hasPermission('webhooks:manage')) {
    return (
      <div className="p-6 bg-yellow-50 border border-yellow-200 rounded-lg">
        <p className="text-yellow-800">You don't have permission to manage webhooks</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Webhooks</h1>
          <p className="text-gray-500 mt-1">Configure outgoing webhook subscriptions</p>
        </div>
        <button
          onClick={() => setShowCreateForm(!showCreateForm)}
          className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition"
        >
          <Plus className="w-4 h-4" />
          New Webhook
        </button>
      </div>

      {/* Error Alert */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
          <AlertCircle className="w-5 h-5 text-red-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-medium text-red-800">{error}</p>
          </div>
        </div>
      )}

      {/* Create Form */}
      {showCreateForm && (
        <div className="bg-white rounded-lg border border-gray-200 p-6 space-y-4">
          <h2 className="text-lg font-semibold text-gray-900">Create Webhook</h2>

          <form onSubmit={handleCreate} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Webhook URL
              </label>
              <input
                type="url"
                required
                placeholder="https://example.com/webhooks/ninai"
                value={form.url}
                onChange={(e) => setForm({ ...form, url: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Event Types (leave empty for all)
              </label>
              <div className="grid grid-cols-2 gap-2">
                {AVAILABLE_EVENTS.map((event) => (
                  <label key={event} className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={form.eventTypes.includes(event)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setForm({
                            ...form,
                            eventTypes: [...form.eventTypes, event],
                          });
                        } else {
                          setForm({
                            ...form,
                            eventTypes: form.eventTypes.filter((t) => t !== event),
                          });
                        }
                      }}
                      className="rounded border-gray-300"
                    />
                    <span className="text-sm text-gray-700">{event}</span>
                  </label>
                ))}
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description (optional)
              </label>
              <input
                type="text"
                placeholder="e.g., Analytics platform webhook"
                value={form.description}
                onChange={(e) => setForm({ ...form, description: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="flex gap-2 pt-4">
              <button
                type="submit"
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium"
              >
                Create Webhook
              </button>
              <button
                type="button"
                onClick={() => setShowCreateForm(false)}
                className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition"
              >
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Webhooks List */}
      {loading ? (
        <div className="text-center py-8 text-gray-500">Loading webhooks...</div>
      ) : webhooks.length === 0 ? (
        <div className="text-center py-8 bg-gray-50 rounded-lg border border-gray-200">
          <p className="text-gray-500">No webhooks configured yet</p>
        </div>
      ) : (
        <div className="space-y-3">
          {webhooks.map((webhook) => (
            <div key={webhook.id} className="bg-white rounded-lg border border-gray-200">
              {/* Summary */}
              <div className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="font-medium text-gray-900 break-all">{webhook.url}</h3>
                      <span
                        className={`px-2 py-1 text-xs font-semibold rounded ${
                          webhook.is_active
                            ? 'bg-green-100 text-green-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {webhook.is_active ? 'Active' : 'Inactive'}
                      </span>
                    </div>
                    {webhook.description && (
                      <p className="text-sm text-gray-600 mt-1">{webhook.description}</p>
                    )}
                    {webhook.event_types.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {webhook.event_types.map((event) => (
                          <span
                            key={event}
                            className="px-2 py-1 text-xs bg-blue-100 text-blue-800 rounded"
                          >
                            {event}
                          </span>
                        ))}
                      </div>
                    )}
                    {webhook.event_types.length === 0 && (
                      <p className="text-xs text-gray-500 mt-2">Subscribed to all events</p>
                    )}
                  </div>

                  <div className="flex items-center gap-2 ml-4">
                    <button
                      onClick={() => handleTestDelivery(webhook.id)}
                      disabled={testingId === webhook.id}
                      className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition disabled:opacity-50"
                      title="Test webhook delivery"
                    >
                      {testingId === webhook.id ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : (
                        <RefreshCw className="w-4 h-4" />
                      )}
                    </button>

                    <button
                      onClick={() => handleToggleExpand(webhook.id)}
                      className="p-2 text-blue-600 hover:bg-blue-50 rounded-lg transition"
                      title="View delivery history"
                    >
                      <Eye className="w-4 h-4" />
                    </button>

                    <button
                      onClick={() => handleDelete(webhook.id)}
                      className="p-2 text-red-600 hover:bg-red-50 rounded-lg transition"
                      title="Delete webhook"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>

              {/* Delivery History */}
              {expandedId === webhook.id && (
                <div className="border-t border-gray-200 bg-gray-50 p-4">
                  {!deliveries[webhook.id] ? (
                    <div className="text-center py-4 text-gray-500">Loading delivery history...</div>
                  ) : deliveries[webhook.id].length === 0 ? (
                    <div className="text-center py-4 text-gray-500">No deliveries yet</div>
                  ) : (
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                      {deliveries[webhook.id].map((delivery) => (
                        <div
                          key={delivery.id}
                          className="bg-white rounded p-3 text-sm border border-gray-200"
                        >
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center gap-2">
                                <span className="font-medium text-gray-900">{delivery.event_type}</span>
                                {delivery.status === 'delivered' && (
                                  <span className="flex items-center gap-1 text-green-700">
                                    <CheckCircle className="w-4 h-4" />
                                    Delivered
                                  </span>
                                )}
                                {delivery.status === 'pending' && (
                                  <span className="flex items-center gap-1 text-yellow-700">
                                    <Clock className="w-4 h-4" />
                                    Pending (attempt {delivery.attempts})
                                  </span>
                                )}
                                {delivery.status === 'failed' && (
                                  <span className="flex items-center gap-1 text-red-700">
                                    <AlertCircle className="w-4 h-4" />
                                    Failed
                                  </span>
                                )}
                              </div>
                              {delivery.last_error && (
                                <p className="text-red-600 text-xs mt-1">{delivery.last_error}</p>
                              )}
                              <p className="text-gray-500 text-xs mt-1">
                                {new Date(delivery.created_at).toLocaleString()}
                              </p>
                            </div>
                            {delivery.last_http_status && (
                              <span className="text-xs font-mono text-gray-600 ml-2">
                                {delivery.last_http_status}
                              </span>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default WebhookManagement;
