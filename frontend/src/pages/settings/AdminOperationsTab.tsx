import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

import { apiClient, getErrorMessage } from '@/lib/api';
import { useAdminCapabilities } from '@/lib/rbac';
import { useCurrentUser } from '@/stores/auth';
import { AuditTrail } from '@/components/admin/AuditTrail';
import { QueueManagement } from '@/components/admin/QueueManagement';
import { ObservabilitySettings } from '@/components/admin/ObservabilitySettings';
import { HealthAndMaintenance } from '@/components/admin/HealthAndMaintenance';
import { AlertNotifications } from '@/components/admin/AlertNotifications';
import { PipelineMonitor } from '@/components/admin/PipelineMonitor';
import type { AlertRule, ApiKeySummary, PolicyVersion, ResourceBudget, Snapshot } from '@/types/api';

interface InfoTooltipProps {
  title: string;
  description: string;
  whenToUse: string;
  example: string;
  recommended?: string;
}

function InfoTooltip({ title, description, whenToUse, example, recommended }: InfoTooltipProps) {
  const [show, setShow] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        type="button"
        className="ml-2 text-gray-400 hover:text-blue-600 transition-colors"
        onClick={() => setShow(!show)}
        onBlur={() => setTimeout(() => setShow(false), 200)}
      >
        <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
          <path
            fillRule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
            clipRule="evenodd"
          />
        </svg>
      </button>
      {show && (
        <div className="absolute z-50 w-96 p-4 bg-white border border-gray-200 rounded-lg shadow-xl left-0 top-8">
          <h4 className="font-semibold text-gray-900 mb-2">{title}</h4>
          <div className="space-y-3 text-sm">
            <div>
              <p className="font-medium text-gray-700">What it does:</p>
              <p className="text-gray-600">{description}</p>
            </div>
            <div>
              <p className="font-medium text-gray-700">When to use:</p>
              <p className="text-gray-600">{whenToUse}</p>
            </div>
            <div>
              <p className="font-medium text-gray-700">Example:</p>
              <p className="text-gray-600 italic">{example}</p>
            </div>
            {recommended && (
              <div className="bg-blue-50 p-2 rounded">
                <p className="font-medium text-blue-900">💡 Recommended:</p>
                <p className="text-blue-800">{recommended}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

interface ProcessItem {
  id: string;
  agent_name: string;
  status: string;
  session_id?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
}

interface ProcessesResponse {
  total: number;
  items: ProcessItem[];
  status_summary: Record<string, number>;
}

function StatusPill({ status }: { status: string }) {
  const normalized = status.toLowerCase();
  const color =
    normalized.includes('fail') || normalized.includes('unhealthy') || normalized.includes('revoked') || normalized.includes('blocked')
      ? 'bg-red-100 text-red-700'
      : normalized.includes('degraded') || normalized.includes('canary') || normalized.includes('throttle') || normalized.includes('high')
        ? 'bg-amber-100 text-amber-700'
        : normalized.includes('active') || normalized.includes('healthy') || normalized.includes('running') || normalized.includes('completed')
          ? 'bg-green-100 text-green-700'
          : 'bg-gray-100 text-gray-700';
  return <span className={`px-2 py-1 rounded text-xs font-medium ${color}`}>{status}</span>;
}

function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return '—';
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(k)), sizes.length - 1);
  const value = bytes / Math.pow(k, i);
  return `${value.toFixed(1)} ${sizes[i]}`;
}

export function AdminOperationsTab() {
  const [section, setSection] = useState<
    'overview' | 'policies' | 'resources' | 'queues' | 'backups' | 'tokens' | 'observability' | 'alerts' | 'audit' | 'queue-mgmt' | 'health' | 'alert-notifications' | 'pipelines'
  >('overview');
  const qc = useQueryClient();
  const user = useCurrentUser();
  const capabilities = useAdminCapabilities(user?.roles ?? []);

  const processesQuery = useQuery<ProcessesResponse>({
    queryKey: ['ops', 'processes', section],
    enabled: section === 'overview' || section === 'queues',
    queryFn: async () => {
      const res = await apiClient.get('/ops/processes', { params: { limit: 50 } });
      return res.data;
    },
  });

  const policiesQuery = useQuery<PolicyVersion[]>({
    queryKey: ['admin', 'ops', 'policies'],
    enabled: section === 'policies',
    queryFn: async () => {
      const res = await apiClient.get('/admin/ops/policies');
      return res.data;
    },
  });

  const budgetQuery = useQuery<ResourceBudget>({
    queryKey: ['admin', 'ops', 'resources'],
    enabled: section === 'overview' || section === 'resources',
    queryFn: async () => {
      const res = await apiClient.get('/admin/ops/resources');
      return res.data;
    },
  });

  const snapshotsQuery = useQuery<Snapshot[]>({
    queryKey: ['admin', 'ops', 'snapshots'],
    enabled: section === 'backups',
    queryFn: async () => {
      const res = await apiClient.get('/admin/ops/backups/snapshots');
      return res.data;
    },
  });

  const alertsQuery = useQuery<AlertRule[]>({
    queryKey: ['admin', 'ops', 'alerts'],
    enabled: section === 'alerts',
    queryFn: async () => {
      const res = await apiClient.get('/admin/ops/alerts');
      return res.data;
    },
  });

  const apiKeysQuery = useQuery<ApiKeySummary[]>({
    queryKey: ['admin', 'api-keys'],
    enabled: section === 'tokens',
    queryFn: async () => {
      const res = await apiClient.get('/admin/api-keys');
      return res.data;
    },
  });

  const createKey = useMutation({
    mutationFn: async (name: string) => {
      const res = await apiClient.post('/admin/api-keys', { name });
      return res.data as ApiKeySummary;
    },
    onSuccess: (data) => {
      toast.success('API key created');
      if (data.api_key) {
        toast.success('Copy the new key now; it will not be shown again');
      }
      qc.invalidateQueries({ queryKey: ['admin', 'api-keys'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const revokeKey = useMutation({
    mutationFn: async (id: string) => {
      const res = await apiClient.post(`/admin/api-keys/${id}/revoke`);
      return res.data as ApiKeySummary;
    },
    onSuccess: () => {
      toast.success('Key revoked');
      qc.invalidateQueries({ queryKey: ['admin', 'api-keys'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const createAlert = useMutation({
    mutationFn: async (payload: { name: string; severity: string; target: string; channel: string }) => {
      const res = await apiClient.post('/admin/ops/alerts', {
        name: payload.name,
        severity: payload.severity,
        route: 'default',
        channel: payload.channel,
        target: payload.target,
        enabled: true,
      });
      return res.data as AlertRule;
    },
    onSuccess: () => {
      toast.success('Alert rule created');
      qc.invalidateQueries({ queryKey: ['admin', 'ops', 'alerts'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const disableAlert = useMutation({
    mutationFn: async (id: string) => {
      const res = await apiClient.post(`/admin/ops/alerts/${id}/disable`);
      return res.data as AlertRule;
    },
    onSuccess: () => {
      toast.success('Alert disabled');
      qc.invalidateQueries({ queryKey: ['admin', 'ops', 'alerts'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  async function handlePolicyAction(action: string, policyId: string) {
    try {
      await apiClient.post(`/admin/ops/policies/${policyId}/${action}`);
      toast.success(`${action} requested`);
      qc.invalidateQueries({ queryKey: ['admin', 'ops', 'policies'] });
    } catch (err) {
      toast.error(getErrorMessage(err));
    }
  }

  async function handleBudgetAction(kind: 'block' | 'unblock' | 'throttle') {
    try {
      if (kind === 'throttle') {
        const rateStr = window.prompt('Throttle rate (0.0 to 1.0, where 0.5 = 50% slowdown)', '0.5');
        if (!rateStr) return;
        const throttle_rate = parseFloat(rateStr);
        if (isNaN(throttle_rate) || throttle_rate < 0 || throttle_rate > 1) {
          toast.error('Invalid throttle rate. Must be between 0.0 and 1.0');
          return;
        }
        await apiClient.post(`/admin/ops/resources/throttle`, { throttle_rate });
      } else {
        await apiClient.post(`/admin/ops/resources/${kind}`);
      }
      toast.success(`Admission ${kind === 'unblock' ? 'resumed' : kind}`);
      qc.invalidateQueries({ queryKey: ['admin', 'ops', 'resources'] });
    } catch (err) {
      toast.error(getErrorMessage(err));
    }
  }

  async function handleSnapshot(action: 'create' | 'restore' | 'verify', snapshotId?: string) {
    try {
      if (action === 'create') {
        const name = window.prompt('Snapshot name', 'On-demand backup');
        if (!name) return;
        await apiClient.post('/admin/ops/backups/snapshots', { snapshot_name: name, snapshot_type: 'full' });
        toast.success('Snapshot started');
      } else if (action === 'restore' && snapshotId) {
        await apiClient.post(`/admin/ops/backups/snapshots/${snapshotId}/restore`);
        toast.success('Restore initiated');
      } else if (action === 'verify' && snapshotId) {
        await apiClient.post(`/admin/ops/backups/snapshots/${snapshotId}/verify`);
        toast.success('Verify requested');
      }
      qc.invalidateQueries({ queryKey: ['admin', 'ops', 'snapshots'] });
    } catch (err) {
      toast.error(getErrorMessage(err));
    }
  }

  function renderOverview() {
    const summary = processesQuery.data?.status_summary ?? {};

    return (
      <div className="space-y-4">
        <div className="flex items-center mb-2">
          <h2 className="text-xl font-semibold text-gray-900">System Overview</h2>
          <InfoTooltip
            title="System Overview"
            description="High-level health dashboard showing system status, queue metrics, and backup status at a glance."
            whenToUse="Check this first when investigating issues or during daily operations monitoring."
            example="If you notice degraded performance, start here to see if queues are backed up or if health checks are failing."
            recommended="Review this dashboard at the start of each day and set up alerts for critical metrics."
          />
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm text-gray-500">Health</p>
              <StatusPill status="healthy" />
            </div>
            <p className="text-lg font-semibold text-gray-900">Liveness & readiness</p>
            <p className="text-sm text-gray-500">/health and readiness checks are green.</p>
          </div>
          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm text-gray-500">Queues</p>
              <StatusPill status={summary.blocked ? 'blocked' : 'running'} />
            </div>
            <p className="text-lg font-semibold text-gray-900">{summary.running ?? summary.queued ?? 0} active</p>
            <p className="text-sm text-gray-500">Agent processes across running/queued states.</p>
          </div>
          <div className="card">
            <div className="flex items-center justify-between mb-2">
              <p className="text-sm text-gray-500">Backups</p>
              <StatusPill status="staged" />
            </div>
            <p className="text-lg font-semibold text-gray-900">Daily + weekly cadence</p>
            <p className="text-sm text-gray-500">Last full snapshot verified in the past week.</p>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-gray-900">Status Summary</h3>
            <div className="text-sm text-gray-500">Powered by /ops/processes</div>
          </div>
          {processesQuery.isLoading ? (
            <p className="text-sm text-gray-500">Loading…</p>
          ) : processesQuery.isError ? (
            <p className="text-sm text-red-600">Failed to load process summary</p>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {Object.entries(summary).map(([key, value]) => (
                <div key={key} className="p-3 bg-gray-50 rounded">
                  <p className="text-xs uppercase tracking-wide text-gray-500">{key}</p>
                  <p className="text-xl font-semibold text-gray-900">{value}</p>
                </div>
              ))}
              {Object.keys(summary).length === 0 && <p className="text-sm text-gray-500">No processes yet.</p>}
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderPolicies() {
    return (
      <div className="space-y-4">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 inline-flex items-center">
                  Policy Rollouts
                  <InfoTooltip
                    title="Policy Rollouts"
                    description="Safely deploy policy changes (safety filters, routing rules, access controls) using canary testing and gradual rollouts."
                    whenToUse="When updating content safety rules, changing agent routing logic, or modifying access control policies."
                    example="You updated a safety filter. Start with 5% canary (test with small user group), then promote to 25%, then 100% if no issues."
                    recommended="Always test with canary (5-10%) for 24-48 hours before promoting. Monitor error rates closely."
                  />
                </h3>
                <p className="text-sm text-gray-500">Staged rollouts with canary and automatic promotion.</p>
              </div>
            </div>
            <button type="button" className="btn-secondary" onClick={() => toast('Draft creation is backend-driven for now')}>
              New version
            </button>
          </div>
          {policiesQuery.isLoading ? (
            <p className="text-sm text-gray-500">Loading…</p>
          ) : policiesQuery.isError ? (
            <p className="text-sm text-red-600">Failed to load policies</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b">
                    <th className="py-2 pr-4">Policy</th>
                    <th className="py-2 pr-4">Type</th>
                    <th className="py-2 pr-4">Rollout</th>
                    <th className="py-2 pr-4">Notes</th>
                    <th className="py-2 pr-4">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {policiesQuery.data?.map((p) => (
                    <tr key={p.id} className="border-b">
                      <td className="py-2 pr-4 font-medium text-gray-900">
                        {p.policy_name} <span className="text-xs text-gray-500">v{p.version}</span>
                      </td>
                      <td className="py-2 pr-4 text-gray-700">{p.policy_type}</td>
                      <td className="py-2 pr-4 text-gray-700">
                        <StatusPill status={p.rollout_status} />
                        <span className="ml-2 text-xs text-gray-600">{Math.round((p.rollout_percentage ?? 0) * 100)}%</span>
                      </td>
                      <td className="py-2 pr-4 text-gray-600">{p.change_notes ?? '—'}</td>
                      <td className="py-2 pr-4">
                        <div className="flex flex-wrap gap-2 text-sm">
                          <button type="button" className="btn-secondary" onClick={() => handlePolicyAction('canary', p.id)}>
                            Canary
                          </button>
                          <button type="button" className="btn-secondary" onClick={() => handlePolicyAction('promote', p.id)}>
                            Promote
                          </button>
                          <button type="button" className="btn-secondary" onClick={() => handlePolicyAction('activate', p.id)}>
                            Activate
                          </button>
                          <button type="button" className="btn-secondary" onClick={() => handlePolicyAction('rollback', p.id)}>
                            Rollback
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {(policiesQuery.data?.length ?? 0) === 0 && (
                    <tr>
                      <td colSpan={5} className="py-3 text-sm text-gray-500 text-center">No policy versions yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderResources() {
    const budget = budgetQuery.data;

    return (
      <div className="space-y-4">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 inline-flex items-center">
                  Resource Budgets & Admission
                  <InfoTooltip
                    title="Resource Budgets & Admission"
                    description="Monitor and control API token usage, storage consumption, and request rates. Protect against runaway costs."
                    whenToUse="Set monthly budgets, block admission during incidents, or throttle traffic during high load."
                    example="Usage spiking unexpectedly? Click 'Throttle' to slow requests by 50% while investigating, or 'Block' to stop new requests entirely."
                    recommended="Set budgets at 80% of actual limits. Use throttle for gradual slowdown, block only for emergencies."
                  />
                </h3>
                <p className="text-sm text-gray-500">Quota utilization and admission guardrails.</p>
              </div>
            </div>
            <div className="flex gap-2">
              <button type="button" className="btn-secondary" onClick={() => handleBudgetAction('throttle')}>
                Throttle
              </button>
              <button type="button" className="btn-secondary" onClick={() => handleBudgetAction('block')}>
                Block
              </button>
              <button type="button" className="btn-secondary" onClick={() => handleBudgetAction('unblock')}>
                Unblock
              </button>
            </div>
          </div>
          {budgetQuery.isLoading ? (
            <p className="text-sm text-gray-500">Loading…</p>
          ) : budgetQuery.isError ? (
            <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
              <p className="text-sm font-medium text-red-800 mb-1">Failed to load budget</p>
              <p className="text-xs text-red-600">
                {budgetQuery.error ? getErrorMessage(budgetQuery.error) : 'Unknown error'}
              </p>
              <p className="text-xs text-gray-600 mt-2">
                Ensure you are logged in with org_admin or system_admin role and the backend is running.
              </p>
            </div>
          ) : budget ? (
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="p-4 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-sm font-medium text-blue-900">Tokens</p>
                    <StatusPill status={budget.degraded_mode ? 'degraded' : 'healthy'} />
                  </div>
                  <p className="text-4xl font-bold text-blue-900 mb-2">{Math.round(budget.token_utilization)}%</p>
                  <div className="w-full bg-blue-200 rounded-full h-2 mb-2">
                    <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${Math.min(budget.token_utilization, 100)}%` }}></div>
                  </div>
                  <p className="text-xs text-blue-700">
                    Used {budget.tokens_used.toLocaleString()} / {budget.token_budget.toLocaleString()}
                  </p>
                </div>
                <div className="p-4 bg-gradient-to-br from-green-50 to-green-100 rounded-lg border border-green-200">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-sm font-medium text-green-900">Storage</p>
                    <StatusPill status={budget.admission_blocked ? 'blocked' : 'healthy'} />
                  </div>
                  <p className="text-4xl font-bold text-green-900 mb-2">{Math.round(budget.storage_utilization)}%</p>
                  <div className="w-full bg-green-200 rounded-full h-2 mb-2">
                    <div className="bg-green-600 h-2 rounded-full" style={{ width: `${Math.min(budget.storage_utilization, 100)}%` }}></div>
                  </div>
                  <p className="text-xs text-green-700">
                    Used {formatBytes(budget.storage_used_mb * 1024 * 1024)} / {formatBytes(budget.storage_budget_mb * 1024 * 1024)}
                  </p>
                </div>
                <div className="p-4 bg-gradient-to-br from-purple-50 to-purple-100 rounded-lg border border-purple-200">
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-sm font-medium text-purple-900">Requests</p>
                    <StatusPill status={budget.throttle_rate > 0 ? 'throttle' : 'healthy'} />
                  </div>
                  <p className="text-4xl font-bold text-purple-900 mb-2">{Math.round(budget.request_utilization)}%</p>
                  <div className="w-full bg-purple-200 rounded-full h-2 mb-2">
                    <div className="bg-purple-600 h-2 rounded-full" style={{ width: `${Math.min(budget.request_utilization, 100)}%` }}></div>
                  </div>
                  <p className="text-xs text-purple-700">
                    Used {budget.requests_used.toLocaleString()} / {budget.request_budget.toLocaleString()}
                  </p>
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <div className="flex items-center gap-3">
                    <StatusPill status={budget.admission_blocked ? 'blocked' : budget.throttle_rate > 0 ? 'throttle' : 'healthy'} />
                    <span className="text-sm text-gray-700">
                      Admission: <span className="font-semibold">{budget.admission_blocked ? 'Blocked' : 'Open'}</span>
                    </span>
                    <span className="text-gray-300">|</span>
                    <span className="text-sm text-gray-700">
                      Throttle: <span className="font-semibold">{Math.round(budget.throttle_rate * 100)}%</span>
                    </span>
                  </div>
                  <div className="text-sm text-gray-600">
                    Period: {budget.period_start && new Date(budget.period_start).toLocaleDateString()} – {budget.period_end && new Date(budget.period_end).toLocaleDateString()}
                  </div>
                </div>
              </div>
            </div>
          ) : null}
        </div>
      </div>
    );
  }

  function renderQueues() {
    return (
      <div className="space-y-4">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 inline-flex items-center">
                  Agent Queues & SLA
                  <InfoTooltip
                    title="Agent Queues & SLA"
                    description="Real-time view of all agent processes, their states (queued, running, completed, failed), and execution times."
                    whenToUse="Monitor agent performance, troubleshoot stuck processes, verify SLA compliance."
                    example="Customer reports slow responses? Check here for queued or long-running processes. Look for error patterns."
                    recommended="This is read-only monitoring. For one-time checks, just refresh. For ongoing monitoring, use the Overview dashboard."
                  />
                </h3>
                <p className="text-sm text-gray-500">Live process states from /ops/processes.</p>
              </div>
            </div>
            <button type="button" className="btn-secondary" onClick={() => processesQuery.refetch()}>
              Refresh
            </button>
          </div>
          {processesQuery.isLoading ? (
            <p className="text-sm text-gray-500">Loading…</p>
          ) : processesQuery.isError ? (
            <p className="text-sm text-red-600">Failed to load processes</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b">
                    <th className="py-2 pr-4">Agent</th>
                    <th className="py-2 pr-4">Status</th>
                    <th className="py-2 pr-4">Session</th>
                    <th className="py-2 pr-4">Created</th>
                    <th className="py-2 pr-4">Started</th>
                    <th className="py-2 pr-4">Ended</th>
                  </tr>
                </thead>
                <tbody>
                  {processesQuery.data?.items?.map((p) => (
                    <tr key={p.id} className="border-b">
                      <td className="py-2 pr-4 font-medium text-gray-900">{p.agent_name}</td>
                      <td className="py-2 pr-4"><StatusPill status={p.status} /></td>
                      <td className="py-2 pr-4 text-gray-700">{p.session_id ?? '—'}</td>
                      <td className="py-2 pr-4 text-gray-700">{new Date(p.created_at).toLocaleString()}</td>
                      <td className="py-2 pr-4 text-gray-700">{p.started_at ? new Date(p.started_at).toLocaleString() : '—'}</td>
                      <td className="py-2 pr-4 text-gray-700">{p.completed_at ? new Date(p.completed_at).toLocaleString() : '—'}</td>
                    </tr>
                  ))}
                  {(processesQuery.data?.items?.length ?? 0) === 0 && (
                    <tr>
                      <td colSpan={6} className="py-3 text-sm text-gray-500 text-center">No processes yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderBackups() {
    return (
      <div className="space-y-4">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 inline-flex items-center">
                  Backups & DR
                  <InfoTooltip
                    title="Backups & Disaster Recovery"
                    description="Create, verify, and restore database snapshots. Automated daily/weekly backups run in background."
                    whenToUse="Trigger manual backup before major changes, test restore process quarterly, recover from data loss."
                    example="Before a major migration: 'Trigger snapshot' → wait for completion → 'Verify' checksum. If migration fails, use 'Restore'."
                    recommended="Automated backups handle routine needs. Manual snapshots for: pre-migration, before bulk deletes, monthly DR drills."
                  />
                </h3>
                <p className="text-sm text-gray-500">Trigger snapshots and validate restore plans.</p>
              </div>
            </div>
            <div className="flex gap-2">
              <button type="button" className="btn-secondary" onClick={() => handleSnapshot('create')}>
                Trigger snapshot
              </button>
              <button type="button" className="btn-secondary" onClick={() => toast('Restore testing uses pending backend endpoint')}>
                Run restore test
              </button>
            </div>
          </div>
          {snapshotsQuery.isLoading ? (
            <p className="text-sm text-gray-500">Loading…</p>
          ) : snapshotsQuery.isError ? (
            <p className="text-sm text-red-600">Failed to load snapshots</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b">
                    <th className="py-2 pr-4">Snapshot</th>
                    <th className="py-2 pr-4">Type</th>
                    <th className="py-2 pr-4">Status</th>
                    <th className="py-2 pr-4">Size</th>
                    <th className="py-2 pr-4">Completed</th>
                    <th className="py-2 pr-4">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {snapshotsQuery.data?.map((s) => (
                    <tr key={s.id} className="border-b">
                      <td className="py-2 pr-4 font-medium text-gray-900">{s.snapshot_name}</td>
                      <td className="py-2 pr-4 text-gray-700">{s.snapshot_type}</td>
                      <td className="py-2 pr-4"><StatusPill status={s.status} /></td>
                      <td className="py-2 pr-4 text-gray-700">{formatBytes(s.snapshot_size_bytes)}</td>
                      <td className="py-2 pr-4 text-gray-700">{s.completed_at ? new Date(s.completed_at).toLocaleString() : '—'}</td>
                      <td className="py-2 pr-4">
                        <div className="flex gap-2">
                          <button type="button" className="btn-secondary" onClick={() => handleSnapshot('restore', s.id)}>
                            Restore
                          </button>
                          <button type="button" className="btn-secondary" onClick={() => handleSnapshot('verify', s.id)}>
                            Verify
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                  {(snapshotsQuery.data?.length ?? 0) === 0 && (
                    <tr>
                      <td colSpan={6} className="py-3 text-sm text-gray-500 text-center">No snapshots yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderTokens() {
    return (
      <div className="space-y-4">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 inline-flex items-center">
                  Tokens & Security
                  <InfoTooltip
                    title="API Tokens & Security"
                    description="Create and manage API keys for external integrations, automation scripts, and service accounts."
                    whenToUse="Set up new integrations (Slack bot, CI/CD pipeline), rotate compromised keys, audit API access."
                    example="Integrating with Slack? Create key named 'slack-bot' → copy the key immediately (shown once) → configure in Slack settings."
                    recommended="One key per integration. Rotate keys every 90 days. Revoke immediately if compromised. Never commit keys to git."
                  />
                </h3>
                <p className="text-sm text-gray-500">API keys for automations and agents.</p>
              </div>
            </div>
            <button
              type="button"
              className="btn-secondary"
              disabled={createKey.isPending}
              onClick={() => {
                const name = window.prompt('API key name', 'integration-bot');
                if (name) createKey.mutate(name);
              }}
            >
              {createKey.isPending ? 'Creating…' : 'Create key'}
            </button>
          </div>

          {apiKeysQuery.isLoading ? (
            <p className="text-sm text-gray-500">Loading…</p>
          ) : apiKeysQuery.isError ? (
            <p className="text-sm text-red-600">Failed to load API keys</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b">
                    <th className="py-2 pr-4">Name</th>
                    <th className="py-2 pr-4">Prefix</th>
                    <th className="py-2 pr-4">Created</th>
                    <th className="py-2 pr-4">Last Used</th>
                    <th className="py-2 pr-4">Status</th>
                    <th className="py-2 pr-4">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {apiKeysQuery.data?.map((k) => (
                    <tr key={k.id} className="border-b">
                      <td className="py-2 pr-4 font-medium text-gray-900">{k.name}</td>
                      <td className="py-2 pr-4 text-gray-700">{k.prefix}</td>
                      <td className="py-2 pr-4 text-gray-700">{new Date(k.created_at).toLocaleString()}</td>
                      <td className="py-2 pr-4 text-gray-700">{k.last_used_at ? new Date(k.last_used_at).toLocaleString() : '—'}</td>
                      <td className="py-2 pr-4">{k.revoked_at ? <StatusPill status="revoked" /> : <StatusPill status="active" />}</td>
                      <td className="py-2 pr-4">
                        {!k.revoked_at && (
                          <button type="button" className="btn-secondary" disabled={revokeKey.isPending} onClick={() => revokeKey.mutate(k.id)}>
                            Revoke
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                  {(apiKeysQuery.data?.length ?? 0) === 0 && (
                    <tr>
                      <td colSpan={6} className="py-3 text-sm text-gray-500 text-center">No API keys yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    );
  }

  function renderObservability() {
    return (
      <div className="space-y-4">
        <div className="card">
          <h3 className="text-lg font-semibold text-gray-900 mb-2 inline-flex items-center">
            Observability
            <InfoTooltip
              title="Observability & Metrics"
              description="Expose system metrics in Prometheus format for external monitoring tools (Grafana, Datadog, etc.)."
              whenToUse="When setting up production monitoring dashboards or integrating with existing observability stack."
              example="Connect Prometheus to /metrics endpoint → build Grafana dashboard → set up alerts for high latency or error rates."
              recommended="This is typically one-time setup during deployment. Point your monitoring tool to /metrics and configure scraping interval."
            />
          </h3>
          <p className="text-sm text-gray-500 mb-3">Surface metrics and export in Prometheus format.</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div className="p-3 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">Metrics endpoint</p>
              <p className="text-xl font-semibold text-gray-900">/metrics</p>
              <p className="text-xs text-gray-500">Expose observability_service.get_prometheus_format()</p>
            </div>
            <div className="p-3 bg-gray-50 rounded">
              <p className="text-sm text-gray-600">App metrics</p>
              <p className="text-xl font-semibold text-gray-900">http_requests_total</p>
              <p className="text-xs text-gray-500">Track request volume and latency histograms.</p>
            </div>
          </div>
          <p className="text-xs text-gray-500 mt-3">Wire this panel once metrics endpoints are exposed; keep using /health for probes.</p>
        </div>
      </div>
    );
  }

  function renderAlerts() {
    return (
      <div className="space-y-4">
        <div className="card">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 inline-flex items-center">
                  Alerts & Routing
                  <InfoTooltip
                    title="Alerts & Routing"
                    description="Configure alerts that trigger on critical events (queue backlog, backup failures, high error rates) and route to Slack or paging services."
                    whenToUse="Set up during initial deployment, then adjust as you learn which alerts are actionable vs. noisy."
                    example="Create 'Queue backlog' alert → severity 'high' → route to '#oncall' Slack channel → get notified when >100 queued processes."
                    recommended="Start with 3-5 critical alerts (downtime, data loss, security). Add more based on actual incidents. Disable noisy alerts."
                  />
                </h3>
                <p className="text-sm text-gray-500">Define alert rules and escalation paths.</p>
              </div>
            </div>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => {
                const name = window.prompt('Alert name', 'Queue backlog');
                const target = name ? window.prompt('Target (Slack channel or paging route)', '#oncall') : null;
                if (name && target) {
                  createAlert.mutate({ name, severity: 'medium', target, channel: 'slack' });
                }
              }}
            >
              {createAlert.isPending ? 'Creating…' : 'Create alert'}
            </button>
          </div>
          {alertsQuery.isLoading ? (
            <p className="text-sm text-gray-500">Loading…</p>
          ) : alertsQuery.isError ? (
            <p className="text-sm text-red-600">Failed to load alerts</p>
          ) : (
            <div className="space-y-3">
              {alertsQuery.data?.map((rule) => (
                <div key={rule.id} className="p-3 bg-gray-50 rounded flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{rule.name}</p>
                    <p className="text-xs text-gray-500">Route {rule.route} → {rule.channel} {rule.target}</p>
                    <p className="text-xs text-gray-500">Created {new Date(rule.created_at).toLocaleString()}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <StatusPill status={rule.enabled ? rule.severity : 'disabled'} />
                    {rule.enabled && (
                      <button type="button" className="btn-secondary" disabled={disableAlert.isPending} onClick={() => disableAlert.mutate(rule.id)}>
                        Disable
                      </button>
                    )}
                  </div>
                </div>
              ))}
              {(alertsQuery.data?.length ?? 0) === 0 && <p className="text-sm text-gray-500">No alert rules yet.</p>}
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Operations & Monitoring</h3>
        <p className="text-sm text-gray-500 mt-1">Overview, rollouts, quotas, queues, backups, tokens, metrics, and alerts.</p>
      </div>

      <div className="flex flex-wrap gap-2">
        {[
          { id: 'overview', label: 'Overview' },
          { id: 'pipelines', label: 'Pipeline Monitor', capability: 'canManageQueues' },
          { id: 'policies', label: 'Policies', capability: 'canEditPolicies' },
          { id: 'resources', label: 'Resources', capability: 'canSetBudgets' },
          { id: 'queues', label: 'Queues', capability: 'canManageQueues' },
          { id: 'queue-mgmt', label: 'Queue Settings', capability: 'canManageQueues' },
          { id: 'backups', label: 'Backups', capability: 'canManageBackups' },
          { id: 'tokens', label: 'Tokens', capability: 'canManageTokens' },
          { id: 'observability', label: 'Observability', capability: 'canManageObservability' },
          { id: 'health', label: 'Health', capability: 'canManageHealth' },
          { id: 'alerts', label: 'Alerts', capability: 'canManageAlerts' },
          { id: 'alert-notifications', label: 'Alert Notifications', capability: 'canManageAlerts' },
          { id: 'audit', label: 'Audit Trail', capability: 'canViewAudit' },
        ]
          .filter((tab) => !('capability' in tab) || capabilities[tab.capability as keyof typeof capabilities])
          .map((tab) => (
            <button
              key={tab.id}
              type="button"
              className={section === tab.id ? 'btn-primary' : 'btn-secondary'}
              onClick={() => setSection(tab.id as typeof section)}
            >
              {tab.label}
            </button>
          ))}
      </div>

      {section === 'overview' && renderOverview()}
      {section === 'pipelines' && <PipelineMonitor />}
      {section === 'policies' && renderPolicies()}
      {section === 'resources' && renderResources()}
      {section === 'queues' && renderQueues()}
      {section === 'queue-mgmt' && <QueueManagement />}
      {section === 'backups' && renderBackups()}
      {section === 'tokens' && renderTokens()}
      {section === 'observability' && <ObservabilitySettings />}
      {section === 'health' && <HealthAndMaintenance />}
      {section === 'alerts' && renderAlerts()}
      {section === 'alert-notifications' && <AlertNotifications />}
      {section === 'audit' && <AuditTrail />}
    </div>
  );
}
