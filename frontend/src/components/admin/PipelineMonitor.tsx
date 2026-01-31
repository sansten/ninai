/**
 * Pipeline Monitor Component
 * ==========================
 * 
 * Displays real-time pipeline task queue status with SLA tracking,
 * resource utilization, queue depth visualization, historical trends,
 * manual task actions, dependency visualization, and export capabilities.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState } from 'react';
import { apiClient } from '@/lib/api';
import type { PipelineTask, PipelineStats } from '@/types/api';
import { CheckCircle, XCircle, Clock, AlertTriangle, Pause, Play, Download, RefreshCw, X, GitBranch } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import toast from 'react-hot-toast';

const TASK_TYPE_LABELS: Record<string, string> = {
  CONSOLIDATION: 'Consolidation',
  CRITIQUE: 'Critique',
  EVALUATION: 'Evaluation',
  FEEDBACK_LOOP: 'Feedback Loop',
  EMBEDDING_REFRESH: 'Embedding Refresh',
};

const STATUS_COLORS: Record<string, string> = {
  QUEUED: 'bg-blue-100 text-blue-800',
  RUNNING: 'bg-green-100 text-green-800',
  BLOCKED: 'bg-orange-100 text-orange-800',
  SUCCEEDED: 'bg-emerald-100 text-emerald-800',
  FAILED: 'bg-red-100 text-red-800',
};

const SLA_CATEGORY_COLORS: Record<string, string> = {
  critical: 'text-red-600',
  high: 'text-orange-600',
  medium: 'text-yellow-600',
  low: 'text-gray-600',
};

function formatDuration(ms: number | null | undefined): string {
  if (!ms) return '-';
  if (ms < 1000) return `${ms.toFixed(0)}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60000).toFixed(1)}m`;
}

function formatTokens(tokens: number | null | undefined): string {
  if (!tokens) return '-';
  if (tokens < 1000) return tokens.toString();
  if (tokens < 1000000) return `${(tokens / 1000).toFixed(1)}K`;
  return `${(tokens / 1000000).toFixed(1)}M`;
}

export function PipelineMonitor() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string>('');
  const [taskTypeFilter, setTaskTypeFilter] = useState<string>('');
  const [slaBreachedOnly, setSlaBreachedOnly] = useState(false);
  const [showTrends, setShowTrends] = useState(true);
  const [selectedTaskDeps, setSelectedTaskDeps] = useState<string | null>(null);
  const [autoAlertEnabled, setAutoAlertEnabled] = useState(false);
  const [autoAlertThreshold, setAutoAlertThreshold] = useState(15);

  // Fetch stats
  const statsQuery = useQuery<PipelineStats>({
    queryKey: ['pipeline-stats'],
    queryFn: () => apiClient.get('/admin/pipelines/stats').then(r => r.data),
    refetchInterval: 5000, // Refresh every 5 seconds
  });

  // Fetch tasks
  const tasksQuery = useQuery<PipelineTask[]>({
    queryKey: ['pipeline-tasks', statusFilter, taskTypeFilter, slaBreachedOnly],
    queryFn: () => {
      const params = new URLSearchParams();
      if (statusFilter) params.set('status_filter', statusFilter);
      if (taskTypeFilter) params.set('task_type', taskTypeFilter);
      if (slaBreachedOnly) params.set('sla_breached_only', 'true');
      params.set('limit', '100');
      
      return apiClient.get(`/admin/pipelines?${params.toString()}`).then(r => r.data);
    },
    refetchInterval: 3000, // Refresh every 3 seconds
  });

  const stats = statsQuery.data;
  const tasks = tasksQuery.data || [];

  // Fetch historical trends
  const trendsQuery = useQuery<Array<{timestamp: string; completed_tasks: number; sla_compliance_rate: number; avg_duration_ms: number | null}>>({
    queryKey: ['pipeline-stats-history'],
    queryFn: () => apiClient.get('/admin/pipelines/stats/history?hours=24').then(r => r.data),
    refetchInterval: 60000, // Refresh every minute
    enabled: showTrends,
  });

  // Fetch dependencies for selected task
  const depsQuery = useQuery<{task: PipelineTask; dependencies: PipelineTask[]; dependents: PipelineTask[]}>({
    queryKey: ['pipeline-task-deps', selectedTaskDeps],
    queryFn: () => apiClient.get(`/admin/pipelines/${selectedTaskDeps}/dependencies`).then(r => r.data),
    enabled: !!selectedTaskDeps,
  });

  // Cancel task mutation
  const cancelTask = useMutation({
    mutationFn: (taskId: string) => apiClient.post(`/admin/pipelines/${taskId}/cancel`),
    onSuccess: () => {
      toast.success('Task cancelled');
      qc.invalidateQueries({ queryKey: ['pipeline-tasks'] });
      qc.invalidateQueries({ queryKey: ['pipeline-stats'] });
    },
    onError: () => toast.error('Failed to cancel task'),
  });

  // Retry task mutation
  const retryTask = useMutation({
    mutationFn: (taskId: string) => apiClient.post(`/admin/pipelines/${taskId}/retry`),
    onSuccess: () => {
      toast.success('Task queued for retry');
      qc.invalidateQueries({ queryKey: ['pipeline-tasks'] });
      qc.invalidateQueries({ queryKey: ['pipeline-stats'] });
    },
    onError: () => toast.error('Failed to retry task'),
  });

  // Auto-create alert mutation
  const createAutoAlert = useMutation({
    mutationFn: (threshold: number) => 
      apiClient.post(`/admin/ops/alerts/auto-create?threshold=${threshold}&severity=high`),
    onSuccess: () => {
      toast.success('SLA breach alert created');
      setAutoAlertEnabled(true);
    },
    onError: () => toast.error('Failed to create alert'),
  });

  // Export function
  const handleExport = (format: 'csv' | 'json') => {
    const url = `/admin/pipelines/export?format=${format}&hours=24`;
    window.open(url, '_blank');
    toast.success(`Exporting ${format.toUpperCase()}...`);
  };


  return (
    <div className="space-y-6">
      {/* Action Bar */}
      <div className="bg-white border rounded-lg p-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowTrends(!showTrends)}
              className={`px-3 py-1.5 rounded text-sm font-medium ${showTrends ? 'bg-blue-100 text-blue-700' : 'bg-gray-100 text-gray-700'}`}
            >
              {showTrends ? 'Hide' : 'Show'} Trends
            </button>
            
            <div className="border-l pl-3 flex items-center gap-2">
              <Download className="w-4 h-4 text-gray-500" />
              <button
                onClick={() => handleExport('csv')}
                className="text-sm text-blue-600 hover:text-blue-700"
              >
                Export CSV
              </button>
              <span className="text-gray-300">|</span>
              <button
                onClick={() => handleExport('json')}
                className="text-sm text-blue-600 hover:text-blue-700"
              >
                Export JSON
              </button>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <label className="flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={autoAlertEnabled}
                onChange={(e) => {
                  if (e.target.checked && !autoAlertEnabled) {
                    createAutoAlert.mutate(autoAlertThreshold);
                  } else {
                    setAutoAlertEnabled(false);
                  }
                }}
                className="rounded"
              />
              <span>Auto-alert SLA breach</span>
            </label>
            {autoAlertEnabled && (
              <input
                type="number"
                value={autoAlertThreshold}
                onChange={(e) => setAutoAlertThreshold(Number(e.target.value))}
                className="w-16 px-2 py-1 border rounded text-sm"
                min="1"
                max="100"
              />
            )}
            {autoAlertEnabled && <span className="text-sm text-gray-500">% threshold</span>}
          </div>
        </div>
      </div>

      {/* Historical Trends Chart */}
      {showTrends && trendsQuery.data && (
        <div className="bg-white border rounded-lg p-4">
          <h3 className="font-semibold mb-4">SLA Compliance Trend (24h)</h3>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={trendsQuery.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(val) => new Date(val).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
              />
              <YAxis yAxisId="left" domain={[0, 100]} />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip 
                labelFormatter={(val) => new Date(val).toLocaleString()}
                formatter={(value: number, name: string) => {
                  if (name === 'sla_compliance_rate') return [`${value.toFixed(1)}%`, 'SLA Compliance'];
                  if (name === 'completed_tasks') return [value, 'Completed'];
                  if (name === 'avg_duration_ms') return [`${value.toFixed(0)}ms`, 'Avg Duration'];
                  return [value, name];
                }}
              />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="sla_compliance_rate" stroke="#10b981" name="SLA Compliance %" strokeWidth={2} />
              <Line yAxisId="right" type="monotone" dataKey="completed_tasks" stroke="#3b82f6" name="Completed Tasks" strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-white border rounded-lg p-4">
          <div className="text-sm text-gray-500">Queue Depth</div>
          <div className="text-2xl font-bold mt-1">{stats?.queued_tasks || 0}</div>
          <div className="text-xs text-gray-400 mt-1">
            {stats?.running_tasks || 0} running, {stats?.blocked_tasks || 0} blocked
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="text-sm text-gray-500">SLA Compliance</div>
          <div className="text-2xl font-bold mt-1 text-green-600">
            {stats?.sla_compliance_rate?.toFixed(1) || 0}%
          </div>
          <div className="text-xs text-gray-400 mt-1">
            {stats?.sla_breached_count || 0} breached
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="text-sm text-gray-500">Avg Queue Time</div>
          <div className="text-2xl font-bold mt-1">
            {formatDuration(stats?.avg_queue_time_ms)}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Exec: {formatDuration(stats?.avg_execution_time_ms)}
          </div>
        </div>

        <div className="bg-white border rounded-lg p-4">
          <div className="text-sm text-gray-500">Tokens/Hour</div>
          <div className="text-2xl font-bold mt-1">
            {formatTokens(stats?.total_tokens_consumed_last_hour)}
          </div>
          <div className="text-xs text-gray-400 mt-1">
            Avg: {formatTokens(stats?.avg_tokens_per_task)}/task
          </div>
        </div>
      </div>

      {/* Last Hour Throughput */}
      <div className="bg-white border rounded-lg p-4">
        <h3 className="font-semibold mb-3">Last Hour</h3>
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-green-600" />
            <span className="text-sm">
              <span className="font-semibold">{stats?.succeeded_tasks_last_hour || 0}</span> succeeded
            </span>
          </div>
          <div className="flex items-center gap-2">
            <XCircle className="w-4 h-4 text-red-600" />
            <span className="text-sm">
              <span className="font-semibold">{stats?.failed_tasks_last_hour || 0}</span> failed
            </span>
          </div>
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-orange-600" />
            <span className="text-sm">
              <span className="font-semibold">{stats?.sla_breached_count || 0}</span> SLA breaches
            </span>
          </div>
        </div>
      </div>

      {/* Queue Depth by Type */}
      {stats?.queue_depth_by_type && Object.keys(stats.queue_depth_by_type).length > 0 && (
        <div className="bg-white border rounded-lg p-4">
          <h3 className="font-semibold mb-3">Queue Depth by Type</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
            {Object.entries(stats.queue_depth_by_type).map(([type, count]) => (
              <div key={type} className="border rounded p-2">
                <div className="text-xs text-gray-500">{TASK_TYPE_LABELS[type] || type}</div>
                <div className="text-lg font-bold mt-1">{count}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="bg-white border rounded-lg p-4">
        <div className="flex flex-wrap gap-3">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="border rounded px-3 py-1.5 text-sm"
          >
            <option value="">All Statuses</option>
            <option value="queued">Queued</option>
            <option value="running">Running</option>
            <option value="blocked">Blocked</option>
            <option value="succeeded">Succeeded</option>
            <option value="failed">Failed</option>
          </select>

          <select
            value={taskTypeFilter}
            onChange={(e) => setTaskTypeFilter(e.target.value)}
            className="border rounded px-3 py-1.5 text-sm"
          >
            <option value="">All Task Types</option>
            {Object.entries(TASK_TYPE_LABELS).map(([value, label]) => (
              <option key={value} value={value}>{label}</option>
            ))}
          </select>

          <label className="flex items-center gap-2 border rounded px-3 py-1.5 text-sm cursor-pointer">
            <input
              type="checkbox"
              checked={slaBreachedOnly}
              onChange={(e) => setSlaBreachedOnly(e.target.checked)}
              className="rounded"
            />
            <span>SLA Breached Only</span>
          </label>

          <button
            onClick={() => {
              setStatusFilter('');
              setTaskTypeFilter('');
              setSlaBreachedOnly(false);
            }}
            className="text-sm text-blue-600 hover:text-blue-700 px-3"
          >
            Clear Filters
          </button>
        </div>
      </div>

      {/* Task List */}
      <div className="bg-white border rounded-lg overflow-hidden">
        <div className="px-4 py-3 border-b bg-gray-50">
          <h3 className="font-semibold">Pipeline Tasks ({tasks.length})</h3>
        </div>

        {tasksQuery.isLoading ? (
          <div className="p-8 text-center text-gray-500">Loading tasks...</div>
        ) : tasks.length === 0 ? (
          <div className="p-8 text-center text-gray-500">No tasks found</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b text-xs text-gray-600 uppercase">
                <tr>
                  <th className="px-4 py-3 text-left">Type</th>
                  <th className="px-4 py-3 text-left">Status</th>
                  <th className="px-4 py-3 text-left">Priority</th>
                  <th className="px-4 py-3 text-left">SLA</th>
                  <th className="px-4 py-3 text-left">Progress</th>
                  <th className="px-4 py-3 text-left">Resources</th>
                  <th className="px-4 py-3 text-left">Created</th>
                  <th className="px-4 py-3 text-left">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {tasks.map((task) => {
                  const now = new Date().getTime();
                  const createdAt = new Date(task.created_at).getTime();
                  const ageMinutes = Math.floor((now - createdAt) / 60000);

                  return (
                    <tr key={task.id} className="hover:bg-gray-50">
                      <td className="px-4 py-3 text-sm">
                        {TASK_TYPE_LABELS[task.task_type] || task.task_type}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${STATUS_COLORS[task.status] || 'bg-gray-100 text-gray-800'}`}>
                          {task.status}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm font-medium">
                        {task.priority}/10
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {task.sla_deadline ? (
                          <div>
                            <div className={`font-medium ${SLA_CATEGORY_COLORS[task.sla_category || 'low']}`}>
                              {task.sla_category?.toUpperCase() || 'N/A'}
                            </div>
                            {task.sla_breached ? (
                              <div className="text-xs text-red-600 font-semibold flex items-center gap-1">
                                <AlertTriangle className="w-3 h-3" />
                                BREACHED
                              </div>
                            ) : task.sla_remaining_ms !== null && task.sla_remaining_ms !== undefined ? (
                              <div className="text-xs text-gray-500">
                                {formatDuration(task.sla_remaining_ms)} left
                              </div>
                            ) : null}
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {task.blocked_by_quota ? (
                          <div className="text-orange-600 font-medium flex items-center gap-1">
                            <Pause className="w-3 h-3" />
                            Quota
                          </div>
                        ) : task.status === 'RUNNING' ? (
                          <div className="text-green-600 font-medium flex items-center gap-1">
                            <Play className="w-3 h-3" />
                            {task.attempts}/{task.max_attempts}
                          </div>
                        ) : task.status === 'FAILED' ? (
                          <div className="text-red-600 text-xs">
                            {task.last_error?.substring(0, 30)}...
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {task.actual_tokens || task.estimated_tokens ? (
                          <div>
                            <div className="font-medium">
                              {formatTokens(task.actual_tokens || task.estimated_tokens)} tokens
                            </div>
                            {task.duration_ms && (
                              <div className="text-xs text-gray-500">
                                {formatDuration(task.duration_ms)}
                              </div>
                            )}
                          </div>
                        ) : (
                          <span className="text-gray-400">-</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-500">
                        <div className="flex items-center gap-1">
                          <Clock className="w-3 h-3" />
                          {ageMinutes < 1 ? 'Just now' : `${ageMinutes}m ago`}
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {(task.status === 'QUEUED' || task.status === 'RUNNING' || task.status === 'BLOCKED') && (
                            <button
                              onClick={() => cancelTask.mutate(task.id)}
                              disabled={cancelTask.isPending}
                              className="text-red-600 hover:text-red-700 p-1 rounded hover:bg-red-50"
                              title="Cancel task"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          )}
                          {task.status === 'FAILED' && task.attempts < task.max_attempts && (
                            <button
                              onClick={() => retryTask.mutate(task.id)}
                              disabled={retryTask.isPending}
                              className="text-blue-600 hover:text-blue-700 p-1 rounded hover:bg-blue-50"
                              title="Retry task"
                            >
                              <RefreshCw className="w-4 h-4" />
                            </button>
                          )}
                          {(task.blocks_on_task_id || true) && (
                            <button
                              onClick={() => setSelectedTaskDeps(task.id)}
                              className="text-gray-600 hover:text-gray-700 p-1 rounded hover:bg-gray-50"
                              title="View dependencies"
                            >
                              <GitBranch className="w-4 h-4" />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Dependency Visualization Modal */}
      {selectedTaskDeps && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-4xl w-full max-h-[80vh] overflow-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold">Task Dependencies</h3>
              <button
                onClick={() => setSelectedTaskDeps(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {depsQuery.isLoading ? (
              <div className="p-8 text-center text-gray-500">Loading dependencies...</div>
            ) : depsQuery.data ? (
              <div className="space-y-6">
                {/* Current Task */}
                <div className="border-2 border-blue-500 rounded-lg p-4 bg-blue-50">
                  <div className="text-xs text-blue-600 font-semibold mb-1">CURRENT TASK</div>
                  <div className="font-medium">{TASK_TYPE_LABELS[depsQuery.data.task.task_type] || depsQuery.data.task.task_type}</div>
                  <div className="text-sm text-gray-600 mt-1">
                    Status: <span className={`font-medium ${STATUS_COLORS[depsQuery.data.task.status]?.replace('bg-', 'text-').replace('-100', '-700')}`}>
                      {depsQuery.data.task.status}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600">Priority: {depsQuery.data.task.priority}/10</div>
                </div>

                {/* Dependencies (blocks on) */}
                {depsQuery.data.dependencies.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <div className="text-sm font-semibold text-gray-700">DEPENDS ON</div>
                      <div className="text-xs text-gray-500">({depsQuery.data.dependencies.length})</div>
                    </div>
                    <div className="space-y-2">
                      {depsQuery.data.dependencies.map((dep) => (
                        <div key={dep.id} className="border rounded-lg p-3 bg-orange-50 border-orange-200">
                          <div className="font-medium text-sm">{TASK_TYPE_LABELS[dep.task_type] || dep.task_type}</div>
                          <div className="text-xs text-gray-600 mt-1">
                            Status: <span className="font-medium">{dep.status}</span> | Priority: {dep.priority}/10
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Dependents (blocked by this) */}
                {depsQuery.data.dependents.length > 0 && (
                  <div>
                    <div className="flex items-center gap-2 mb-3">
                      <div className="text-sm font-semibold text-gray-700">BLOCKS THESE TASKS</div>
                      <div className="text-xs text-gray-500">({depsQuery.data.dependents.length})</div>
                    </div>
                    <div className="space-y-2">
                      {depsQuery.data.dependents.map((dep) => (
                        <div key={dep.id} className="border rounded-lg p-3 bg-purple-50 border-purple-200">
                          <div className="font-medium text-sm">{TASK_TYPE_LABELS[dep.task_type] || dep.task_type}</div>
                          <div className="text-xs text-gray-600 mt-1">
                            Status: <span className="font-medium">{dep.status}</span> | Priority: {dep.priority}/10
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {depsQuery.data.dependencies.length === 0 && depsQuery.data.dependents.length === 0 && (
                  <div className="text-center text-gray-500 py-8">
                    This task has no dependencies
                  </div>
                )}
              </div>
            ) : null}
          </div>
        </div>
      )}
    </div>
  );
}
