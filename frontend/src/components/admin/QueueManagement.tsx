/**
 * Queue Management Settings
 * ========================
 * 
 * Configure SLA bands, queue priority, and queue operations.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import { InformationCircleIcon, PauseIcon, PlayIcon, TrashIcon } from '@heroicons/react/24/solid';
import { apiClient, getErrorMessage } from '@/lib/api';

interface QueueConfig {
  name: string;
  priority_weight: number;
  max_retries: number;
  retry_backoff_ms: number;
  concurrency: number;
  status: 'active' | 'paused';
}

interface QueueResponse {
  queues: QueueConfig[];
}

export function QueueManagement() {
  const qc = useQueryClient();
  const [editingQueue, setEditingQueue] = useState<string | null>(null);
  const [formData, setFormData] = useState<Partial<QueueConfig>>({});

  const queuesQuery = useQuery<QueueResponse>({
    queryKey: ['admin', 'queues'],
    queryFn: async () => {
      const res = await apiClient.get('/admin/ops/queues');
      return res.data;
    },
  });

  const updateQueue = useMutation({
    mutationFn: async (data: QueueConfig) => {
      const res = await apiClient.put(`/admin/ops/queues/${data.queue_name}`, data);
      return res.data;
    },
    onSuccess: () => {
      toast.success('Queue updated');
      setEditingQueue(null);
      qc.invalidateQueries({ queryKey: ['admin', 'queues'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const pauseQueue = useMutation({
    mutationFn: async (queueName: string) => {
      const res = await apiClient.post(`/admin/ops/queues/${queueName}/pause`);
      return res.data;
    },
    onSuccess: () => {
      toast.success('Queue paused');
      qc.invalidateQueries({ queryKey: ['admin', 'queues'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const resumeQueue = useMutation({
    mutationFn: async (queueName: string) => {
      const res = await apiClient.post(`/admin/ops/queues/${queueName}/resume`);
      return res.data;
    },
    onSuccess: () => {
      toast.success('Queue resumed');
      qc.invalidateQueries({ queryKey: ['admin', 'queues'] });
    },
    onError: (err) => toast.error(getErrorMessage(err)),
  });

  const drainQueue = useMutation({
    mutationFn: async (queueName: string) => {
      const confirmed = window.confirm(
        `Drain queue "${queueName}"? Pending tasks will be cancelled.`
      );
      if (!confirmed) throw new Error('Cancelled');
      const res = await apiClient.post(`/admin/ops/queues/${queueName}/drain`);
      return res.data;
    },
    onSuccess: () => {
      toast.success('Queue drained');
      qc.invalidateQueries({ queryKey: ['admin', 'queues'] });
    },
    onError: (err) => {
      if (getErrorMessage(err) !== 'Cancelled') {
        toast.error(getErrorMessage(err));
      }
    },
  });

  if (queuesQuery.isLoading) return <p className="text-sm text-gray-500">Loading queues…</p>;
  if (queuesQuery.isError) return <p className="text-sm text-red-600">Failed to load queues</p>;

  return (
    <div className="space-y-4">
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex gap-3">
        <InformationCircleIcon className="w-5 h-5 text-amber-600 flex-shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-amber-900 text-sm">Queue Management</h3>
          <p className="text-amber-800 text-xs mt-1">Configure how tasks are processed with priority weights, retry policies, and concurrency limits. Use Pause to stop processing, Resume to restart, and Drain to empty the queue gracefully.</p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="py-3 px-4 text-left font-medium text-gray-700">Queue</th>
              <th className="py-3 px-4 text-left font-medium text-gray-700">Priority</th>
              <th className="py-3 px-4 text-left font-medium text-gray-700">Max Attempts</th>
              <th className="py-3 px-4 text-left font-medium text-gray-700">Backoff (s)</th>
              <th className="py-3 px-4 text-left font-medium text-gray-700">Concurrency</th>
              <th className="py-3 px-4 text-left font-medium text-gray-700">Status</th>
              <th className="py-3 px-4 text-left font-medium text-gray-700">Actions</th>
            </tr>
          </thead>
          <tbody>
            {queuesQuery.data?.queues.map((queue) => (
              <tr key={queue.name} className="border-b hover:bg-gray-50">
                <td className="py-3 px-4 font-medium text-gray-900">{queue.name}</td>
                <td className="py-3 px-4 text-gray-700">{queue.priority_weight}</td>
                <td className="py-3 px-4 text-gray-700">{queue.max_retries}</td>
                <td className="py-3 px-4 text-gray-700">{queue.retry_backoff_ms}ms</td>
                <td className="py-3 px-4 text-gray-700">{queue.concurrency}</td>
                <td className="py-3 px-4">
                  <span className={clsx(
                    'px-2 py-1 rounded-full text-xs font-medium',
                    queue.status === 'paused' ? 'bg-yellow-100 text-yellow-800' : 'bg-green-100 text-green-800'
                  )}>
                    {queue.status === 'paused' ? 'Paused' : 'Running'}
                  </span>
                </td>
                <td className="py-3 px-4 space-x-2">
                  {queue.status === 'paused' ? (
                    <button
                      onClick={() => resumeQueue.mutate(queue.name)}
                      className="text-blue-600 hover:text-blue-800 text-xs font-medium"
                    >
                      Resume
                    </button>
                  ) : (
                    <button
                      onClick={() => pauseQueue.mutate(queue.name)}
                      className="text-gray-600 hover:text-gray-800 text-xs font-medium"
                    >
                      Pause
                    </button>
                  )}
                  <button
                    onClick={() => drainQueue.mutate(queue.name)}
                    className="text-red-600 hover:text-red-800 text-xs font-medium"
                  >
                    Drain
                  </button>
                  <button
                    onClick={() => {
                      setEditingQueue(queue.name);
                      setFormData(queue);
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

      {editingQueue && (
        <div className="mt-6 p-4 border border-gray-200 rounded-lg bg-gray-50">
          <h3 className="font-semibold text-gray-900 mb-4">Edit {editingQueue}</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="label">Priority Weight</label>
              <input
                type="number"
                className="input w-full"
                value={formData.priority_weight || 0}
                onChange={(e) => setFormData({ ...formData, priority_weight: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <label className="label">Max Attempts</label>
              <input
                type="number"
                className="input w-full"
                value={formData.max_attempts || 0}
                onChange={(e) => setFormData({ ...formData, max_attempts: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <label className="label">Backoff (seconds)</label>
              <input
                type="number"
                className="input w-full"
                value={formData.backoff_seconds || 0}
                onChange={(e) => setFormData({ ...formData, backoff_seconds: parseInt(e.target.value) })}
              />
            </div>
            <div>
              <label className="label">Concurrency Limit</label>
              <input
                type="number"
                className="input w-full"
                value={formData.concurrency_limit || 0}
                onChange={(e) => setFormData({ ...formData, concurrency_limit: parseInt(e.target.value) })}
              />
            </div>
          </div>
          <div className="flex gap-2 mt-4">
            <button
              onClick={() => updateQueue.mutate(formData as QueueConfig)}
              disabled={updateQueue.isPending}
              className="btn-primary"
            >
              Save
            </button>
            <button onClick={() => setEditingQueue(null)} className="btn-secondary">
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

import clsx from 'clsx';
