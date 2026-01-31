/**
 * Backup Management Tab
 * ======================
 * 
 * Database backup and restore operations for admins.
 */

import { useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import {
  CloudArrowUpIcon,
  CloudArrowDownIcon,
  ClockIcon,
  ChartBarIcon,
  InformationCircleIcon,
  CheckCircleIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';
import { apiClient, getErrorMessage } from '@/lib/api';
import { useIsAdmin } from '@/stores/auth';

interface BackupTask {
  id: string;
  backup_type: string;
  status: string;
  size_bytes: number;
  duration_seconds: number | null;
  checksum_sha256: string;
  s3_path: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

interface BackupSchedule {
  id: string;
  frequency: string;
  retention_days: number;
  backup_time: string;
  enabled: boolean;
  s3_bucket: string;
  max_backup_size_gb: number;
  enable_incremental: boolean;
  last_run_at: string | null;
  next_run_at: string | null;
  last_success_at: string | null;
  consecutive_failures: number;
  created_at: string;
}

interface BackupStatistics {
  total_backups: number;
  total_size_gb: number;
  failed_backups: number;
  last_backup_time: string | null;
  success_rate: number;
}

export function BackupTab() {
  const isAdmin = useIsAdmin();
  const queryClient = useQueryClient();
  const [restoreBackupId, setRestoreBackupId] = useState<string | null>(null);
  const [showRestoreConfirm, setShowRestoreConfirm] = useState(false);

  // Fetch backup statistics
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['backup-statistics'],
    queryFn: async () => {
      const response = await apiClient.get('/backups/statistics');
      return response.data as BackupStatistics;
    },
    enabled: isAdmin,
  });

  // Fetch backup history
  const { data: backups, isLoading: backupsLoading, refetch: refetchBackups } = useQuery({
    queryKey: ['backup-history'],
    queryFn: async () => {
      const response = await apiClient.get('/backups?page=1&page_size=10');
      return response.data.backups as BackupTask[];
    },
    enabled: isAdmin,
  });

  // Fetch backup schedule
  const { data: schedule, isLoading: scheduleLoading, refetch: refetchSchedule } = useQuery({
    queryKey: ['backup-schedule'],
    queryFn: async () => {
      const response = await apiClient.get('/backups/schedule');
      return response.data as BackupSchedule;
    },
    enabled: isAdmin,
  });

  // Create backup mutation
  const createBackupMutation = useMutation({
    mutationFn: async (backupType: string) => {
      const response = await apiClient.post('/backups/create', { backup_type: backupType });
      return response.data;
    },
    onSuccess: () => {
      toast.success('Backup started');
      refetchBackups();
      queryClient.invalidateQueries({ queryKey: ['backup-statistics'] });
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  // Restore backup mutation
  const restoreBackupMutation = useMutation({
    mutationFn: async (backupId: string) => {
      const response = await apiClient.post('/backups/restore', {
        backup_id: backupId,
        confirm: true,
      });
      return response.data;
    },
    onSuccess: () => {
      toast.success('Restore started');
      setShowRestoreConfirm(false);
      setRestoreBackupId(null);
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  // Update schedule mutation
  const updateScheduleMutation = useMutation({
    mutationFn: async (enabled: boolean) => {
      const response = await apiClient.patch('/backups/schedule', { enabled });
      return response.data;
    },
    onSuccess: () => {
      toast.success('Schedule updated');
      refetchSchedule();
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  if (!isAdmin) {
    return (
      <div className="card">
        <p className="text-sm text-gray-600">
          Backup management is only available to administrators.
        </p>
      </div>
    );
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
  };

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return 'N/A';
    const minutes = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return minutes > 0 ? `${minutes}m ${secs}s` : `${secs}s`;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg font-medium text-gray-900">Database Backup & Restore</h3>
        <p className="text-sm text-gray-500 mt-1">
          Manage database backups, schedule automated backups, and restore from previous snapshots.
        </p>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Total Backups</p>
              <p className="text-2xl font-semibold text-gray-900">
                {statsLoading ? '...' : stats?.total_backups || 0}
              </p>
            </div>
            <ChartBarIcon className="h-8 w-8 text-primary-600" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Total Size</p>
              <p className="text-2xl font-semibold text-gray-900">
                {statsLoading ? '...' : `${stats?.total_size_gb.toFixed(2) || 0} GB`}
              </p>
            </div>
            <CloudArrowUpIcon className="h-8 w-8 text-blue-600" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Success Rate</p>
              <p className="text-2xl font-semibold text-gray-900">
                {statsLoading ? '...' : `${(stats?.success_rate * 100).toFixed(1) || 0}%`}
              </p>
            </div>
            <CheckCircleIcon className="h-8 w-8 text-green-600" />
          </div>
        </div>

        <div className="card">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-500">Failed</p>
              <p className="text-2xl font-semibold text-gray-900">
                {statsLoading ? '...' : stats?.failed_backups || 0}
              </p>
            </div>
            <XCircleIcon className="h-8 w-8 text-red-600" />
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2">
          <h4 className="font-medium text-gray-900">Quick Actions</h4>
          <span title="Create on-demand backups or enable scheduled backups.">
            <InformationCircleIcon className="h-4 w-4 text-gray-400" />
          </span>
        </div>
        <div className="flex flex-wrap gap-3">
          <button
            type="button"
            className="btn-primary"
            onClick={() => createBackupMutation.mutate('full')}
            disabled={createBackupMutation.isPending}
          >
            {createBackupMutation.isPending ? 'Creating...' : 'Create Full Backup'}
          </button>
          <button
            type="button"
            className="btn-secondary"
            onClick={() => createBackupMutation.mutate('incremental')}
            disabled={createBackupMutation.isPending}
          >
            {createBackupMutation.isPending ? 'Creating...' : 'Create Incremental Backup'}
          </button>
        </div>
      </div>

      {/* Backup Schedule */}
      <div className="card space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h4 className="font-medium text-gray-900">Automated Backup Schedule</h4>
            <span title="Configure automatic daily/weekly/monthly backups.">
              <InformationCircleIcon className="h-4 w-4 text-gray-400" />
            </span>
          </div>
          {schedule && (
            <span className={schedule.enabled ? 'badge-success' : 'badge-gray'}>
              {scheduleLoading ? 'Loading' : schedule.enabled ? 'Enabled' : 'Disabled'}
            </span>
          )}
        </div>

        {scheduleLoading ? (
          <p className="text-sm text-gray-500">Loading schedule...</p>
        ) : schedule ? (
          <div className="space-y-3">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
              <div>
                <p className="text-gray-500">Frequency</p>
                <p className="font-medium text-gray-900 capitalize">{schedule.frequency}</p>
              </div>
              <div>
                <p className="text-gray-500">Backup Time (UTC)</p>
                <p className="font-medium text-gray-900">{schedule.backup_time}</p>
              </div>
              <div>
                <p className="text-gray-500">Retention</p>
                <p className="font-medium text-gray-900">{schedule.retention_days} days</p>
              </div>
              <div>
                <p className="text-gray-500">S3 Bucket</p>
                <p className="font-medium text-gray-900">{schedule.s3_bucket}</p>
              </div>
              <div>
                <p className="text-gray-500">Last Success</p>
                <p className="font-medium text-gray-900">
                  {schedule.last_success_at
                    ? new Date(schedule.last_success_at).toLocaleString()
                    : 'Never'}
                </p>
              </div>
              <div>
                <p className="text-gray-500">Next Run</p>
                <p className="font-medium text-gray-900">
                  {schedule.next_run_at
                    ? new Date(schedule.next_run_at).toLocaleString()
                    : 'Not scheduled'}
                </p>
              </div>
            </div>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => updateScheduleMutation.mutate(!schedule.enabled)}
              disabled={updateScheduleMutation.isPending}
            >
              {updateScheduleMutation.isPending
                ? 'Updating...'
                : schedule.enabled
                ? 'Disable Schedule'
                : 'Enable Schedule'}
            </button>
          </div>
        ) : (
          <p className="text-sm text-gray-500">No schedule configured</p>
        )}
      </div>

      {/* Backup History */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2">
          <h4 className="font-medium text-gray-900">Recent Backups</h4>
          <span title="View recent backup history and restore from any backup.">
            <InformationCircleIcon className="h-4 w-4 text-gray-400" />
          </span>
        </div>

        {backupsLoading ? (
          <p className="text-sm text-gray-500">Loading backups...</p>
        ) : backups && backups.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Type
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Size
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Duration
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Created
                  </th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {backups.map((backup) => (
                  <tr key={backup.id}>
                    <td className="px-4 py-3 text-sm text-gray-900 capitalize">
                      {backup.backup_type}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      <span
                        className={
                          backup.status === 'completed'
                            ? 'badge-success'
                            : backup.status === 'failed'
                            ? 'badge-danger'
                            : backup.status === 'running'
                            ? 'badge-warning'
                            : 'badge-gray'
                        }
                      >
                        {backup.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {formatBytes(backup.size_bytes)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {formatDuration(backup.duration_seconds)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-900">
                      {new Date(backup.created_at).toLocaleString()}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      {backup.status === 'completed' && (
                        <button
                          type="button"
                          className="text-primary-600 hover:text-primary-700 font-medium"
                          onClick={() => {
                            setRestoreBackupId(backup.id);
                            setShowRestoreConfirm(true);
                          }}
                        >
                          Restore
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-gray-500">No backups found</p>
        )}
      </div>

      {/* Restore Confirmation Modal */}
      {showRestoreConfirm && restoreBackupId && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Confirm Restore</h3>
            <p className="text-sm text-gray-600 mb-6">
              This will restore the database from the selected backup. All current data will be
              replaced. This action cannot be undone.
            </p>
            <div className="flex gap-3 justify-end">
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setShowRestoreConfirm(false);
                  setRestoreBackupId(null);
                }}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn-primary bg-red-600 hover:bg-red-700"
                onClick={() => restoreBackupMutation.mutate(restoreBackupId)}
                disabled={restoreBackupMutation.isPending}
              >
                {restoreBackupMutation.isPending ? 'Restoring...' : 'Confirm Restore'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
