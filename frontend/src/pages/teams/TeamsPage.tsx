/**
 * Teams Page
 * ==========
 * 
 * Team listing and management.
 */

import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Dialog } from '@headlessui/react';
import { useForm } from 'react-hook-form';
import toast from 'react-hot-toast';
import {
  PlusIcon,
  UserGroupIcon,
  PencilIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import { apiClient, getErrorMessage } from '@/lib/api';
import { useCurrentOrg } from '@/stores/auth';
import type { Team, TeamCreate } from '@/types/api';

/**
 * Team Card Component
 */
interface TeamCardProps {
  team: Team;
  onEdit: (team: Team) => void;
  onDelete: (team: Team) => void;
}

function TeamCard({ team, onEdit, onDelete }: TeamCardProps) {
  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-primary-100 rounded-lg">
            <UserGroupIcon className="h-6 w-6 text-primary-600" />
          </div>
          <div>
            <h3 className="font-semibold text-gray-900">{team.name}</h3>
            <p className="text-sm text-gray-500">@{team.slug}</p>
          </div>
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => onEdit(team)}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <PencilIcon className="h-4 w-4 text-gray-400" />
          </button>
          <button
            onClick={() => onDelete(team)}
            className="p-1 hover:bg-gray-100 rounded"
          >
            <TrashIcon className="h-4 w-4 text-gray-400" />
          </button>
        </div>
      </div>

      {team.description && (
        <p className="mt-3 text-sm text-gray-600">{team.description}</p>
      )}

      <div className="mt-4 pt-4 border-t border-gray-100 flex items-center justify-between text-sm">
        <span className={team.is_active ? 'badge-success' : 'badge-gray'}>
          {team.is_active ? 'Active' : 'Inactive'}
        </span>
        <span className="text-gray-400">
          Created {new Date(team.created_at).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
}

/**
 * Create/Edit Team Modal
 */
interface TeamModalProps {
  isOpen: boolean;
  onClose: () => void;
  team?: Team | null;
}

function TeamModal({ isOpen, onClose, team }: TeamModalProps) {
  const queryClient = useQueryClient();
  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<TeamCreate>({
    defaultValues: team
      ? { name: team.name, slug: team.slug, description: team.description }
      : {},
  });

  const createMutation = useMutation({
    mutationFn: async (data: TeamCreate) => {
      const response = await apiClient.post('/teams', data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['teams'] });
      toast.success('Team created');
      reset();
      onClose();
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const updateMutation = useMutation({
    mutationFn: async (data: TeamCreate) => {
      const response = await apiClient.patch(`/teams/${team?.id}`, data);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['teams'] });
      toast.success('Team updated');
      reset();
      onClose();
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const onSubmit = (data: TeamCreate) => {
    if (team) {
      updateMutation.mutate(data);
    } else {
      createMutation.mutate(data);
    }
  };

  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-50">
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />
      <div className="fixed inset-0 flex items-center justify-center p-4">
        <Dialog.Panel className="w-full max-w-md bg-white rounded-xl shadow-lg p-6">
          <Dialog.Title className="text-lg font-semibold text-gray-900">
            {team ? 'Edit Team' : 'Create Team'}
          </Dialog.Title>

          <form onSubmit={handleSubmit(onSubmit)} className="mt-4 space-y-4">
            <div>
              <label className="label">Name</label>
              <input
                type="text"
                className="input"
                {...register('name', { required: 'Name is required' })}
              />
              {errors.name && (
                <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>
              )}
            </div>

            <div>
              <label className="label">Slug</label>
              <input
                type="text"
                className="input"
                {...register('slug', {
                  required: 'Slug is required',
                  pattern: {
                    value: /^[a-z0-9-]+$/,
                    message: 'Slug must be lowercase letters, numbers, and hyphens',
                  },
                })}
              />
              {errors.slug && (
                <p className="mt-1 text-sm text-red-600">{errors.slug.message}</p>
              )}
            </div>

            <div>
              <label className="label">Description</label>
              <textarea
                className="input"
                rows={3}
                {...register('description')}
              />
            </div>

            <div className="flex justify-end gap-2 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={createMutation.isPending || updateMutation.isPending}
                className="btn-primary"
              >
                {createMutation.isPending || updateMutation.isPending
                  ? 'Saving...'
                  : team
                  ? 'Update'
                  : 'Create'}
              </button>
            </div>
          </form>
        </Dialog.Panel>
      </div>
    </Dialog>
  );
}

/**
 * Teams Page Component
 */
export function TeamsPage() {
  const org = useCurrentOrg();
  const queryClient = useQueryClient();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [editingTeam, setEditingTeam] = useState<Team | null>(null);

  // Fetch teams
  const { data: teams, isLoading } = useQuery<Team[]>({
    queryKey: ['teams', org.id],
    queryFn: async () => {
      const response = await apiClient.get('/teams');
      return response.data;
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async (teamId: string) => {
      await apiClient.delete(`/teams/${teamId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['teams'] });
      toast.success('Team deleted');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const handleEdit = (team: Team) => {
    setEditingTeam(team);
    setIsModalOpen(true);
  };

  const handleDelete = (team: Team) => {
    if (confirm(`Are you sure you want to delete "${team.name}"?`)) {
      deleteMutation.mutate(team.id);
    }
  };

  const handleModalClose = () => {
    setIsModalOpen(false);
    setEditingTeam(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Teams</h1>
          <p className="text-gray-500 mt-1">
            Manage teams and their members
          </p>
        </div>
        <button
          onClick={() => setIsModalOpen(true)}
          className="btn-primary"
        >
          <PlusIcon className="h-5 w-5 mr-2" />
          New Team
        </button>
      </div>

      {/* Teams Grid */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      ) : teams?.length ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {teams.map((team) => (
            <TeamCard
              key={team.id}
              team={team}
              onEdit={handleEdit}
              onDelete={handleDelete}
            />
          ))}
        </div>
      ) : (
        <div className="text-center py-12">
          <UserGroupIcon className="mx-auto h-12 w-12 text-gray-400" />
          <h3 className="mt-2 text-sm font-semibold text-gray-900">No teams</h3>
          <p className="mt-1 text-sm text-gray-500">
            Get started by creating a new team.
          </p>
          <button
            onClick={() => setIsModalOpen(true)}
            className="btn-primary mt-4"
          >
            <PlusIcon className="h-5 w-5 mr-2" />
            New Team
          </button>
        </div>
      )}

      {/* Modal */}
      <TeamModal
        isOpen={isModalOpen}
        onClose={handleModalClose}
        team={editingTeam}
      />
    </div>
  );
}
