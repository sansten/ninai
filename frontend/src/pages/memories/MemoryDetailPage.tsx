/**
 * Memory Detail Page
 * ==================
 * 
 * View and edit a single memory.
 */

import { useMemo, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';
import {
  ArrowLeftIcon,
  PencilIcon,
  TrashIcon,
  ShareIcon,
  ShieldCheckIcon,
  PaperClipIcon,
  ArrowDownTrayIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { apiClient, getErrorMessage } from '@/lib/api';
import { useCurrentOrg } from '@/stores/auth';
import type { Memory, AccessExplanation } from '@/types/api';

type MemoryAttachment = {
  id: string;
  memory_id: string;
  file_name: string;
  content_type?: string | null;
  size_bytes: number;
  created_at: string;
};

/**
 * Memory Detail Page Component
 */
export function MemoryDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const org = useCurrentOrg();
  const queryClient = useQueryClient();
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);

  const formatBytes = useMemo(
    () =>
      (bytes: number) => {
        if (bytes < 1024) return `${bytes} B`;
        const kb = bytes / 1024;
        if (kb < 1024) return `${kb.toFixed(1)} KB`;
        const mb = kb / 1024;
        return `${mb.toFixed(1)} MB`;
      },
    []
  );

  // Fetch memory
  const { data: memory, isLoading, error } = useQuery<Memory>({
    queryKey: ['memory', id, org.id],
    queryFn: async () => {
      const response = await apiClient.get(`/memories/${id}`);
      return response.data;
    },
    enabled: !!id,
  });

  // Fetch access explanation
  const { data: accessInfo } = useQuery<AccessExplanation>({
    queryKey: ['memory-access', id, org.id],
    queryFn: async () => {
      const response = await apiClient.get(`/memories/${id}/explain`);
      return response.data;
    },
    enabled: !!id,
  });

  // Fetch attachments
  const { data: attachmentsData } = useQuery<{ items: MemoryAttachment[]; total: number }>({
    queryKey: ['memory-attachments', id, org.id],
    queryFn: async () => {
      const response = await apiClient.get(`/memories/${id}/attachments`);
      return response.data;
    },
    enabled: !!id,
  });

  const attachments = attachmentsData?.items ?? [];

  // Upload attachment
  const uploadAttachmentMutation = useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData();
      form.append('file', file);
      const response = await apiClient.post(`/memories/${id}/attachments`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory-attachments', id] });
      setUploadFile(null);
      toast.success('Attachment uploaded');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const deleteAttachmentMutation = useMutation({
    mutationFn: async (attachmentId: string) => {
      await apiClient.delete(`/memories/${id}/attachments/${attachmentId}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory-attachments', id] });
      toast.success('Attachment deleted');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const downloadAttachment = async (attachment: MemoryAttachment) => {
    try {
      const response = await apiClient.get(`/memories/${id}/attachments/${attachment.id}`, {
        responseType: 'blob',
      });
      const blobUrl = window.URL.createObjectURL(response.data);
      const a = document.createElement('a');
      a.href = blobUrl;
      a.download = attachment.file_name;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(blobUrl);
    } catch (error) {
      toast.error(getErrorMessage(error));
    }
  };

  // Update mutation
  const updateMutation = useMutation({
    mutationFn: async (content: string) => {
      const response = await apiClient.patch(`/memories/${id}`, { content });
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['memory', id] });
      setIsEditing(false);
      toast.success('Memory updated');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: async () => {
      await apiClient.delete(`/memories/${id}`);
    },
    onSuccess: () => {
      toast.success('Memory deleted');
      navigate('/memories');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const handleEdit = () => {
    setEditContent(memory?.content || '');
    setIsEditing(true);
  };

  const handleSave = () => {
    updateMutation.mutate(editContent);
  };

  const handleDelete = () => {
    if (confirm('Are you sure you want to delete this memory?')) {
      deleteMutation.mutate();
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  if (error || !memory) {
    return (
      <div className="text-center py-12">
        <h2 className="text-lg font-semibold text-gray-900">Memory not found</h2>
        <p className="text-gray-500 mt-1">
          The memory you're looking for doesn't exist or you don't have access.
        </p>
        <button
          onClick={() => navigate('/memories')}
          className="btn-secondary mt-4"
        >
          Back to Memories
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={() => navigate('/memories')}
          className="p-2 hover:bg-gray-100 rounded-lg"
        >
          <ArrowLeftIcon className="h-5 w-5" />
        </button>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-gray-900">
            {memory.title || 'Untitled Memory'}
          </h1>
          <p className="text-sm text-gray-500">
            Created {new Date(memory.created_at).toLocaleDateString()}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleEdit} className="btn-secondary">
            <PencilIcon className="h-4 w-4 mr-2" />
            Edit
          </button>
          <button className="btn-secondary">
            <ShareIcon className="h-4 w-4 mr-2" />
            Share
          </button>
          <button onClick={handleDelete} className="btn-danger">
            <TrashIcon className="h-4 w-4 mr-2" />
            Delete
          </button>
        </div>
      </div>

      {/* Metadata */}
      <div className="card">
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div>
            <p className="text-sm text-gray-500">Type</p>
            <p className="font-medium capitalize">{memory.memory_type}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Visibility</p>
            <p className="font-medium capitalize">{memory.visibility_level}</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Importance</p>
            <p className="font-medium">{memory.importance_score}/10</p>
          </div>
          <div>
            <p className="text-sm text-gray-500">Access Count</p>
            <p className="font-medium">{memory.access_count}</p>
          </div>
        </div>

        {memory.tags && memory.tags.length > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-100">
            <p className="text-sm text-gray-500 mb-2">Tags</p>
            <div className="flex flex-wrap gap-2">
              {memory.tags.map((tag) => (
                <span key={tag} className="badge-gray">
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Access Info */}
      {accessInfo && (
        <div className={`card ${accessInfo.has_access ? 'bg-green-50' : 'bg-red-50'}`}>
          <div className="flex items-start gap-3">
            <ShieldCheckIcon className={`h-5 w-5 ${accessInfo.has_access ? 'text-green-600' : 'text-red-600'}`} />
            <div>
              <p className="font-medium">
                {accessInfo.has_access ? 'You have access' : 'Access restricted'}
              </p>
              <p className="text-sm text-gray-600">{accessInfo.reason}</p>
              {accessInfo.permission_path && (
                <p className="text-xs text-gray-500 mt-1">
                  Via: {accessInfo.permission_path}
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Content */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Content</h2>
        
        {isEditing ? (
          <div className="space-y-4">
            <textarea
              value={editContent}
              onChange={(e) => setEditContent(e.target.value)}
              className="input min-h-[200px]"
            />
            <div className="flex justify-end gap-2">
              <button
                onClick={() => setIsEditing(false)}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleSave}
                disabled={updateMutation.isPending}
                className="btn-primary"
              >
                {updateMutation.isPending ? 'Saving...' : 'Save'}
              </button>
            </div>
          </div>
        ) : (
          <div className="prose prose-sm max-w-none">
            {memory.content_format === 'markdown' ? (
              <div dangerouslySetInnerHTML={{ __html: memory.content }} />
            ) : memory.content_format === 'code' ? (
              <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                <code>{memory.content}</code>
              </pre>
            ) : memory.content_format === 'json' ? (
              <pre className="bg-gray-50 p-4 rounded-lg overflow-x-auto">
                <code>{JSON.stringify(JSON.parse(memory.content), null, 2)}</code>
              </pre>
            ) : (
              <p className="whitespace-pre-wrap">{memory.content}</p>
            )}
          </div>
        )}
      </div>

      {/* Attachments */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Attachments</h2>
          <div className="flex items-center gap-2">
            <label className="btn-secondary cursor-pointer">
              <PaperClipIcon className="h-4 w-4 mr-2" />
              Choose file
              <input
                type="file"
                className="hidden"
                onChange={(e) => setUploadFile(e.target.files?.[0] ?? null)}
              />
            </label>
            <button
              className="btn-primary"
              disabled={!uploadFile || uploadAttachmentMutation.isPending}
              onClick={() => {
                if (!uploadFile) return;
                uploadAttachmentMutation.mutate(uploadFile);
              }}
            >
              {uploadAttachmentMutation.isPending ? 'Uploading...' : 'Upload'}
            </button>
          </div>
        </div>

        {uploadFile && (
          <p className="text-sm text-gray-600 mb-3">
            Selected: <span className="font-medium">{uploadFile.name}</span> ({formatBytes(uploadFile.size)})
          </p>
        )}

        {attachments.length === 0 ? (
          <p className="text-sm text-gray-500">No attachments yet.</p>
        ) : (
          <div className="divide-y divide-gray-100">
            {attachments.map((a) => (
              <div key={a.id} className="py-3 flex items-center justify-between gap-4">
                <div className="min-w-0">
                  <p className="font-medium text-gray-900 truncate">{a.file_name}</p>
                  <p className="text-xs text-gray-500">
                    {a.content_type || 'unknown type'} • {formatBytes(a.size_bytes)} •{' '}
                    {new Date(a.created_at).toLocaleString()}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    className="btn-secondary"
                    onClick={() => downloadAttachment(a)}
                    title="Download"
                  >
                    <ArrowDownTrayIcon className="h-4 w-4" />
                  </button>
                  <button
                    className="btn-danger"
                    onClick={() => {
                      if (confirm('Delete this attachment?')) {
                        deleteAttachmentMutation.mutate(a.id);
                      }
                    }}
                    title="Delete"
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Summary */}
      {memory.summary && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-2">Summary</h2>
          <p className="text-gray-600">{memory.summary}</p>
        </div>
      )}

      {/* Source Metadata */}
      {memory.source_metadata && Object.keys(memory.source_metadata).length > 0 && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-2">Source Metadata</h2>
          <pre className="bg-gray-50 p-4 rounded-lg text-sm overflow-x-auto">
            {JSON.stringify(memory.source_metadata, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
