import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import toast from 'react-hot-toast';

import { apiClient, getErrorMessage } from '@/lib/api';

type ReviewRequest = {
  id: string;
  item_id: string;
  item_version_id: string;
  status: string;
  requested_by_user_id: string | null;
  reviewed_by_user_id: string | null;
  reviewed_at: string | null;
  decision_comment: string | null;
  created_at: string;
};

type ReviewListResponse = { items: ReviewRequest[] };

type Draft = {
  promote_to_memory: boolean;
  tagsText: string;
  topicsText: string;
  primaryTopic: string;
};

function parseCsv(input: string): string[] {
  return Array.from(
    new Set(
      input
        .split(/[\n,]/g)
        .map((s) => s.trim())
        .filter(Boolean)
    )
  );
}

export function AdminKnowledgeReviewTab({ apiBasePath = '/admin/knowledge' }: { apiBasePath?: string }) {
  const qc = useQueryClient();

  const [drafts, setDrafts] = useState<Record<string, Draft>>({});

  const pendingQuery = useQuery<ReviewListResponse>({
    queryKey: ['admin', 'knowledge', 'review-requests', 'pending'],
    queryFn: async () => {
      const res = await apiClient.get(`${apiBasePath}/review-requests`, {
        params: { status: 'pending', limit: 100 },
      });
      return res.data;
    },
  });

  const approveMutation = useMutation({
    mutationFn: async (req: ReviewRequest) => {
      const draft = drafts[req.id];
      const comment = window.prompt('Approval comment (optional)') ?? undefined;

      const tags = parseCsv(draft?.tagsText ?? '');
      const topics = parseCsv(draft?.topicsText ?? '');
      const primary_topic = (draft?.primaryTopic ?? '').trim() || undefined;

      const payload = {
        comment,
        promote_to_memory: Boolean(draft?.promote_to_memory),
        tags,
        topics,
        primary_topic,
        // defaults are fine for now; can be exposed later
        memory_scope: 'organization',
        memory_type: 'procedural',
        classification: 'internal',
        topic_confidence: 0.8,
      };

      const res = await apiClient.post(`${apiBasePath}/review-requests/${req.id}/approve`, payload);
      return res.data as ReviewRequest;
    },
    onSuccess: () => {
      toast.success('Approved (and promoted if selected)');
      qc.invalidateQueries({ queryKey: ['admin', 'knowledge', 'review-requests', 'pending'] });
    },
    onError: (err: unknown) => toast.error(getErrorMessage(err)),
  });

  const rejectMutation = useMutation({
    mutationFn: async (req: ReviewRequest) => {
      const comment = window.prompt('Rejection reason (optional)') ?? undefined;
      const res = await apiClient.post(`${apiBasePath}/review-requests/${req.id}/reject`, { comment });
      return res.data as ReviewRequest;
    },
    onSuccess: () => {
      toast.success('Rejected');
      qc.invalidateQueries({ queryKey: ['admin', 'knowledge', 'review-requests', 'pending'] });
    },
    onError: (err: unknown) => toast.error(getErrorMessage(err)),
  });

  const items = useMemo(() => pendingQuery.data?.items ?? [], [pendingQuery.data?.items]);

  useEffect(() => {
    if (items.length === 0) return;
    setDrafts((prev) => {
      const next: Record<string, Draft> = { ...prev };
      let changed = false;
      for (const r of items) {
        if (!next[r.id]) {
          next[r.id] = { promote_to_memory: true, tagsText: '', topicsText: '', primaryTopic: '' };
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [items]);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Knowledge Review Queue</h3>
        <p className="text-sm text-gray-500 mt-1">Approve/reject submissions and publish vetted procedures.</p>
      </div>

      {pendingQuery.isLoading && <div className="text-sm text-gray-500">Loading…</div>}
      {pendingQuery.isError && <div className="text-sm text-red-600">Failed to load review requests</div>}

      {!pendingQuery.isLoading && !pendingQuery.isError && items.length === 0 && (
        <div className="text-sm text-gray-600">No pending review requests.</div>
      )}

      {items.length > 0 && (
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b">
                <th className="py-2 pr-4">Created</th>
                <th className="py-2 pr-4">Request</th>
                <th className="py-2 pr-4">Item</th>
                <th className="py-2 pr-4">Version</th>
                <th className="py-2 pr-4">Mapping</th>
                <th className="py-2 pr-4">Actions</th>
              </tr>
            </thead>
            <tbody>
              {items.map((r) => (
                <tr key={r.id} className="border-b">
                  <td className="py-2 pr-4 text-gray-700">{new Date(r.created_at).toLocaleString()}</td>
                  <td className="py-2 pr-4 font-mono text-xs text-gray-700">{r.id}</td>
                  <td className="py-2 pr-4 font-mono text-xs text-gray-700">{r.item_id}</td>
                  <td className="py-2 pr-4 font-mono text-xs text-gray-700">{r.item_version_id}</td>
                  <td className="py-2 pr-4">
                    <div className="space-y-2">
                      <label className="flex items-center gap-2 text-xs text-gray-700">
                        <input
                          type="checkbox"
                          checked={drafts[r.id]?.promote_to_memory ?? true}
                          onChange={(e) =>
                            setDrafts((prev) => ({
                              ...prev,
                              [r.id]: { ...(prev[r.id] ?? { promote_to_memory: true, tagsText: '', topicsText: '', primaryTopic: '' }), promote_to_memory: e.target.checked },
                            }))
                          }
                        />
                        Promote to long-term memory
                      </label>

                      <input
                        className="input"
                        placeholder="Tags (comma separated)"
                        value={drafts[r.id]?.tagsText ?? ''}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [r.id]: { ...(prev[r.id] ?? { promote_to_memory: true, tagsText: '', topicsText: '', primaryTopic: '' }), tagsText: e.target.value },
                          }))
                        }
                      />

                      <input
                        className="input"
                        placeholder="Topics (comma separated)"
                        value={drafts[r.id]?.topicsText ?? ''}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [r.id]: { ...(prev[r.id] ?? { promote_to_memory: true, tagsText: '', topicsText: '', primaryTopic: '' }), topicsText: e.target.value },
                          }))
                        }
                      />

                      <input
                        className="input"
                        placeholder="Primary topic (optional)"
                        value={drafts[r.id]?.primaryTopic ?? ''}
                        onChange={(e) =>
                          setDrafts((prev) => ({
                            ...prev,
                            [r.id]: { ...(prev[r.id] ?? { promote_to_memory: true, tagsText: '', topicsText: '', primaryTopic: '' }), primaryTopic: e.target.value },
                          }))
                        }
                      />
                    </div>
                  </td>
                  <td className="py-2 pr-4">
                    <div className="flex gap-2">
                      <button
                        type="button"
                        className="btn-primary"
                        disabled={approveMutation.isPending || rejectMutation.isPending}
                        onClick={() => approveMutation.mutate(r)}
                      >
                        Approve
                      </button>
                      <button
                        type="button"
                        className="btn-secondary"
                        disabled={approveMutation.isPending || rejectMutation.isPending}
                        onClick={() => rejectMutation.mutate(r)}
                      >
                        Reject
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
