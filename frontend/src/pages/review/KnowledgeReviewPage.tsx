import { Navigate } from 'react-router-dom';

import { useIsKnowledgeReviewer } from '@/stores/auth';
import { AdminKnowledgeReviewTab } from '@/pages/settings/AdminKnowledgeReviewTab';

export function KnowledgeReviewPage() {
  const canReview = useIsKnowledgeReviewer();

  if (!canReview) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Review Queue</h1>
        <p className="text-gray-500 mt-1">Approve/reject knowledge submissions without admin access.</p>
      </div>

      <div className="card">
        <AdminKnowledgeReviewTab apiBasePath="/review/knowledge" />
      </div>
    </div>
  );
}
