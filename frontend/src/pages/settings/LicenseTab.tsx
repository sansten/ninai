import { useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';

import { apiClient, getErrorMessage } from '@/lib/api';

interface LicenseTokenInfo {
  token_last_4: string;
  org_id: string | null;
  features: string[];
  expires_at: string;
  valid: boolean;
}

const FEATURE_LABELS: Record<string, string> = {
  'enterprise.admin_ops': 'Admin Operations',
  'enterprise.drift_detection': 'Drift Detection',
  'enterprise.autoevalbench': 'Auto-Eval Benchmarks',
  'enterprise.observability': 'Advanced Observability',
};

export function LicenseTab() {
  const [tokenInput, setTokenInput] = useState('');
  const [showInput, setShowInput] = useState(false);

  const licenseQuery = useQuery({
    queryKey: ['admin', 'license'],
    queryFn: async () => {
      try {
        const res = await apiClient.get<LicenseTokenInfo>('/admin/license');
        return res.data;
      } catch (err) {
        // 404 = no license configured
        return null;
      }
    },
  });

  const updateMutation = useMutation({
    mutationFn: async (token: string) => {
      const res = await apiClient.post<LicenseTokenInfo>('/admin/license', { token });
      return res.data;
    },
    onSuccess: (data) => {
      toast.success('License token updated successfully');
      setTokenInput('');
      setShowInput(false);
      licenseQuery.refetch();
    },
    onError: (err) => {
      toast.error(getErrorMessage(err));
    },
  });

  const handleSubmit = () => {
    const token = tokenInput.trim();
    if (!token) {
      toast.error('Token cannot be empty');
      return;
    }
    updateMutation.mutate(token);
  };

  const license = licenseQuery.data;

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Enterprise License</h3>
        <p className="text-sm text-gray-500 mt-1">
          Manage your Enterprise edition license token. The public key is loaded from file or embedded.
        </p>
      </div>

      <div className="card bg-blue-50 border border-blue-200">
        <p className="text-sm text-blue-900 font-medium">ℹ️ How it works</p>
        <ul className="mt-3 text-sm text-blue-800 space-y-2">
          <li>
            • <span className="font-medium">Public key:</span> Loaded from <code className="bg-blue-100 px-1 rounded">./config/license_public.pem</code> or standard paths (fallback: embedded)
          </li>
          <li>
            • <span className="font-medium">Token:</span> Set via <code className="bg-blue-100 px-1 rounded">NINAI_LICENSE_TOKEN</code> env var or update below
          </li>
          <li>
            • <span className="font-medium">Signature:</span> Ed25519-verified. Token format: <code className="bg-blue-100 px-1 rounded">ninai1.&lt;payload&gt;.&lt;sig&gt;</code>
          </li>
        </ul>
      </div>

      {licenseQuery.isLoading && <div className="text-sm text-gray-500">Loading…</div>}
      {licenseQuery.isError && <div className="text-sm text-red-600">Failed to load license info</div>}

      {license ? (
        <div className="card">
          <div className="space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div>
                <label className="label">Organization ID</label>
                <div className="input bg-gray-50 text-gray-700">{license.org_id || 'N/A'}</div>
              </div>
              <div>
                <label className="label">Token (last 4 chars)</label>
                <div className="input bg-gray-50 text-gray-700">ninai1.........{license.token_last_4}</div>
              </div>
            </div>

            <div>
              <label className="label">Enabled Features</label>
              <div className="space-y-2">
                {license.features.length > 0 ? (
                  license.features.map((feat) => (
                    <div key={feat} className="flex items-center gap-2">
                      <span className="inline-block w-2 h-2 bg-green-500 rounded-full"></span>
                      <span className="text-sm text-gray-700">{FEATURE_LABELS[feat] || feat}</span>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-gray-500">No features enabled</p>
                )}
              </div>
            </div>

            <div>
              <label className="label">Expires At</label>
              <div className="input bg-gray-50 text-gray-700">
                {new Date(license.expires_at).toLocaleString()}
              </div>
            </div>

            {!showInput && (
              <button
                type="button"
                className="btn-secondary"
                onClick={() => setShowInput(true)}
              >
                Update Token
              </button>
            )}
          </div>
        </div>
      ) : (
        <div className="card bg-yellow-50 border border-yellow-200">
          <p className="text-sm text-yellow-900 font-medium">⚠️ No license configured</p>
          <p className="mt-2 text-sm text-yellow-800">
            Enterprise features are disabled. Set <code className="bg-yellow-100 px-1 rounded">NINAI_LICENSE_TOKEN</code> environment variable or paste your token below.
          </p>
        </div>
      )}

      {showInput && (
        <div className="card border-2 border-blue-300">
          <div className="space-y-3">
            <div>
              <label className="label">Paste License Token</label>
              <textarea
                className="input w-full font-mono text-sm"
                rows={4}
                placeholder="ninai1.eyJ..."
                value={tokenInput}
                onChange={(e) => setTokenInput(e.target.value)}
              />
              <p className="mt-1 text-xs text-gray-500">
                Format: <code>ninai1.&lt;base64(payload)&gt;.&lt;base64(signature)&gt;</code>
              </p>
            </div>

            <div className="flex gap-2">
              <button
                type="button"
                className="btn-primary"
                disabled={updateMutation.isPending}
                onClick={handleSubmit}
              >
                {updateMutation.isPending ? 'Verifying…' : 'Verify & Save'}
              </button>
              <button
                type="button"
                className="btn-secondary"
                onClick={() => {
                  setTokenInput('');
                  setShowInput(false);
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="card bg-gray-50">
        <p className="text-sm font-medium text-gray-900">Need a license?</p>
        <p className="mt-2 text-sm text-gray-600">
          Contact Sansten AI sales to get an Enterprise license token.
        </p>
      </div>
    </div>
  );
}
