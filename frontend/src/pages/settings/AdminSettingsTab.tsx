import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';

import { apiClient, getErrorMessage } from '@/lib/api';
import { AdminKnowledgeReviewTab } from '@/pages/settings/AdminKnowledgeReviewTab';
import { AdminOperationsTab } from '@/pages/settings/AdminOperationsTab';
import { BackupTab } from '@/pages/settings/BackupTab';
import { LicenseTab } from '@/pages/settings/LicenseTab';
import { useEnterpriseFeatures } from '@/hooks/useEnterpriseFeatures';
import { useAuthStore } from '@/stores/auth';

type AuthMode = 'password' | 'oidc' | 'both';

type AuthConfig = {
  auth_mode: AuthMode;
  oidc_issuer: string | null;
  oidc_client_id: string | null;
  oidc_audience: string | null;
  oidc_allowed_email_domains: string[] | null;
  oidc_default_org_slug: string | null;
  oidc_default_org_id: string | null;
  oidc_default_role: string | null;
  oidc_groups_claim: string | null;
  oidc_group_to_role_json: string | null;
};

type AuthConfigResponse = {
  effective: AuthConfig;
  overrides: Record<string, unknown>;
};

type EnvSetting = {
  key: string;
  value: string | null;
  is_sensitive: boolean;
  requires_restart: boolean;
};

type EnvSettingsResponse = { items: EnvSetting[] };

function normalizeText(value: string): string {
  return value.trim();
}

function parseDomains(input: string): string[] {
  const parts = input
    .split(/[,\n]/g)
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => p.replace(/^@/, '').toLowerCase());
  return Array.from(new Set(parts));
}

function domainsToText(domains: string[] | null): string {
  return (domains ?? []).join(', ');
}

async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    toast.success('Copied to clipboard');
  } catch {
    // Fallback for older browsers / blocked permissions
    const el = document.createElement('textarea');
    el.value = text;
    el.style.position = 'fixed';
    el.style.left = '-9999px';
    document.body.appendChild(el);
    el.focus();
    el.select();
    document.execCommand('copy');
    document.body.removeChild(el);
    toast.success('Copied to clipboard');
  }
}

function envLine(key: string, value: string | null | undefined): string {
  if (!value) return `# ${key}=`;
  // Quote values with spaces
  const needsQuotes = /\s/.test(value);
  return `${key}=${needsQuotes ? JSON.stringify(value) : value}`;
}

function buildBackendEnvSnippet(cfg: {
  auth_mode: AuthMode;
  oidc_issuer: string;
  oidc_client_id: string;
  oidc_audience: string;
  oidc_allowed_email_domains: string;
  oidc_default_org_slug: string;
  oidc_default_org_id: string;
  oidc_default_role: string;
  oidc_groups_claim: string;
  oidc_group_to_role_json: string;
}): string {
  const domains = parseDomains(cfg.oidc_allowed_email_domains);
  return [
    '# Backend (.env) snippet for Ninai2',
    '# Paste into backend/.env or docker-compose environment for backend',
    envLine('AUTH_MODE', cfg.auth_mode),
    '',
    '# OIDC / SSO (Option A)',
    envLine('OIDC_ISSUER', normalizeText(cfg.oidc_issuer) || null),
    envLine('OIDC_CLIENT_ID', normalizeText(cfg.oidc_client_id) || null),
    envLine('OIDC_AUDIENCE', normalizeText(cfg.oidc_audience) || null),
    envLine('OIDC_ALLOWED_EMAIL_DOMAINS', domains.length ? domains.join(',') : null),
    envLine('OIDC_DEFAULT_ORG_SLUG', normalizeText(cfg.oidc_default_org_slug) || null),
    envLine('OIDC_DEFAULT_ORG_ID', normalizeText(cfg.oidc_default_org_id) || null),
    envLine('OIDC_DEFAULT_ROLE', normalizeText(cfg.oidc_default_role) || null),
    envLine('OIDC_GROUPS_CLAIM', normalizeText(cfg.oidc_groups_claim) || null),
    envLine('OIDC_GROUP_TO_ROLE_JSON', normalizeText(cfg.oidc_group_to_role_json) || null),
    '',
  ].join('\n');
}

function buildFrontendEnvSnippet(cfg: {
  oidc_issuer: string;
  oidc_client_id: string;
}): string {
  // Frontend OIDC client needs authority + client id + redirect URI.
  // Redirect URI must match the route added in the app.
  return [
    '# Frontend (.env) snippet for Ninai2',
    '# Paste into frontend/.env (Vite) or docker-compose environment for frontend',
    '# After changing frontend env vars, you must rebuild/restart the frontend container.',
    envLine('VITE_OIDC_AUTHORITY', normalizeText(cfg.oidc_issuer) || null),
    envLine('VITE_OIDC_CLIENT_ID', normalizeText(cfg.oidc_client_id) || null),
    envLine('VITE_OIDC_REDIRECT_URI', 'http://localhost:3000/auth/oidc/callback'),
    '',
  ].join('\n');
}

type LogseqConfig = {
  effective: { export_base_dir: string; org_export_dir: string; last_nightly_export_at: string | null };
  overrides: Record<string, string>;
};

type FeedbackConfig = {
  updated_thresholds: Record<string, unknown>;
  stopwords: string[];
  heuristic_weights: Record<string, unknown>;
  calibration_delta: Record<string, unknown>;
  last_agent_version: string | null;
  updated_at: string | null;
};

export function AdminSettingsTab() {
  const [subtab, setSubtab] = useState<'auth' | 'env' | 'knowledge' | 'operations' | 'backups' | 'license' | 'logseq' | 'feedback'>('auth');
  const { hasAdminOperations } = useEnterpriseFeatures();
  const accessToken = useAuthStore((state) => state.accessToken);
  const refreshToken = useAuthStore((state) => state.refreshToken);
  const setTokens = useAuthStore((state) => state.setTokens);
  const [benchmarkToken, setBenchmarkToken] = useState('');

  useEffect(() => {
    setBenchmarkToken(accessToken ?? '');
  }, [accessToken]);

  const authQuery = useQuery<AuthConfigResponse>({
    queryKey: ['admin', 'settings', 'auth'],
    enabled: subtab === 'auth',
    queryFn: async () => {
      const res = await apiClient.get('/admin/settings/auth');
      return res.data;
    },
  });

  const envQuery = useQuery<EnvSettingsResponse>({
    queryKey: ['admin', 'settings', 'env'],
    enabled: subtab === 'env',
    queryFn: async () => {
      const res = await apiClient.get('/admin/settings/env');
      return res.data;
    },
  });

  const logseqQuery = useQuery<LogseqConfig>({
    queryKey: ['admin', 'logseq', 'config'],
    enabled: subtab === 'logseq',
    queryFn: async () => {
      const res = await apiClient.get('/logseq/export/config');
      return res.data;
    },
  });

  const [logseqDir, setLogseqDir] = useState('');
  useEffect(() => {
    const override = logseqQuery.data?.overrides?.export_base_dir ?? '';
    setLogseqDir(override);
  }, [logseqQuery.data]);

  const logseqSaveMutation = useMutation({
    mutationFn: async (dir: string | null) => {
      const res = await apiClient.put('/logseq/export/config', { export_base_dir: dir || null });
      return res.data as LogseqConfig;
    },
    onSuccess: (data) => {
      setLogseqDir(data.overrides?.export_base_dir ?? '');
      toast.success('Logseq export path updated');
    },
    onError: (error) => toast.error(getErrorMessage(error)),
  });

  const feedbackQuery = useQuery<FeedbackConfig>({
    queryKey: ['admin', 'feedback-learning'],
    enabled: subtab === 'feedback',
    queryFn: async () => {
      const res = await apiClient.get('/admin/feedback-learning');
      return res.data;
    },
  });

  const feedbackResetMutation = useMutation({
    mutationFn: async () => {
      const res = await apiClient.delete('/admin/feedback-learning');
      return res.data;
    },
    onSuccess: () => {
      toast.success('Feedback learning calibration reset to defaults');
      feedbackQuery.refetch();
    },
    onError: (error) => toast.error(getErrorMessage(error)),
  });

  const effective = authQuery.data?.effective;

  const initialForm = useMemo(() => {
    if (!effective) {
      return {
        auth_mode: '' as '' | AuthMode,
        oidc_issuer: '',
        oidc_client_id: '',
        oidc_audience: '',
        oidc_allowed_email_domains: '',
        oidc_default_org_slug: '',
        oidc_default_org_id: '',
        oidc_default_role: '',
        oidc_groups_claim: '',
        oidc_group_to_role_json: '',
      };
    }

    return {
      auth_mode: effective.auth_mode,
      oidc_issuer: effective.oidc_issuer ?? '',
      oidc_client_id: effective.oidc_client_id ?? '',
      oidc_audience: effective.oidc_audience ?? '',
      oidc_allowed_email_domains: domainsToText(effective.oidc_allowed_email_domains),
      oidc_default_org_slug: effective.oidc_default_org_slug ?? '',
      oidc_default_org_id: effective.oidc_default_org_id ?? '',
      oidc_default_role: effective.oidc_default_role ?? '',
      oidc_groups_claim: effective.oidc_groups_claim ?? '',
      oidc_group_to_role_json: effective.oidc_group_to_role_json ?? '',
    };
  }, [effective]);

  const [form, setForm] = useState(initialForm);

  useEffect(() => {
    setForm(initialForm);
  }, [initialForm]);

  const saveMutation = useMutation({
    mutationFn: async () => {
      const payload = {
        auth_mode: form.auth_mode === '' ? null : form.auth_mode,
        oidc_issuer: normalizeText(form.oidc_issuer) ? normalizeText(form.oidc_issuer) : null,
        oidc_client_id: normalizeText(form.oidc_client_id) ? normalizeText(form.oidc_client_id) : null,
        oidc_audience: normalizeText(form.oidc_audience) ? normalizeText(form.oidc_audience) : null,
        oidc_allowed_email_domains: normalizeText(form.oidc_allowed_email_domains)
          ? parseDomains(form.oidc_allowed_email_domains)
          : null,
        oidc_default_org_slug: normalizeText(form.oidc_default_org_slug) ? normalizeText(form.oidc_default_org_slug) : null,
        oidc_default_org_id: normalizeText(form.oidc_default_org_id) ? normalizeText(form.oidc_default_org_id) : null,
        oidc_default_role: normalizeText(form.oidc_default_role) ? normalizeText(form.oidc_default_role) : null,
        oidc_groups_claim: normalizeText(form.oidc_groups_claim) ? normalizeText(form.oidc_groups_claim) : null,
        oidc_group_to_role_json: normalizeText(form.oidc_group_to_role_json) ? form.oidc_group_to_role_json : null,
      };

      const res = await apiClient.put('/admin/settings/auth', payload);
      return res.data as AuthConfigResponse;
    },
    onSuccess: (data: AuthConfigResponse) => {
      toast.success('Authentication settings updated');
      setForm({
        auth_mode: data.effective.auth_mode,
        oidc_issuer: data.effective.oidc_issuer ?? '',
        oidc_client_id: data.effective.oidc_client_id ?? '',
        oidc_audience: data.effective.oidc_audience ?? '',
        oidc_allowed_email_domains: domainsToText(data.effective.oidc_allowed_email_domains),
        oidc_default_org_slug: data.effective.oidc_default_org_slug ?? '',
        oidc_default_org_id: data.effective.oidc_default_org_id ?? '',
        oidc_default_role: data.effective.oidc_default_role ?? '',
        oidc_groups_claim: data.effective.oidc_groups_claim ?? '',
        oidc_group_to_role_json: data.effective.oidc_group_to_role_json ?? '',
      });
    },
    onError: (err: unknown) => toast.error(getErrorMessage(err)),
  });

  const refreshBenchmarkTokenMutation = useMutation({
    mutationFn: async () => {
      if (!refreshToken) {
        throw new Error('No refresh token found. Please sign in again.');
      }

      const res = await apiClient.post('/auth/refresh', {
        refresh_token: refreshToken,
      });

      return res.data as { access_token: string; refresh_token: string };
    },
    onSuccess: (data) => {
      setTokens(data.access_token, data.refresh_token);
      setBenchmarkToken(data.access_token);
      toast.success('New benchmark token generated');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Admin Settings</h3>
        <p className="text-sm text-gray-500 mt-1">Manage runtime configuration. Some .env values are read-only.</p>
      </div>

      <div className="card bg-amber-50 border border-amber-200">
        <p className="text-sm text-amber-900 font-medium">Restart notes</p>
        <ul className="mt-2 text-sm text-amber-800 list-disc pl-5 space-y-1">
          <li>Changes under <span className="font-medium">Authentication</span> apply immediately to the backend (no restart required).</li>
          <li>Items under <span className="font-medium">Environment (.env)</span> are read-only here; changing them requires updating deployment env and restarting containers.</li>
          <li>SSO redirect uses frontend env vars; changing <span className="font-mono">VITE_OIDC_*</span> requires rebuilding/restarting the frontend.</li>
        </ul>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          className={subtab === 'auth' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setSubtab('auth')}
        >
          Authentication
        </button>
        <button
          type="button"
          className={subtab === 'env' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setSubtab('env')}
        >
          Environment (.env)
        </button>
        <button
          type="button"
          className={subtab === 'knowledge' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setSubtab('knowledge')}
        >
          Knowledge Review
        </button>
        {hasAdminOperations && (
          <button
            type="button"
            className={subtab === 'operations' ? 'btn-primary' : 'btn-secondary'}
            onClick={() => setSubtab('operations')}
          >
            Ops & Monitoring
          </button>
        )}
        <button
          type="button"
          className={subtab === 'backups' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setSubtab('backups')}
        >
          Backups
        </button>
        <button
          type="button"
          className={subtab === 'license' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setSubtab('license')}
        >
          License
        </button>
        <button
          type="button"
          className={subtab === 'logseq' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setSubtab('logseq')}
        >
          Logseq Export
        </button>
        <button
          type="button"
          className={subtab === 'feedback' ? 'btn-primary' : 'btn-secondary'}
          onClick={() => setSubtab('feedback')}
        >
          Feedback Learning
        </button>
      </div>

      {subtab === 'auth' && (
        <div className="space-y-4">
          <div className="card border border-blue-200 bg-blue-50">
            <h4 className="font-medium text-gray-900">Benchmark API Token</h4>
            <p className="text-sm text-gray-600 mt-1">
              Use this bearer token with <span className="font-mono">tests.benchmarks.run_all --api-token</span>.
            </p>

            <div className="mt-4 space-y-3">
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => setBenchmarkToken(accessToken ?? '')}
                  disabled={!accessToken}
                >
                  Use current token
                </button>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() => refreshBenchmarkTokenMutation.mutate()}
                  disabled={refreshBenchmarkTokenMutation.isPending || !refreshToken}
                >
                  {refreshBenchmarkTokenMutation.isPending ? 'Generating…' : 'Generate new token'}
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => copyToClipboard(benchmarkToken)}
                  disabled={!benchmarkToken}
                >
                  Copy token
                </button>
              </div>

              <textarea
                className="input w-full h-28 font-mono text-xs"
                value={benchmarkToken}
                readOnly
                placeholder="Token will appear here"
              />

              <p className="text-xs text-gray-500">
                Example: <span className="font-mono">python -m tests.benchmarks.run_all --mode unit --strategy heuristic --dataset kaggle --json --save-to-api http://localhost:8000/api/v1 --api-token &lt;paste-token&gt;</span>
              </p>
            </div>
          </div>

          {authQuery.isLoading && <div className="text-sm text-gray-500">Loading…</div>}
          {authQuery.isError && <div className="text-sm text-red-600">Failed to load settings</div>}

          {effective && (
            <div className="space-y-4">
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() =>
                    copyToClipboard(
                      buildBackendEnvSnippet({
                        ...form,
                        auth_mode: form.auth_mode === '' ? 'password' : form.auth_mode,
                      })
                    )
                  }
                >
                  Copy backend env snippet
                </button>
                <button
                  type="button"
                  className="btn-secondary"
                  onClick={() => copyToClipboard(buildFrontendEnvSnippet(form))}
                >
                  Copy frontend env snippet
                </button>
              </div>

              <div>
                <label className="label">Auth Mode</label>
                <select
                  className="input w-full max-w-2xl"
                  value={form.auth_mode}
                  onChange={(e) => setForm((s) => ({ ...s, auth_mode: e.target.value as AuthMode }))}
                >
                  <option value="password">password</option>
                  <option value="oidc">oidc</option>
                  <option value="both">both</option>
                </select>
                <p className="mt-1 text-xs text-gray-500">Controls whether password login, SSO, or both are allowed.</p>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div>
                  <label className="label">OIDC Issuer</label>
                  <input className="input w-full" value={form.oidc_issuer} onChange={(e) => setForm((s) => ({ ...s, oidc_issuer: e.target.value }))} />
                </div>
                <div>
                  <label className="label">OIDC Client ID</label>
                  <input className="input w-full" value={form.oidc_client_id} onChange={(e) => setForm((s) => ({ ...s, oidc_client_id: e.target.value }))} />
                </div>
                <div>
                  <label className="label">OIDC Audience (optional)</label>
                  <input className="input w-full" value={form.oidc_audience} onChange={(e) => setForm((s) => ({ ...s, oidc_audience: e.target.value }))} />
                </div>
                <div>
                  <label className="label">Allowed Email Domains</label>
                  <input
                    className="input w-full"
                    placeholder="example.com, example.org"
                    value={form.oidc_allowed_email_domains}
                    onChange={(e) => setForm((s) => ({ ...s, oidc_allowed_email_domains: e.target.value }))}
                  />
                  <p className="mt-1 text-xs text-gray-500">Comma or newline separated. Blank = allow all domains.</p>
                </div>
                <div>
                  <label className="label">Default Org Slug</label>
                  <input className="input w-full" value={form.oidc_default_org_slug} onChange={(e) => setForm((s) => ({ ...s, oidc_default_org_slug: e.target.value }))} />
                </div>
                <div>
                  <label className="label">Default Org ID</label>
                  <input className="input w-full" value={form.oidc_default_org_id} onChange={(e) => setForm((s) => ({ ...s, oidc_default_org_id: e.target.value }))} />
                </div>
                <div>
                  <label className="label">Default Role</label>
                  <input className="input w-full" placeholder="member" value={form.oidc_default_role} onChange={(e) => setForm((s) => ({ ...s, oidc_default_role: e.target.value }))} />
                </div>
                <div>
                  <label className="label">Groups Claim</label>
                  <input className="input w-full" placeholder="groups" value={form.oidc_groups_claim} onChange={(e) => setForm((s) => ({ ...s, oidc_groups_claim: e.target.value }))} />
                </div>
              </div>

              <div>
                <label className="label">Group → Role Mapping (JSON)</label>
                <textarea
                  className="input w-full font-mono min-h-[120px]"
                  placeholder='{"Ninai-Org-Admins": "org_admin"}'
                  value={form.oidc_group_to_role_json}
                  onChange={(e) => setForm((s) => ({ ...s, oidc_group_to_role_json: e.target.value }))}
                />
              </div>

              <div className="pt-2">
                <button
                  type="button"
                  className="btn-primary"
                  disabled={saveMutation.isPending}
                  onClick={() => saveMutation.mutate()}
                >
                  {saveMutation.isPending ? 'Saving…' : 'Save'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {subtab === 'operations' && <AdminOperationsTab />}

      {subtab === 'env' && (
        <div className="space-y-3">
          {envQuery.isLoading && <div className="text-sm text-gray-500">Loading…</div>}
          {envQuery.isError && <div className="text-sm text-red-600">Failed to load env settings</div>}

          {envQuery.data && (
            <div className="overflow-x-auto -mx-4 sm:mx-0">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b">
                    <th className="py-2 pr-4">Key</th>
                    <th className="py-2 pr-4">Value</th>
                    <th className="py-2">Notes</th>
                  </tr>
                </thead>
                <tbody>
                  {envQuery.data.items.map((item) => (
                    <tr key={item.key} className="border-b">
                      <td className="py-2 pr-4 font-mono">{item.key}</td>
                      <td className="py-2 pr-4 font-mono">{item.value ?? ''}</td>
                      <td className="py-2 text-gray-500">
                        {item.is_sensitive ? 'sensitive (masked)' : ''}{item.requires_restart ? (item.is_sensitive ? ', requires restart' : 'requires restart') : ''}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {subtab === 'backups' && <BackupTab />}

      {subtab === 'license' && <LicenseTab />}

      {subtab === 'logseq' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-base font-semibold text-gray-900">Logseq Export Configuration</h3>
            <p className="mt-1 text-sm text-gray-500">
              Controls where write-to-disk exports are stored on the server. Leave blank to use the default path from the environment (<code className="text-xs bg-gray-100 px-1 rounded">LOGSEQ_EXPORT_DIR</code>).
            </p>
          </div>

          {logseqQuery.isLoading && <div className="text-sm text-gray-500">Loading…</div>}
          {logseqQuery.isError && <div className="text-sm text-red-600">Failed to load Logseq config.</div>}

          {logseqQuery.data && (
            <div className="space-y-5">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 rounded-lg border border-gray-200 bg-gray-50 p-4 text-sm">
                <div>
                  <p className="text-xs text-gray-500 mb-0.5">Effective base directory</p>
                  <p className="font-mono text-gray-800 break-all">{logseqQuery.data.effective.export_base_dir}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-0.5">Org export directory</p>
                  <p className="font-mono text-gray-800 break-all">{logseqQuery.data.effective.org_export_dir}</p>
                </div>
                <div>
                  <p className="text-xs text-gray-500 mb-0.5">Last nightly export</p>
                  <p className="text-gray-700">
                    {logseqQuery.data.effective.last_nightly_export_at
                      ? new Date(logseqQuery.data.effective.last_nightly_export_at).toLocaleString()
                      : 'Never'}
                  </p>
                </div>
              </div>

              <div>
                <label className="label">Export Base Directory Override</label>
                <input
                  type="text"
                  className="input w-full max-w-xl"
                  placeholder="e.g. /data/logseq-exports  (blank = use env default)"
                  value={logseqDir}
                  onChange={(e) => setLogseqDir(e.target.value)}
                />
                <p className="mt-1 text-xs text-gray-500">
                  Org exports will land in <code className="bg-gray-100 px-1 rounded">&lt;base_dir&gt;/&lt;org_id&gt;/</code>.
                  Send blank to revert to the environment default.
                </p>
              </div>

              <div className="flex gap-3">
                <button
                  type="button"
                  className="btn-primary"
                  disabled={logseqSaveMutation.isPending}
                  onClick={() => logseqSaveMutation.mutate(logseqDir || null)}
                >
                  {logseqSaveMutation.isPending ? 'Saving…' : 'Save'}
                </button>
                {logseqQuery.data.overrides?.export_base_dir && (
                  <button
                    type="button"
                    className="btn-secondary"
                    disabled={logseqSaveMutation.isPending}
                    onClick={() => { setLogseqDir(''); logseqSaveMutation.mutate(null); }}
                  >
                    Reset to default
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}

      {subtab === 'feedback' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-base font-semibold text-gray-900">Feedback Learning Calibration</h3>
            <p className="mt-1 text-sm text-gray-500">
              Read-only view of the thresholds and weights the <span className="font-medium">FeedbackIntegrationAgent</span> has
              automatically calibrated from human review decisions. Resetting clears all learned values and lets the agent start fresh.
            </p>
          </div>

          {feedbackQuery.isLoading && <div className="text-sm text-gray-500">Loading…</div>}
          {feedbackQuery.isError && (
            <div className="text-sm text-gray-500 italic">
              No calibration data found for this organisation yet. The FeedbackIntegrationAgent writes here after its first successful run.
            </div>
          )}

          {feedbackQuery.data && (
            <div className="space-y-5">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">Agent version</p>
                  <p className="text-gray-700">{feedbackQuery.data.last_agent_version ?? '—'}</p>
                </div>
                <div>
                  <p className="text-xs font-medium text-gray-500 mb-1">Last calibrated</p>
                  <p className="text-gray-700">
                    {feedbackQuery.data.updated_at
                      ? new Date(feedbackQuery.data.updated_at).toLocaleString()
                      : '—'}
                  </p>
                </div>
              </div>

              {([
                { key: 'updated_thresholds', label: 'Updated Thresholds', desc: 'Calibrated decision thresholds (e.g. minimum confidence score to auto-approve)' },
                { key: 'heuristic_weights', label: 'Heuristic Weights', desc: 'Feature importance weights derived from accepted/rejected review outcomes' },
                { key: 'calibration_delta', label: 'Last Calibration Delta', desc: 'The change applied during the most recent calibration run' },
              ] as const).map(({ key, label, desc }) => (
                <div key={key}>
                  <p className="text-xs font-medium text-gray-700 mb-0.5">{label}</p>
                  <p className="text-xs text-gray-400 mb-1">{desc}</p>
                  <pre className="rounded-lg border border-gray-200 bg-gray-50 p-3 text-xs font-mono overflow-x-auto">
                    {JSON.stringify(feedbackQuery.data[key], null, 2) || '{}'}
                  </pre>
                </div>
              ))}

              {feedbackQuery.data.stopwords.length > 0 && (
                <div>
                  <p className="text-xs font-medium text-gray-700 mb-0.5">Learned Stopwords</p>
                  <p className="text-xs text-gray-400 mb-1">Terms the agent has learned to ignore when comparing memory content</p>
                  <div className="flex flex-wrap gap-1.5">
                    {feedbackQuery.data.stopwords.map((w) => (
                      <span key={w} className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600">{w}</span>
                    ))}
                  </div>
                </div>
              )}

              <div className="pt-2 border-t border-gray-100">
                <button
                  type="button"
                  className="btn-secondary text-red-600 border-red-200 hover:bg-red-50"
                  disabled={feedbackResetMutation.isPending}
                  onClick={() => {
                    if (window.confirm('Reset all feedback learning calibration for this organisation? The agent will start fresh on its next run.')) {
                      feedbackResetMutation.mutate();
                    }
                  }}
                >
                  {feedbackResetMutation.isPending ? 'Resetting…' : 'Reset calibration'}
                </button>
                <p className="mt-1.5 text-xs text-gray-400">This cannot be undone. The agent will re-learn from future review decisions.</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
