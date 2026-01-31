import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import toast from 'react-hot-toast';

import { apiClient, getErrorMessage } from '@/lib/api';
import { AdminKnowledgeReviewTab } from '@/pages/settings/AdminKnowledgeReviewTab';
import { AdminOperationsTab } from '@/pages/settings/AdminOperationsTab';
import { BackupTab } from '@/pages/settings/BackupTab';
import { LicenseTab } from '@/pages/settings/LicenseTab';
import { useEnterpriseFeatures } from '@/hooks/useEnterpriseFeatures';

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

export function AdminSettingsTab() {
  const [subtab, setSubtab] = useState<'auth' | 'env' | 'knowledge' | 'operations' | 'backups' | 'license'>('auth');
  const { hasAdminOperations } = useEnterpriseFeatures();

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
      </div>

      {subtab === 'auth' && (
        <div className="space-y-4">
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

      {subtab === 'operations' && hasAdminOperationsge' && <AdminKnowledgeReviewTab apiBasePath="/admin/knowledge" />}

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
    </div>
  );
}
