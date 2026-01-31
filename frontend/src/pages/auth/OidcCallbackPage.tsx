import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';

import { apiClient, getErrorMessage } from '@/lib/api';
import { useAuthStore } from '@/stores/auth';
import { isOidcConfigured, oidcHandleCallback } from '@/auth/oidc';
import type { TokenResponse } from '@/types/api';

export function OidcCallbackPage() {
  const navigate = useNavigate();
  const { login } = useAuthStore();
  const [status, setStatus] = useState<'loading' | 'error'>('loading');

  useEffect(() => {
    let cancelled = false;

    async function run() {
      try {
        if (!isOidcConfigured()) {
          throw new Error('SSO is not configured in the frontend (missing VITE_OIDC_AUTHORITY / VITE_OIDC_CLIENT_ID)');
        }

        const idToken = await oidcHandleCallback();
        const response = await apiClient.post<TokenResponse>('/auth/oidc/exchange', {
          id_token: idToken,
        });

        const { access_token, refresh_token, user: backendUser, organization } = response.data;

        const user = {
          id: backendUser.id,
          email: backendUser.email,
          display_name: backendUser.full_name,
          avatar_url: backendUser.avatar_url,
          is_active: backendUser.is_active,
          created_at: backendUser.created_at,
          roles: Array.isArray(backendUser.roles) ? backendUser.roles : [],
        };

        login(user, access_token, refresh_token, organization);

        if (!cancelled) {
          toast.success(`Welcome back, ${user.display_name}!`);
          navigate('/dashboard');
        }
      } catch (err) {
        if (!cancelled) {
          setStatus('error');
          toast.error(getErrorMessage(err));
        }
      }
    }

    run();

    return () => {
      cancelled = true;
    };
  }, [login, navigate]);

  if (status === 'loading') {
    return (
      <div className="min-h-[50vh] flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <h1 className="text-lg font-semibold">SSO Login Failed</h1>
      <p className="text-sm text-gray-600">Please try again or use email/password.</p>
    </div>
  );
}
