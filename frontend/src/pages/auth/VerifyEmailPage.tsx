/**
 * Verify Email Page
 * =================
 *
 * Shown immediately after signup. User arrives here in two ways:
 *   1. Navigated by SignupPage after a successful POST /signup (state.email set)
 *   2. Landed directly on /signup/verify?token=<token> from the email link
 *
 * Case 2: call GET /signup/verify?token=, log the user in, redirect to /dashboard.
 * Case 1: show "check your inbox" UI with a resend button.
 */

import { useEffect, useState } from 'react';
import { useNavigate, useLocation, useSearchParams, Link } from 'react-router-dom';
import toast from 'react-hot-toast';
import { apiClient, getErrorMessage } from '@/lib/api';
import { useAuthStore } from '@/stores/auth';
import type { VerifyResponse, AuthUser, Organization } from '@/types/api';

export function VerifyEmailPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();
  const { login, setAvailableOrgs } = useAuthStore();

  const email: string | undefined = (location.state as { email?: string } | null)?.email;
  const token = searchParams.get('token');

  const [verifying, setVerifying] = useState(!!token);
  const [resending, setResending] = useState(false);
  const [resendCooldown, setResendCooldown] = useState(0);

  useEffect(() => {
    if (!token) return;

    let cancelled = false;

    async function verify() {
      try {
        const res = await apiClient.get<VerifyResponse>(`/signup/verify?token=${encodeURIComponent(token!)}`);
        const { access_token, refresh_token } = res.data;

        const meRes = await apiClient.get<AuthUser>('/auth/me', {
          headers: { Authorization: `Bearer ${access_token}` },
        });
        const me = meRes.data;

        const orgsRes = await apiClient.get<Organization[]>('/organizations', {
          headers: { Authorization: `Bearer ${access_token}` },
        });
        const organizations = orgsRes.data;

        const org =
          organizations.find((o) => o.id === res.data.org_id) ||
          organizations[0];

        if (!org) {
          throw new Error('Unable to determine organization for the verified account.');
        }

        if (cancelled) return;

        login(
          {
            id: me.id,
            email: me.email,
            display_name: me.full_name,
            avatar_url: me.avatar_url,
            is_active: me.is_active,
            created_at: me.created_at,
            roles: Array.isArray(me.roles) ? me.roles : [],
          },
          access_token,
          refresh_token,
          org,
        );

        setAvailableOrgs(organizations);

        toast.success(`Welcome to Ninai, ${me.full_name.split(' ')[0]}!`);
        navigate('/dashboard', { replace: true });
      } catch (error) {
        if (!cancelled) {
          setVerifying(false);
          toast.error(getErrorMessage(error));
        }
      }
    }

    verify();
    return () => {
      cancelled = true;
    };
  }, [token, login, navigate, setAvailableOrgs]);

  useEffect(() => {
    if (resendCooldown <= 0) return;
    const t = setTimeout(() => setResendCooldown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [resendCooldown]);

  const handleResend = async () => {
    if (!email || resending || resendCooldown > 0) return;
    setResending(true);
    try {
      await apiClient.post('/signup/resend', null, {
        params: { email },
      });
      toast.success('Verification email resent. Check your inbox.');
      setResendCooldown(60);
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setResending(false);
    }
  };

  if (verifying) {
    return (
      <div className="space-y-4 text-center">
        <div className="flex justify-center">
          <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary-600" />
        </div>
        <p className="text-gray-600">Verifying your email address...</p>
      </div>
    );
  }

  return (
    <div className="space-y-6 text-center">
      <div className="flex justify-center">
        <svg
          className="w-16 h-16 text-primary-500"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M21.75 6.75v10.5a2.25 2.25 0 01-2.25 2.25h-15a2.25 2.25 0 01-2.25-2.25V6.75m19.5 0A2.25 2.25 0 0019.5 4.5h-15a2.25 2.25 0 00-2.25 2.25m19.5 0v.243a2.25 2.25 0 01-1.07 1.916l-7.5 4.615a2.25 2.25 0 01-2.36 0L3.32 8.91a2.25 2.25 0 01-1.07-1.916V6.75"
          />
        </svg>
      </div>

      <div>
        <h2 className="text-xl font-semibold text-gray-900">Check your inbox</h2>
        <p className="mt-2 text-sm text-gray-500">
          We sent a verification link to{' '}
          {email ? (
            <span className="font-medium text-gray-700">{email}</span>
          ) : (
            'your email address'
          )}
          . Click the link to activate your workspace.
        </p>
      </div>

      <div className="text-sm text-gray-500 space-y-2">
        <p>Did not receive it? Check your spam folder.</p>

        {email && (
          <button
            type="button"
            onClick={handleResend}
            disabled={resending || resendCooldown > 0}
            className="font-medium text-primary-600 hover:text-primary-500 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {resending
              ? 'Sending...'
              : resendCooldown > 0
              ? `Resend in ${resendCooldown}s`
              : 'Resend verification email'}
          </button>
        )}
      </div>

      <p className="text-sm text-gray-400">
        <Link to="/login" className="hover:text-gray-600 underline">
          Back to sign in
        </Link>
      </p>
    </div>
  );
}
