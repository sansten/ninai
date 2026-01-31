/**
 * Login Page
 * ==========
 * 
 * User authentication form.
 */

import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import toast from 'react-hot-toast';
import { apiClient, getErrorMessage } from '@/lib/api';
import { useAuthStore } from '@/stores/auth';
import { oidcSignInRedirect } from '@/auth/oidc';
import type { LoginRequest, TokenResponse } from '@/types/api';

export function LoginPage() {
  const navigate = useNavigate();
  const { login, setAvailableOrgs } = useAuthStore();
  const [isLoading, setIsLoading] = useState(false);
  const [oidcEnabled, setOidcEnabled] = useState(false);
  const [passwordEnabled, setPasswordEnabled] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadAuthMethods() {
      try {
        const res = await apiClient.get('/auth/methods');
        if (!cancelled) {
          setOidcEnabled(Boolean(res.data?.oidc_enabled));
          setPasswordEnabled(Boolean(res.data?.password_enabled));
        }
      } catch {
        // If this fails, keep defaults (password enabled)
      }
    }

    loadAuthMethods();
    return () => {
      cancelled = true;
    };
  }, []);

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginRequest>({
    defaultValues: {
      email: '',
      password: '',
    },
  });

  const onSubmit = async (data: LoginRequest) => {
    setIsLoading(true);
    try {
      const response = await apiClient.post<TokenResponse>('/auth/login', data);
      const { access_token, refresh_token, user: backendUser, organization } = response.data;
      
      // Transform backend user to frontend User interface
      const user = {
        id: backendUser.id,
        email: backendUser.email,
        display_name: backendUser.full_name,  // Map full_name to display_name
        avatar_url: backendUser.avatar_url,
        is_active: backendUser.is_active,
        created_at: backendUser.created_at,
        roles: Array.isArray(backendUser.roles) ? backendUser.roles : [],
      };
      
      // Store auth state
      login(user, access_token, refresh_token, organization);
      
      // Fetch available organizations
      try {
        const orgsResponse = await apiClient.get('/organizations');
        setAvailableOrgs(orgsResponse.data);
      } catch {
        // Non-critical, continue anyway
        setAvailableOrgs([organization]);
      }
      
      toast.success(`Welcome back, ${user.display_name}!`);
      navigate('/dashboard');
    } catch (error) {
      toast.error(getErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  };

  const onSsoLogin = async () => {
    try {
      await oidcSignInRedirect();
    } catch (error) {
      toast.error(getErrorMessage(error));
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
      {oidcEnabled && (
        <button
          type="button"
          onClick={onSsoLogin}
          className="btn-primary w-full bg-gray-900 hover:bg-gray-800"
        >
          Sign in with SSO
        </button>
      )}

      {oidcEnabled && passwordEnabled && (
        <div className="relative">
          <div className="absolute inset-0 flex items-center" aria-hidden="true">
            <div className="w-full border-t border-gray-200" />
          </div>
          <div className="relative flex justify-center">
            <span className="bg-white px-2 text-xs text-gray-500">or</span>
          </div>
        </div>
      )}

      <div>
        <label htmlFor="email" className="label">
          Email address
        </label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          className="input"
          disabled={!passwordEnabled}
          {...register('email', {
            required: 'Email is required',
            pattern: {
              value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
              message: 'Invalid email address',
            },
          })}
        />
        {errors.email && (
          <p className="mt-1 text-sm text-red-600">{errors.email.message}</p>
        )}
      </div>

      <div>
        <label htmlFor="password" className="label">
          Password
        </label>
        <input
          id="password"
          type="password"
          autoComplete="current-password"
          className="input"
          disabled={!passwordEnabled}
          {...register('password', {
            required: 'Password is required',
            minLength: {
              value: 8,
              message: 'Password must be at least 8 characters',
            },
          })}
        />
        {errors.password && (
          <p className="mt-1 text-sm text-red-600">{errors.password.message}</p>
        )}
      </div>

      <div className="flex items-center justify-between">
        <div className="flex items-center">
          <input
            id="remember-me"
            name="remember-me"
            type="checkbox"
            className="h-4 w-4 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
          />
          <label htmlFor="remember-me" className="ml-2 block text-sm text-gray-700">
            Remember me
          </label>
        </div>

        <div className="text-sm">
          <a
            href="#"
            className="font-medium text-primary-600 hover:text-primary-500"
          >
            Forgot your password?
          </a>
        </div>
      </div>

      <button
        type="submit"
        disabled={isLoading || !passwordEnabled}
        className="btn-primary w-full"
      >
        {isLoading ? (
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
            Signing in...
          </div>
        ) : (
          'Sign in'
        )}
      </button>

      {/* Demo credentials hint */}
      <div className="mt-4 p-4 bg-gray-50 rounded-lg text-sm text-gray-600">
        <p className="font-medium">Demo Credentials</p>
        <p>Email: demo@ninai.dev</p>
        <p>Password: demo1234</p>
      </div>
    </form>
  );
}
