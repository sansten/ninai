/**
 * Signup Page
 * ===========
 *
 * Self-service trial signup. Accessible at /signup and /signup?ref=sansten.
 * Collects org name, user name, email, and password then fires POST /signup.
 * On success, navigates to /signup/verify which shows the "check your inbox" screen.
 */

import { useState } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useForm } from 'react-hook-form';
import axios from 'axios';
import toast from 'react-hot-toast';
import { apiClient, getErrorMessage } from '@/lib/api';
import type { SignupRequest, SignupResponse } from '@/types/api';

interface SignupFormValues {
  org_name: string;
  full_name: string;
  email: string;
  password: string;
  confirm_password: string;
}

export function SignupPage() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const ref = searchParams.get('ref') ?? undefined;
  const [isLoading, setIsLoading] = useState(false);

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<SignupFormValues>({
    defaultValues: {
      org_name: '',
      full_name: '',
      email: '',
      password: '',
      confirm_password: '',
    },
  });

  const password = watch('password');

  const onSubmit = async (data: SignupFormValues) => {
    setIsLoading(true);
    try {
      const payload: SignupRequest = {
        org_name: data.org_name.trim(),
        full_name: data.full_name.trim(),
        email: data.email.trim().toLowerCase(),
        password: data.password,
        ref,
      };

      const response = await apiClient.post<SignupResponse>('/signup', payload);

      navigate('/signup/verify', {
        state: { email: payload.email, org_id: response.data.org_id },
        replace: true,
      });
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const data = error.response?.data as { detail?: unknown } | undefined;
        const detail = data?.detail;

        const detailText =
          typeof detail === 'string'
            ? detail
            : Array.isArray(detail)
            ? detail
                .map((item) => {
                  if (typeof item === 'string') return item;
                  if (item && typeof item === 'object' && 'msg' in item) {
                    return String((item as { msg?: unknown }).msg ?? '');
                  }
                  return '';
                })
                .join(' ')
            : '';

        if (/already exists/i.test(detailText)) {
          toast.error('User already exists. Redirecting to sign in...');
          navigate('/login', { replace: true });
          return;
        }
      }

      toast.error(getErrorMessage(error));
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900">Start your free trial</h2>
        <p className="mt-1 text-sm text-gray-500">
          No credit card required. 14 days free, then choose a plan.
        </p>
      </div>

      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        <div>
          <label htmlFor="org_name" className="label">
            Organisation name
          </label>
          <input
            id="org_name"
            type="text"
            autoComplete="organization"
            className="input"
            placeholder="Acme Corp"
            {...register('org_name', {
              required: 'Organisation name is required',
              maxLength: { value: 100, message: 'Max 100 characters' },
            })}
          />
          {errors.org_name && (
            <p className="mt-1 text-sm text-red-600">{errors.org_name.message}</p>
          )}
        </div>

        <div>
          <label htmlFor="full_name" className="label">
            Your full name
          </label>
          <input
            id="full_name"
            type="text"
            autoComplete="name"
            className="input"
            placeholder="Alice Smith"
            {...register('full_name', {
              required: 'Your name is required',
              maxLength: { value: 100, message: 'Max 100 characters' },
            })}
          />
          {errors.full_name && (
            <p className="mt-1 text-sm text-red-600">{errors.full_name.message}</p>
          )}
        </div>

        <div>
          <label htmlFor="email" className="label">
            Work email
          </label>
          <input
            id="email"
            type="email"
            autoComplete="email"
            className="input"
            placeholder="alice@acme.com"
            {...register('email', {
              required: 'Email is required',
              pattern: {
                value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                message: 'Enter a valid email address',
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
            autoComplete="new-password"
            className="input"
            placeholder="Min 10 chars, 1 uppercase, 1 number"
            {...register('password', {
              required: 'Password is required',
              minLength: { value: 10, message: 'At least 10 characters' },
              validate: {
                hasUpper: (v) =>
                  /[A-Z]/.test(v) || 'Must contain at least one uppercase letter',
                hasDigit: (v) =>
                  /[0-9]/.test(v) || 'Must contain at least one number',
              },
            })}
          />
          {errors.password && (
            <p className="mt-1 text-sm text-red-600">{errors.password.message}</p>
          )}
        </div>

        <div>
          <label htmlFor="confirm_password" className="label">
            Confirm password
          </label>
          <input
            id="confirm_password"
            type="password"
            autoComplete="new-password"
            className="input"
            {...register('confirm_password', {
              required: 'Please confirm your password',
              validate: (v) => v === password || 'Passwords do not match',
            })}
          />
          {errors.confirm_password && (
            <p className="mt-1 text-sm text-red-600">{errors.confirm_password.message}</p>
          )}
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="btn-primary w-full mt-2"
        >
          {isLoading ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2" />
              Creating your workspace...
            </div>
          ) : (
            'Create free workspace'
          )}
        </button>
      </form>

      <p className="text-center text-sm text-gray-500">
        Already have an account?{' '}
        <Link to="/login" className="font-medium text-primary-600 hover:text-primary-500">
          Sign in
        </Link>
      </p>

      <p className="text-center text-xs text-gray-400">
        By signing up you agree to our{' '}
        <a
          href="https://ninai.app/terms"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-gray-600"
        >
          Terms of Service
        </a>{' '}
        and{' '}
        <a
          href="https://ninai.app/privacy"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:text-gray-600"
        >
          Privacy Policy
        </a>
        .
      </p>
    </div>
  );
}
