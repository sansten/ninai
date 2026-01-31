/**
 * Settings Page
 * =============
 * 
 * Organization and user settings.
 */

import { useState, type ElementType } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { useForm } from 'react-hook-form';
import toast from 'react-hot-toast';
import {
  BuildingOfficeIcon,
  UserIcon,
  KeyIcon,
  BellIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';
import clsx from 'clsx';
import { apiClient, getErrorMessage } from '@/lib/api';
import { useAuthStore, useCurrentOrg, useCurrentUser, useIsAdmin } from '@/stores/auth';
import { AdminSettingsTab } from '@/pages/settings/AdminSettingsTab';

/**
 * Settings Tab Component
 */
interface TabProps {
  tabs: { id: string; name: string; icon: ElementType }[];
  activeTab: string;
  onChange: (id: string) => void;
}

function SettingsTabs({ tabs, activeTab, onChange }: TabProps) {
  return (
    <nav className="space-y-1">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={clsx(
            'w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors',
            activeTab === tab.id
              ? 'bg-primary-50 text-primary-700'
              : 'text-gray-600 hover:bg-gray-50'
          )}
        >
          <tab.icon className="h-5 w-5" />
          {tab.name}
        </button>
      ))}
    </nav>
  );
}

/**
 * Organization Settings Tab
 */
function OrganizationSettings() {
  const org = useCurrentOrg();
  const queryClient = useQueryClient();

  const { register, handleSubmit, formState: { errors, isDirty } } = useForm({
    defaultValues: {
      name: org.name,
    },
  });

  const updateMutation = useMutation({
    mutationFn: async (data: { name: string }) => {
      const response = await apiClient.patch(`/organizations/${org.id}`, data);
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['organization'] });
      useAuthStore.getState().setCurrentOrg(data);
      toast.success('Organization updated');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Organization Settings</h3>
        <p className="text-sm text-gray-500 mt-1">
          Manage your organization's details and preferences.
        </p>
      </div>

      <form onSubmit={handleSubmit((data) => updateMutation.mutate(data))} className="space-y-6">
        <div>
          <label className="label">Organization Name</label>
          <input
            type="text"
            className="input w-full max-w-2xl"
            {...register('name', { required: 'Name is required' })}
          />
          {errors.name && (
            <p className="mt-1 text-sm text-red-600">{errors.name.message}</p>
          )}
        </div>

        <div>
          <label className="label">Organization ID</label>
          <input
            type="text"
            className="input w-full max-w-2xl bg-gray-50"
            value={org.id}
            disabled
          />
          <p className="mt-1 text-xs text-gray-500">
            This cannot be changed.
          </p>
        </div>

        <div>
          <label className="label">Slug</label>
          <input
            type="text"
            className="input w-full max-w-2xl bg-gray-50"
            value={org.slug}
            disabled
          />
        </div>

        <div>
          <label className="label">Tier</label>
          <span className="badge-primary">{org.tier}</span>
        </div>

        <div className="pt-4">
          <button
            type="submit"
            disabled={!isDirty || updateMutation.isPending}
            className="btn-primary"
          >
            {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </form>
    </div>
  );
}

/**
 * Profile Settings Tab
 */
function ProfileSettings() {
  const user = useCurrentUser();
  const queryClient = useQueryClient();

  const { register, handleSubmit, formState: { errors, isDirty } } = useForm({
    defaultValues: {
      display_name: user.display_name,
    },
  });

  const updateMutation = useMutation({
    mutationFn: async (data: { display_name: string }) => {
      const response = await apiClient.patch(`/auth/me`, data);
      return response.data;
    },
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['me'] });
      useAuthStore.getState().setUser(data);
      toast.success('Profile updated');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Profile Settings</h3>
        <p className="text-sm text-gray-500 mt-1">
          Update your personal information.
        </p>
      </div>

      <form onSubmit={handleSubmit((data) => updateMutation.mutate(data))} className="space-y-6">
        <div>
          <label className="label">Email</label>
          <input
            type="email"
            className="input w-full max-w-2xl bg-gray-50"
            value={user.email}
            disabled
          />
        </div>

        <div>
          <label className="label">Display Name</label>
          <input
            type="text"
            className="input w-full max-w-2xl"
            {...register('display_name', { required: 'Display name is required' })}
          />
          {errors.display_name && (
            <p className="mt-1 text-sm text-red-600">{errors.display_name.message}</p>
          )}
        </div>

        <div className="pt-4">
          <button
            type="submit"
            disabled={!isDirty || updateMutation.isPending}
            className="btn-primary"
          >
            {updateMutation.isPending ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </form>
    </div>
  );
}

/**
 * Security Settings Tab
 */
function SecuritySettings() {
  const [totpSetup, setTotpSetup] = useState<{
    secret: string;
    qr_code_url: string;
    backup_codes: string[];
  } | null>(null);
  const [totpToken, setTotpToken] = useState('');
  const [totpVerified, setTotpVerified] = useState(false);
  const [smsPhone, setSmsPhone] = useState('');
  const [smsSetupMessage, setSmsSetupMessage] = useState<string | null>(null);
  const [smsOtp, setSmsOtp] = useState('');

  const { data: mfaStatus, isLoading: mfaStatusLoading, refetch: refetchMfaStatus } = useQuery({
    queryKey: ['mfa-status'],
    queryFn: async () => {
      const response = await apiClient.get('/mfa/status');
      return response.data as {
        totp_enabled: boolean;
        sms_enabled: boolean;
        webauthn_enabled: boolean;
        mfa_required: boolean;
        grace_period_until: string | null;
      };
    },
  });

  const totpSetupMutation = useMutation({
    mutationFn: async () => {
      const response = await apiClient.post('/mfa/totp/setup', {});
      return response.data as { secret: string; qr_code_url: string; backup_codes: string[] };
    },
    onSuccess: (data) => {
      setTotpSetup(data);
      setTotpVerified(false);
      setTotpToken('');
      toast.success('TOTP setup ready. Scan the QR code to continue.');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const totpVerifyMutation = useMutation({
    mutationFn: async (token: string) => {
      const response = await apiClient.post('/mfa/totp/verify', { token });
      return response.data as { success: boolean; message: string };
    },
    onSuccess: (data) => {
      if (data.success) {
        setTotpVerified(true);
        refetchMfaStatus();
        toast.success(data.message || 'TOTP enabled');
      } else {
        toast.error(data.message || 'TOTP verification failed');
      }
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const smsSetupMutation = useMutation({
    mutationFn: async (phoneNumber: string) => {
      const response = await apiClient.post('/mfa/sms/setup', { phone_number: phoneNumber });
      return response.data as { success: boolean; phone_number: string; message: string };
    },
    onSuccess: (data) => {
      setSmsSetupMessage(data.message);
      toast.success(data.message || 'SMS setup started');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const smsSendMutation = useMutation({
    mutationFn: async () => {
      const response = await apiClient.post('/mfa/sms/send-otp', {});
      return response.data as { success: boolean; message: string };
    },
    onSuccess: (data) => {
      toast.success(data.message || 'OTP sent');
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  const smsVerifyMutation = useMutation({
    mutationFn: async (otp: string) => {
      const response = await apiClient.post('/mfa/sms/verify-otp', { otp });
      return response.data as { success: boolean; message: string };
    },
    onSuccess: (data) => {
      if (data.success) {
        toast.success(data.message || 'SMS OTP verified');
        setSmsOtp('');
        refetchMfaStatus();
      } else {
        toast.error(data.message || 'SMS OTP verification failed');
      }
    },
    onError: (error) => {
      toast.error(getErrorMessage(error));
    },
  });

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Security Settings</h3>
        <p className="text-sm text-gray-500 mt-1">
          Manage your security preferences and authentication.
        </p>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <span className={mfaStatus?.mfa_required ? 'badge-warning' : 'badge-gray'}>
            {mfaStatusLoading ? 'MFA Required: Loading' : mfaStatus?.mfa_required ? 'MFA Required' : 'MFA Optional'}
          </span>
          {mfaStatus?.grace_period_until && (
            <span className="badge-gray">
              Grace period until {new Date(mfaStatus.grace_period_until).toLocaleDateString()}
            </span>
          )}
        </div>
        {mfaStatus?.mfa_required && (
          <p className="mt-2 text-xs text-gray-500">
            Your organization requires MFA. Set up at least one method before the grace period ends.
          </p>
        )}
      </div>

      <div className="card">
        <h4 className="font-medium text-gray-900">Change Password</h4>
        <p className="text-sm text-gray-500 mt-1">
          Update your password to keep your account secure.
        </p>
        <button
          type="button"
          className="btn-secondary mt-4"
          onClick={() => toast('Change password is not implemented yet.')}
        >
          Change Password
        </button>
      </div>

      <div className="card space-y-4">
        <div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-gray-900">Authenticator App (TOTP)</h4>
              <span title="Use a time-based code from an authenticator app.">
                <InformationCircleIcon className="h-4 w-4 text-gray-400" />
              </span>
            </div>
            <span className={mfaStatus?.totp_enabled ? 'badge-success' : 'badge-gray'}>
              {mfaStatusLoading ? 'Loading' : mfaStatus?.totp_enabled ? 'Enabled' : 'Not enabled'}
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Use an authenticator app like Google Authenticator or 1Password.
          </p>
        </div>

        {!totpSetup && (
          <button
            type="button"
            className="btn-secondary"
            onClick={() => totpSetupMutation.mutate()}
            disabled={totpSetupMutation.isPending}
          >
            {totpSetupMutation.isPending ? 'Preparing...' : 'Start TOTP Setup'}
          </button>
        )}

        {totpSetup && (
          <div className="space-y-4">
            <div className="rounded-lg border border-gray-200 bg-gray-50 p-4">
              <p className="text-sm text-gray-700">
                Scan this QR code with your authenticator app.
              </p>
              <div className="mt-3 flex items-center gap-6">
                <img
                  src={totpSetup.qr_code_url}
                  alt="TOTP QR code"
                  className="h-32 w-32 rounded bg-white p-2"
                />
                <div className="text-sm text-gray-600">
                  <p className="font-medium text-gray-900">Manual setup key</p>
                  <p className="mt-1 font-mono text-xs break-all">{totpSetup.secret}</p>
                </div>
              </div>
            </div>

            <div>
              <label className="label">Enter the 6-digit code</label>
              <div className="flex items-center gap-3">
                <input
                  type="text"
                  inputMode="numeric"
                  maxLength={6}
                  className="input w-40"
                  value={totpToken}
                  onChange={(event) => setTotpToken(event.target.value)}
                />
                <button
                  type="button"
                  className="btn-primary"
                  disabled={totpToken.length !== 6 || totpVerifyMutation.isPending}
                  onClick={() => totpVerifyMutation.mutate(totpToken)}
                >
                  {totpVerifyMutation.isPending ? 'Verifying...' : 'Verify'}
                </button>
              </div>
              {totpVerified && (
                <p className="mt-2 text-sm text-green-600">TOTP is enabled.</p>
              )}
            </div>

            <div>
              <p className="text-sm font-medium text-gray-900">Backup codes</p>
              <p className="text-xs text-gray-500 mt-1">
                Store these codes somewhere safe. Each code can only be used once.
              </p>
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs font-mono text-gray-700">
                {totpSetup.backup_codes.map((code) => (
                  <span key={code} className="rounded bg-gray-100 px-2 py-1">
                    {code}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="card space-y-4">
        <div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-gray-900">SMS One-Time Passwords</h4>
              <span title="Receive a verification code by text message.">
                <InformationCircleIcon className="h-4 w-4 text-gray-400" />
              </span>
            </div>
            <span className={mfaStatus?.sms_enabled ? 'badge-success' : 'badge-gray'}>
              {mfaStatusLoading ? 'Loading' : mfaStatus?.sms_enabled ? 'Enabled' : 'Not enabled'}
            </span>
          </div>
          <p className="text-sm text-gray-500 mt-1">
            Receive a one-time code via SMS to verify your login.
          </p>
        </div>

        <div className="space-y-3">
          <div>
            <label className="label">Phone Number</label>
            <input
              type="tel"
              className="input w-full max-w-sm"
              placeholder="+1 555 000 1234"
              value={smsPhone}
              onChange={(event) => setSmsPhone(event.target.value)}
            />
          </div>
          <div className="flex flex-wrap gap-3">
            <button
              type="button"
              className="btn-secondary"
              disabled={!smsPhone || smsSetupMutation.isPending}
              onClick={() => smsSetupMutation.mutate(smsPhone)}
            >
              {smsSetupMutation.isPending ? 'Saving...' : 'Save Phone'}
            </button>
            <button
              type="button"
              className="btn-secondary"
              disabled={!smsSetupMessage || smsSendMutation.isPending}
              onClick={() => smsSendMutation.mutate()}
            >
              {smsSendMutation.isPending ? 'Sending...' : 'Send OTP'}
            </button>
          </div>
          {smsSetupMessage && (
            <p className="text-sm text-gray-600">{smsSetupMessage}</p>
          )}
          <div>
            <label className="label">Enter OTP</label>
            <div className="flex items-center gap-3">
              <input
                type="text"
                inputMode="numeric"
                maxLength={6}
                className="input w-32"
                value={smsOtp}
                onChange={(event) => setSmsOtp(event.target.value)}
              />
              <button
                type="button"
                className="btn-primary"
                disabled={smsOtp.length !== 6 || smsVerifyMutation.isPending}
                onClick={() => smsVerifyMutation.mutate(smsOtp)}
              >
                {smsVerifyMutation.isPending ? 'Verifying...' : 'Verify OTP'}
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h4 className="font-medium text-gray-900">Security Keys (WebAuthn)</h4>
            <span title="Use a hardware key or passkey for phishing-resistant login.">
              <InformationCircleIcon className="h-4 w-4 text-gray-400" />
            </span>
          </div>
          <span className={mfaStatus?.webauthn_enabled ? 'badge-success' : 'badge-gray'}>
            {mfaStatusLoading ? 'Loading' : mfaStatus?.webauthn_enabled ? 'Enabled' : 'Not enabled'}
          </span>
        </div>
        <p className="text-sm text-gray-500 mt-1">
          Register a hardware security key like YubiKey or Passkeys.
        </p>
        <button
          type="button"
          className="btn-secondary mt-4"
          onClick={() => toast('WebAuthn enrollment is coming soon.')}
        >
          Add Security Key
        </button>
      </div>

      <div className="card">
        <h4 className="font-medium text-gray-900">Active Sessions</h4>
        <p className="text-sm text-gray-500 mt-1">
          View and manage your active sessions across devices.
        </p>
        <button
          type="button"
          className="btn-secondary mt-4"
          onClick={() => toast('Session management is not implemented yet.')}
        >
          View Sessions
        </button>
      </div>
    </div>
  );
}

/**
 * Notifications Settings Tab
 */
function NotificationSettings() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Notification Preferences</h3>
        <p className="text-sm text-gray-500 mt-1">
          Choose how you want to be notified about activity.
        </p>
      </div>

      <div className="space-y-4">
        {[
          { id: 'security', label: 'Security Alerts', desc: 'Get notified about security events' },
          { id: 'access', label: 'Access Denials', desc: 'Notifications when access is denied' },
          { id: 'team', label: 'Team Activity', desc: 'Updates about team changes' },
          { id: 'system', label: 'System Updates', desc: 'Important system announcements' },
        ].map((item) => (
          <div key={item.id} className="flex items-center justify-between card">
            <div>
              <p className="font-medium text-gray-900">{item.label}</p>
              <p className="text-sm text-gray-500">{item.desc}</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input type="checkbox" className="sr-only peer" defaultChecked />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-primary-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-primary-600"></div>
            </label>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Settings Page Component
 */
export function SettingsPage() {
  const [activeTab, setActiveTab] = useState('organization');
  const isAdmin = useIsAdmin();
  const user = useCurrentUser();

  const tabs = [
    { id: 'organization', name: 'Organization', icon: BuildingOfficeIcon },
    { id: 'profile', name: 'Profile', icon: UserIcon },
    { id: 'security', name: 'Security', icon: KeyIcon },
    { id: 'notifications', name: 'Notifications', icon: BellIcon },
    ...(isAdmin ? [{ id: 'admin', name: 'Admin', icon: KeyIcon }] : []),
  ];

  return (
    <div className="w-full">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Settings</h1>
        <p className="text-gray-500 mt-1">
          Manage your account and organization settings
        </p>

        <div className="mt-3 text-sm text-gray-600">
          <span className="font-medium">Signed in as:</span> {user.email}
          <span className="mx-2 text-gray-300">|</span>
          <span className="font-medium">Roles:</span> {(user.roles?.length ? user.roles.join(', ') : 'none')}
        </div>

        {!isAdmin && (
          <div className="mt-3 text-sm text-amber-800 bg-amber-50 border border-amber-200 rounded-lg p-3">
            Admin settings are only visible to <span className="font-medium">org_admin</span> or <span className="font-medium">system_admin</span>.
            In the seeded demo DB, try <span className="font-medium">admin@ninai.dev</span> / <span className="font-medium">admin1234</span>.
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Sidebar */}
        <div className="lg:col-span-3 xl:col-span-2">
          <div className="lg:sticky lg:top-6">
            <SettingsTabs
              tabs={tabs}
              activeTab={activeTab}
              onChange={setActiveTab}
            />
          </div>
        </div>

        {/* Content */}
        <div className="lg:col-span-9 xl:col-span-10">
          {activeTab === 'admin' && isAdmin ? (
            <AdminSettingsTab />
          ) : (
            <div className="card">
              {activeTab === 'organization' && <OrganizationSettings />}
              {activeTab === 'profile' && <ProfileSettings />}
              {activeTab === 'security' && <SecuritySettings />}
              {activeTab === 'notifications' && <NotificationSettings />}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
