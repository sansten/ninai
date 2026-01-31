import { UserManager, WebStorageStateStore, type UserManagerSettings } from 'oidc-client-ts';

function requireEnv(name: string): string {
  const value = (import.meta.env as Record<string, string | undefined>)[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

export function isOidcConfigured(): boolean {
  return Boolean(import.meta.env.VITE_OIDC_AUTHORITY && import.meta.env.VITE_OIDC_CLIENT_ID);
}

export function createOidcUserManager(): UserManager {
  const authority = requireEnv('VITE_OIDC_AUTHORITY');
  const clientId = requireEnv('VITE_OIDC_CLIENT_ID');

  const redirectUri =
    import.meta.env.VITE_OIDC_REDIRECT_URI || `${window.location.origin}/auth/oidc/callback`;

  const settings: UserManagerSettings = {
    authority,
    client_id: clientId,
    redirect_uri: redirectUri,
    response_type: 'code',
    scope: import.meta.env.VITE_OIDC_SCOPE || 'openid profile email',
    // Avoid hard dependency on silent renew unless configured
    automaticSilentRenew: false,
    userStore: new WebStorageStateStore({ store: window.localStorage }),
  };

  return new UserManager(settings);
}

export async function oidcSignInRedirect(): Promise<void> {
  const mgr = createOidcUserManager();
  await mgr.signinRedirect();
}

export async function oidcHandleCallback(): Promise<string> {
  const mgr = createOidcUserManager();
  const user = await mgr.signinRedirectCallback();
  if (!user?.id_token) {
    throw new Error('SSO completed but no id_token was returned');
  }
  return user.id_token;
}
