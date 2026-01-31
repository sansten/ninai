import type { ReactNode } from 'react';
import React from 'react';

type Props = {
  children: ReactNode;
};

type State = {
  hasError: boolean;
  error?: unknown;
};

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(error: unknown): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: unknown) {
    // Keep this minimal; Vite/React will still log stack traces in devtools.
    // This just prevents a hard blank screen.
    console.error('App crashed:', error);
  }

  render() {
    if (!this.state.hasError) return this.props.children;

    const message =
      this.state.error instanceof Error
        ? this.state.error.message
        : 'Unknown error';

    return (
      <div style={{ padding: 16, fontFamily: 'ui-sans-serif, system-ui' }}>
        <h1 style={{ fontSize: 18, fontWeight: 700 }}>Frontend crashed</h1>
        <p style={{ marginTop: 8 }}>
          Open DevTools Console for details.
        </p>
        <pre
          style={{
            marginTop: 12,
            padding: 12,
            background: '#111827',
            color: '#f9fafb',
            borderRadius: 8,
            overflowX: 'auto',
          }}
        >
          {message}
        </pre>
      </div>
    );
  }
}
