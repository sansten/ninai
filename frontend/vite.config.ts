import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig(() => {
  // When running via docker-compose, the backend hostname is "backend".
  // When running the frontend locally, the backend is typically exposed on localhost:8000.
  // Allow overriding explicitly via env var.
  const apiProxyTarget =
    process.env.VITE_API_PROXY_TARGET?.trim() ||
    process.env.VITE_BACKEND_URL?.trim() ||
    'http://127.0.0.1:8000';

  // Vite defaults to 5173. In docker-compose we expose 3000.
  // Keep this configurable so local dev on 5173 doesn't fight with docker on 3000.
  const devPort = Number(process.env.VITE_DEV_PORT || 5173);
  const hmrHost = process.env.VITE_HMR_HOST?.trim();
  const hmrPortRaw = process.env.VITE_HMR_PORT?.trim();
  const hmrPort = hmrPortRaw ? Number(hmrPortRaw) : undefined;

  return {
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: devPort,
      proxy: {
        '/api': {
          target: apiProxyTarget,
          changeOrigin: true,
        },
      },
      ...(hmrHost || hmrPort
        ? {
            hmr: {
              ...(hmrHost ? { host: hmrHost } : {}),
              ...(hmrPort ? { port: hmrPort } : {}),
            },
          }
        : {}),
    },
    build: {
      // Code splitting optimization
      rollupOptions: {
        output: {
          // Manual chunk splitting for better caching
          manualChunks: {
            // Vendor chunks
            'vendor-react': ['react', 'react-dom', 'react-router-dom'],
            'vendor-ui': ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu', '@radix-ui/react-navigation-menu'],
            'vendor-utils': ['axios', 'date-fns', 'zustand', 'clsx'],
            // Feature-based chunks
            'admin': [
              './src/pages/admin',
              './src/components/admin',
              './src/hooks/useAdmin',
              './src/services/admin'
            ],
            'memories': [
              './src/pages/memories',
              './src/components/memory',
              './src/hooks/useMemory',
            ],
            'auth': [
              './src/pages/auth',
              './src/services/auth',
              './src/hooks/useAuth',
            ],
          },
        },
      },
      // Minification and compression
      minify: 'terser',
      terserOptions: {
        compress: {
          drop_console: true,
          drop_debugger: true,
        },
      },
      // Output directory
      outDir: 'dist',
      // Source maps for production debugging
      sourcemap: false,
      // Chunk size warnings
      reportCompressedSize: true,
      chunkSizeWarningLimit: 500,
    },
  };
});
