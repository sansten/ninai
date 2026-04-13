import { defineConfig } from 'vite';
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
  const allowedHosts = (
    process.env.VITE_ALLOWED_HOSTS?.split(',').map((host) => host.trim()).filter(Boolean) ||
    []
  );

  return {
    // Intentionally avoid React Fast Refresh plugin in this deployment path.
    // It injects a preamble guard that can prevent app mount behind some L7 proxies.
    plugins: [],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
      },
    },
    server: {
      port: devPort,
      // Needed for ingress/LB probes and custom domains in containerized deployments.
      allowedHosts: allowedHosts.length > 0 ? allowedHosts : true,
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
            'vendor-react': ['react', 'react-dom', 'react-router-dom'],
            'vendor-utils': ['axios', 'date-fns', 'zustand', 'clsx'],
          },
        },
      },
      // Minification and compression
      minify: 'esbuild',
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
