# ── Stage 1: build ────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package*.json ./
RUN npm ci --ignore-scripts

COPY . .

# VITE_API_URL is baked into the bundle; the ingress routes /api → backend.
ARG VITE_API_URL=/api/v1
ENV VITE_API_URL=$VITE_API_URL

# Use vite build directly — skips tsc type-check pass (fast, correct bundle)
RUN npx vite build

# ── Stage 2: serve with nginx ──────────────────────────────────────────────────
FROM nginx:1.27-alpine

# Remove default config; replace with our own
RUN rm /etc/nginx/conf.d/default.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy built assets from stage 1
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
