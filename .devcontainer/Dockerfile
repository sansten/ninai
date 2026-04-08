# Dockerfile for Ninai2 Development Container
# Based on Microsoft's Python development image

FROM mcr.microsoft.com/devcontainers/python:3.12

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    redis-tools \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js (for frontend)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Docker CLI
RUN curl -fsSL https://get.docker.com -o get-docker.sh && \
    sh get-docker.sh && \
    rm get-docker.sh

# Install Poetry for Python dependency management
RUN pip install --no-cache-dir --user poetry

# Create non-root user (vscode)
RUN groupadd -r vscode && useradd -r -g vscode vscode || true

# Add vscode user to docker group
RUN usermod -aG docker vscode || true

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV PATH="/home/vscode/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    LOG_LEVEL=debug

# Health check for container
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["/bin/bash"]
