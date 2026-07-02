# =============================================================================
# Ninai Backend Dockerfile
# =============================================================================

FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

# Set work directory
WORKDIR /app

# Install Python dependencies and pre-download the CE reranker model.
# TRANSFORMERS_OFFLINE is NOT set yet so the download can reach HuggingFace.
# Force CPU-only torch first so pip does not resolve the CUDA build (saves ~500MB).
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('BAAI/bge-reranker-base')"

# Now lock down HuggingFace access — the model is cached at HF_HOME.
# Prevents 50s startup hangs in GKE (no internet) from HEAD requests.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

# Copy application code
COPY . .

# Copy and set permissions for entrypoint script (must be done as root)
COPY entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r//' /entrypoint.sh && \
    chmod +x /entrypoint.sh && \
    adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app && \
    chown appuser:appuser /entrypoint.sh && \
    echo "ninai:x:1001:0::/tmp:/bin/sh" >> /etc/passwd

USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys, urllib.request; urllib.request.urlopen('http://localhost:8000/health'); sys.exit(0)"

# Run application with entrypoint script
ENTRYPOINT ["/entrypoint.sh"]
