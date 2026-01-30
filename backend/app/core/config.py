"""
Application Configuration
=========================

Centralized configuration using Pydantic Settings.
All configuration is loaded from environment variables with sensible defaults.
"""

from typing import List, Optional
from functools import lru_cache

from pydantic import Field, field_validator, AliasChoices
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Settings are validated using Pydantic and cached for performance.
    See .env.example for all available configuration options.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application
    # -------------------------------------------------------------------------
    APP_NAME: str | None = None
    APP_ENV: str | None = None
    DEBUG: bool | None = None
    LOG_LEVEL: str | None = None

    # -------------------------------------------------------------------------
    # API
    # -------------------------------------------------------------------------
    API_HOST: str | None = None
    API_PORT: int | None = None
    API_PREFIX: str | None = None
    CORS_ORIGINS: List[str] = Field(default_factory=list)

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from string or list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    # -------------------------------------------------------------------------
    # Security
    # -------------------------------------------------------------------------
    SECRET_KEY: str | None = None
    JWT_ALGORITHM: str | None = None
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int | None = None
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int | None = None

    # Authentication mode
    # - "password": only local email/password
    # - "oidc": only OIDC SSO
    # - "both": allow both flows
    AUTH_MODE: str | None = None

    # -------------------------------------------------------------------------
    # OIDC (SSO) - Option A (recommended)
    # -------------------------------------------------------------------------
    # Example issuer: https://login.microsoftonline.com/<tenant-id>/v2.0
    OIDC_ISSUER: str | None = None
    OIDC_CLIENT_ID: str | None = None
    # Optional override if your provider expects a different audience
    OIDC_AUDIENCE: str | None = None
    # Comma-separated list of allowed email domains (e.g. "example.com,example.org")
    OIDC_ALLOWED_EMAIL_DOMAINS: List[str] | None = None
    # Organization mapping defaults for first-time SSO users
    OIDC_DEFAULT_ORG_SLUG: str | None = None
    OIDC_DEFAULT_ORG_ID: str | None = None
    OIDC_DEFAULT_ROLE: str | None = None
    # Claim name that contains group list (provider-specific)
    OIDC_GROUPS_CLAIM: str | None = None
    # JSON mapping from group name -> Ninai role name
    # Example: {"Ninai-Org-Admins": "org_admin", "Ninai-Members": "member"}
    OIDC_GROUP_TO_ROLE_JSON: str | None = None

    @field_validator("OIDC_ALLOWED_EMAIL_DOMAINS", mode="before")
    @classmethod
    def parse_oidc_allowed_domains(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            return [d.strip().lstrip("@") for d in v.split(",") if d.strip()]
        return [str(d).strip().lstrip("@") for d in v]

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    POSTGRES_HOST: str | None = None
    POSTGRES_PORT: int | None = None
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_DB: str | None = None

    # Optional full DSN overrides (used by some deployments and tooling)
    POSTGRES_URL: Optional[str] = None
    POSTGRES_URL_SYNC: Optional[str] = None

    # Test-only DB overrides (used by pytest fixtures)
    TEST_DATABASE_URL: Optional[str] = None
    TEST_DATABASE_URL_SYNC: Optional[str] = None

    @property
    def DATABASE_URL(self) -> str:
        """Build the async database URL."""
        if self.APP_ENV == "test" and self.TEST_DATABASE_URL:
            return self.TEST_DATABASE_URL
        if self.POSTGRES_URL:
            return self.POSTGRES_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    @property
    def DATABASE_URL_SYNC(self) -> str:
        """Build the sync database URL (for Alembic)."""
        if self.APP_ENV == "test" and self.TEST_DATABASE_URL_SYNC:
            return self.TEST_DATABASE_URL_SYNC
        if self.POSTGRES_URL_SYNC:
            return self.POSTGRES_URL_SYNC
        return (
            f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # -------------------------------------------------------------------------
    # Redis
    # -------------------------------------------------------------------------
    REDIS_HOST: str | None = None
    REDIS_PORT: int | None = None
    REDIS_PASSWORD: str | None = None
    REDIS_DB: int | None = None
    PERMISSION_CACHE_TTL: int | None = None

    # Short-term memory default TTL (in seconds)
    SHORT_TERM_TTL: int | None = None

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL."""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # -------------------------------------------------------------------------
    # Qdrant
    # -------------------------------------------------------------------------
    QDRANT_HOST: str | None = None
    QDRANT_PORT: int | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "memories"

    # -------------------------------------------------------------------------
    # Elasticsearch
    # -------------------------------------------------------------------------
    ELASTICSEARCH_HOST: str | None = None
    ELASTICSEARCH_PORT: int | None = None
    ELASTICSEARCH_USER: str | None = None
    ELASTICSEARCH_PASSWORD: str | None = None

    @property
    def ELASTICSEARCH_URL(self) -> str:
        """Construct Elasticsearch URL."""
        return f"http://{self.ELASTICSEARCH_HOST}:{self.ELASTICSEARCH_PORT}"

    # -------------------------------------------------------------------------
    # Celery
    # -------------------------------------------------------------------------
    CELERY_BROKER_URL: str | None = None
    CELERY_RESULT_BACKEND: str | None = None

    # Optional: service user id for background tasks (must be an org_admin/system_admin or superuser).
    SYSTEM_TASK_USER_ID: str | None = None

    # -------------------------------------------------------------------------
    # Alerts & Notifications
    # -------------------------------------------------------------------------
    # alert delivery mode: "log" (no-op, logs only) | "deliver" (actually send)
    ALERT_DELIVERY_MODE: str = "log"
    # email delivery settings (used when ALERT_DELIVERY_MODE=deliver and channel=email)
    SMTP_HOST: str | None = None
    SMTP_PORT: int | None = None
    SMTP_USERNAME: str | None = None
    SMTP_PASSWORD: str | None = None
    SMTP_USE_TLS: bool = True
    SMTP_FROM_EMAIL: str | None = None
    # slack fallback webhook; per-alert webhooks are preferred
    SLACK_DEFAULT_WEBHOOK: str | None = None

    # -------------------------------------------------------------------------
    # LLM (Optional)
    # -------------------------------------------------------------------------
    # Local-first LLM (Ollama). Defaults are safe for local dev; containers should
    # override base URL to reach the `ollama` service (e.g., http://ollama:11434).
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_URL"),
    )
    # Default local model (override via env OLLAMA_MODEL)
    OLLAMA_MODEL: str = "qwen2.5:7b"
    # Keep this low so missing Ollama doesn't stall the pipeline; agents will
    # fall back to heuristics automatically.
    OLLAMA_TIMEOUT_SECONDS: float = 5.0

    # Limit concurrent Ollama requests per worker process.
    OLLAMA_MAX_CONCURRENCY: int = 2

    # Global agent strategy (advanced): set to "heuristic" to disable LLM calls.
    AGENT_STRATEGY: str = "llm"  # llm | heuristic

    # ---------------------------------------------------------------------
    # Sandbox / reproducible demos
    # ---------------------------------------------------------------------
    # If enabled, LLM helper endpoints can return deterministic stub outputs
    # so notebooks and SDK examples run without Ollama.
    SANDBOX_LLM_ENABLED: bool = False

    # If true, /llm/* helper endpoints require org admin. For local demos you
    # can set this false (ideally alongside SANDBOX_LLM_ENABLED).
    LLM_ADMIN_ONLY: bool = True

    # Cache agent results keyed by stable input hash (reduces repeated LLM calls).
    AGENT_CACHE_ENABLED: bool = True
    # Optional TTL for cache entries (seconds). If unset/0, entries do not expire.
    AGENT_CACHE_TTL_SECONDS: int | None = None

    # Per-agent override (advanced): set to "heuristic" to disable LLM calls for metadata.
    METADATA_EXTRACTION_STRATEGY: str | None = None

    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # -------------------------------------------------------------------------
    # Memory Attachments (Multimodal MVP)
    # -------------------------------------------------------------------------
    # Directory to store attachment bytes (can be a mounted volume in Docker).
    MEMORY_ATTACHMENTS_DIR: str | None = None
    # Maximum upload size in bytes (default 25MB).
    MAX_ATTACHMENT_SIZE_BYTES: int | None = None
    # Enable extracted-text indexing into Qdrant.
    ATTACHMENT_INDEXING_ENABLED: bool | None = None
    # Maximum characters to embed per attachment.
    ATTACHMENT_INDEX_MAX_CHARS: int | None = None
    # Maximum file size (bytes) eligible for extraction/indexing.
    ATTACHMENT_INDEX_MAX_FILE_BYTES: int | None = None
    # Read at most N bytes from head of file for indexing (default 2MB).
    ATTACHMENT_INDEX_READ_HEAD_BYTES: int | None = None

    # Optional OCR sidecar (HTTP API around tesseract)
    OCR_SERVICE_URL: str | None = None
    OCR_SERVICE_TIMEOUT_SECONDS: float | None = None

    # -------------------------------------------------------------------------
    # Search Ranking
    # -------------------------------------------------------------------------
    # If enabled, downranks older memories using a half-life decay.
    SEARCH_TEMPORAL_DECAY_ENABLED: bool = False
    # Half-life in days for ranking decay (smaller = more aggressive).
    SEARCH_TEMPORAL_DECAY_HALF_LIFE_DAYS: float = 30.0

    # HNMS-inspired ranking modes (configurable + optional request override).
    # - balanced: use SEARCH_TEMPORAL_DECAY_* knobs
    # - performance: stronger recency bias
    # - research: weaker recency bias
    SEARCH_HNMS_MODE_DEFAULT: str = "balanced"
    SEARCH_HNMS_MODE_ALLOW_REQUEST_OVERRIDE: bool = True
    SEARCH_HNMS_MODE_PERFORMANCE_HALF_LIFE_DAYS: float = 7.0
    SEARCH_HNMS_MODE_RESEARCH_HALF_LIFE_DAYS: float = 90.0

    # Optional feedback-driven reranking (closed-loop retrieval).
    # If enabled, recent per-user relevance feedback can boost/downrank results.
    SEARCH_FEEDBACK_RERANK_ENABLED: bool = False
    # How far back to consider relevance feedback when reranking.
    SEARCH_FEEDBACK_RERANK_WINDOW_DAYS: float = 90.0
    # Multiplier applied to the base score for positive/negative relevance.
    SEARCH_FEEDBACK_RERANK_POSITIVE_MULTIPLIER: float = 1.15
    SEARCH_FEEDBACK_RERANK_NEGATIVE_MULTIPLIER: float = 0.5

    # -------------------------------------------------------------------------
    # Logseq Integration
    # -------------------------------------------------------------------------
    # Directory for admin-only Logseq Markdown exports.
    LOGSEQ_EXPORT_DIR: str | None = None

    # Default export mode for admin-only write-to-disk exports.
    # - single_file: one .md file per export request (current behavior)
    # - vault_pages: Logseq vault structure with one page per memory
    LOGSEQ_EXPORT_MODE: str | None = None

    # -------------------------------------------------------------------------
    # Snapshot/Export Jobs
    # -------------------------------------------------------------------------
    # Directory for snapshot export bundles (zip artifacts). Defaults to exports/snapshots.
    SNAPSHOT_EXPORT_DIR: str | None = None

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.APP_ENV == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.APP_ENV == "production"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once and reused.
    
    Returns:
        Settings: Validated settings instance
    """
    return Settings()


# Global settings instance
settings = get_settings()
