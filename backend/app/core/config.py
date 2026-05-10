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
    DEPLOYMENT_MODE: str = "cloud"
    LOCAL_FIRST_MODE: bool = False

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

    # PR-10: Organization federated login gate (disabled by default)
    ORG_FEDERATED_LOGIN_ENABLED: bool = False

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

    # Optional DB pooler DSN (for PgBouncer-style deployments)
    POSTGRES_POOLER_URL: Optional[str] = None
    POSTGRES_POOLER_URL_SYNC: Optional[str] = None

    # SQLAlchemy connection pool tuning
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    DB_POOL_TIMEOUT_SECONDS: int = 30
    DB_POOL_RECYCLE_SECONDS: int = 1800
    DB_POOL_USE_LIFO: bool = True

    @property
    def DATABASE_URL(self) -> str:
        """Build the async database URL."""
        if self.APP_ENV == "test" and self.TEST_DATABASE_URL:
            return self.TEST_DATABASE_URL
        if self.POSTGRES_POOLER_URL:
            return self.POSTGRES_POOLER_URL
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
        if self.POSTGRES_POOLER_URL_SYNC:
            return self.POSTGRES_POOLER_URL_SYNC
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
    REDIS_URL_OVERRIDE: str | None = Field(
        default=None,
        validation_alias=AliasChoices("REDIS_URL"),
    )
    PERMISSION_CACHE_TTL: int | None = None

    # Short-term memory default TTL (in seconds)
    SHORT_TERM_TTL: int | None = None
    
    # Short-term memory promotion strategy: "count", "spacing", "hybrid"
    STM_PROMOTION_STRATEGY: str = "spacing"
    
    # Access count threshold for promotion (used in count/hybrid modes)
    STM_ACCESS_COUNT_THRESHOLD: int = 3
    
    # Minimum hours between accesses for spacing validation (used in spacing/hybrid modes)
    STM_MIN_ACCESS_SPACING_HOURS: float = 6.0
    
    # Importance score threshold for immediate promotion bypass
    STM_IMPORTANCE_THRESHOLD: float = 0.7

    @property
    def REDIS_URL(self) -> str:
        """Construct Redis URL."""
        if self.REDIS_URL_OVERRIDE:
            return self.REDIS_URL_OVERRIDE

        host = self.REDIS_HOST
        port = self.REDIS_PORT
        db = self.REDIS_DB if self.REDIS_DB is not None else 0

        if host and port is not None:
            if self.REDIS_PASSWORD:
                return f"redis://:{self.REDIS_PASSWORD}@{host}:{port}/{db}"
            return f"redis://{host}:{port}/{db}"

        if self.CELERY_BROKER_URL:
            return self.CELERY_BROKER_URL

        # Last-resort local fallback for developer environments.
        return "redis://localhost:6379/0"

    # Dedicated Redis endpoint for FalkorDB graph queries.
    # Falls back to REDIS_URL when not provided.
    GRAPH_REDIS_URL: str | None = None

    # -------------------------------------------------------------------------
    # Qdrant
    # -------------------------------------------------------------------------
    QDRANT_HOST: str | None = None
    QDRANT_PORT: int | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION_NAME: str = "memories"
    QDRANT_URL: str | None = None

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
    WRITE_TIME_AGENT_ENABLED_TIERS: List[int] = Field(default_factory=lambda: [1, 2])

    @field_validator("WRITE_TIME_AGENT_ENABLED_TIERS", mode="before")
    @classmethod
    def parse_write_time_agent_enabled_tiers(cls, v):
        if v is None or v == "":
            return [1, 2]
        if isinstance(v, str):
            return [int(part.strip()) for part in v.split(",") if part.strip()]
        return [int(part) for part in v]

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
    # SaaS / Signup
    # -------------------------------------------------------------------------
    CURRENT_API_VERSION: str = "v1"
    SENDGRID_API_KEY: str | None = None
    EMAIL_FROM: str = "noreply@ninai.app"
    EMAIL_FROM_NAME: str = "Ninai"
    # Used in email links (verification URL, upgrade CTA)
    FRONTEND_URL: str = "https://ninai.app"

    # Trial defaults — individual org limits come from OrgSubscription.seat_limit
    TRIAL_DAYS: int = 14
    TRIAL_MAX_MEMORIES: int = 1_000
    TRIAL_MAX_USERS: int = 3
    TRIAL_MAX_API_CALLS_PER_DAY: int = 500

    # -------------------------------------------------------------------------
    # LLM (Optional)
    # -------------------------------------------------------------------------
    # Local-first LLM (Ollama). Defaults are safe for local dev; containers should
    # override base URL to reach the `ollama` service (e.g., http://ollama:11434).
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "OLLAMA_URL"),
    )
    # Optional split-route endpoints for CPU-default + GPU-overflow behavior.
    # If unset, OLLAMA_BASE_URL is used as the primary endpoint.
    OLLAMA_BASE_URL_CPU: str | None = None
    OLLAMA_BASE_URL_GPU: str | None = None
    # Enable automatic spillover to GPU endpoint on overload/failure.
    OLLAMA_OVERFLOW_ENABLED: bool = False
    # If >0, route requests to GPU first when in-flight requests on CPU endpoint
    # reach this threshold (per worker process).
    OLLAMA_OVERFLOW_PRIMARY_MAX_INFLIGHT: int = 0
    # Default local model (override via env OLLAMA_MODEL)
    OLLAMA_MODEL: str = "qwen2.5:7b"
    # Optional per-purpose model overrides (fall back to OLLAMA_MODEL when unset).
    OLLAMA_MODEL_FAST: str | None = None
    OLLAMA_MODEL_REASONING: str | None = None
    OLLAMA_MODEL_PLANNING: str | None = None
    OLLAMA_MODEL_DISTILLATION: str | None = None
    OLLAMA_MODEL_BOUNDARY: str | None = None
    OLLAMA_MODEL_UNCERTAINTY: str | None = None
    OLLAMA_MODEL_AGENTS: str | None = None
    # Timeout per Ollama request; 30s accommodates 7b+ models without stalling.
    OLLAMA_TIMEOUT_SECONDS: float = 30.0

    # Limit concurrent Ollama requests per worker process.
    OLLAMA_MAX_CONCURRENCY: int = 4

    # Global agent strategy (advanced):
    # - heuristic: disable LLM calls
    # - llm: prefer LLM and fall back to heuristics on failure
    # - hybrid: alias of llm (explicitly documented for deploy ergonomics)
    AGENT_STRATEGY: str = "llm"  # llm | heuristic | hybrid(alias)

    # -------------------------------------------------------------------------
    # AD / SCIM Identity Enrichment (opt-in, fully optional)
    # -------------------------------------------------------------------------
    # When AD_SCIM_ENDPOINT is not set, IdentityResolverService falls back to
    # JWT claims for role and uses no department/location enrichment.
    AD_SCIM_ENDPOINT: Optional[str] = None
    """e.g. https://graph.microsoft.com/v1.0  or  https://your-idp/scim/v2"""
    AD_SCIM_TOKEN: Optional[str] = None
    """Bearer token for the AD/SCIM endpoint."""
    AD_SCIM_TENANT_ID: Optional[str] = None
    """Azure AD tenant ID (Microsoft Graph only)."""

    @field_validator("AGENT_STRATEGY", mode="before")
    @classmethod
    def normalize_agent_strategy(cls, v):
        """Normalize global agent strategy to backend-supported runtime values.

        We keep runtime branching binary (llm vs heuristic) to avoid touching all
        agents/services, while allowing deploy configs to use a clearer "hybrid"
        label for llm-with-fallback behavior.
        """
        strategy = str(v or "llm").strip().lower()
        if strategy == "hybrid":
            return "llm"
        if strategy in {"llm", "heuristic"}:
            return strategy
        return "llm"

    # Enable LLM-based causal edge discovery (off by default to avoid slow Ollama calls).
    ENABLE_LLM_CAUSAL_DISCOVERY: bool = False

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
    EMBEDDING_PROVIDER: str = "auto"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"

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

    # Graph-guided vector retrieval expansion.
    # When enabled and request.use_graph=true, retrieval will:
    # 1) seed from initial vector hits,
    # 2) expand through graph_relationships,
    # 3) run additional enriched vector queries,
    # 4) blend vector + lexical + graph evidence.
    SEARCH_GRAPH_EXPANSION_ENABLED: bool = True
    SEARCH_GRAPH_SEED_LIMIT: int = 12
    SEARCH_GRAPH_NEIGHBOR_LIMIT: int = 32
    SEARCH_MULTI_QUERY_MAX_VARIANTS: int = 4

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

    def get_ollama_model(self, purpose: str | None = None) -> str:
        """Resolve the Ollama model for a specific purpose.

        If no purpose-specific override is configured, this returns OLLAMA_MODEL.
        """
        default_model = str(self.OLLAMA_MODEL or "").strip() or "qwen2.5:7b"
        if not purpose:
            return default_model

        purpose_key = str(purpose).strip().lower()
        overrides = {
            "fast": self.OLLAMA_MODEL_FAST,
            "reasoning": self.OLLAMA_MODEL_REASONING,
            "planning": self.OLLAMA_MODEL_PLANNING,
            "distillation": self.OLLAMA_MODEL_DISTILLATION,
            "boundary": self.OLLAMA_MODEL_BOUNDARY,
            "uncertainty": self.OLLAMA_MODEL_UNCERTAINTY,
            "agents": self.OLLAMA_MODEL_AGENTS,
            "enrichment": self.OLLAMA_MODEL_AGENTS,
        }
        selected = overrides.get(purpose_key)
        if selected and str(selected).strip():
            return str(selected).strip()
        return default_model


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
