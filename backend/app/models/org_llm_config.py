"""Multi-org LLM provider and model configuration."""

from sqlalchemy import Boolean, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin, TenantMixin

VALID_PROVIDERS = frozenset({"ollama", "openai", "anthropic"})


class OrgLlmConfig(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "org_llm_configs"

    provider: Mapped[str] = mapped_column(String(30), nullable=False, default="ollama")
    model: Mapped[str] = mapped_column(String(200), nullable=False, default="qwen2.5:7b")
    api_key_ref: Mapped[str | None] = mapped_column(String(500), nullable=True)
    base_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    config: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (UniqueConstraint("organization_id", name="uq_org_llm_config_org"),)
