"""OrgFeatureFlag - per-org feature flag overrides for gradual rollout."""

from sqlalchemy import Boolean, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TenantMixin, TimestampMixin, UUIDMixin


class OrgFeatureFlag(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "org_feature_flags"

    flag_name: Mapped[str] = mapped_column(String(200), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    rollout_pct: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False, default=dict)

    __table_args__ = (
        UniqueConstraint("organization_id", "flag_name", name="uq_org_feature_flags_org_flag"),
    )
