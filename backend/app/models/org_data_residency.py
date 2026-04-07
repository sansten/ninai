"""OrgDataResidency - data residency declaration per org."""

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin, TenantMixin

VALID_REGIONS = frozenset({"us", "eu", "apac", "ca"})


class OrgDataResidency(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "org_data_residency"

    region: Mapped[str] = mapped_column(String(20), nullable=False, default="us")
    gdpr_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    backup_region: Mapped[str | None] = mapped_column(String(20), nullable=True)
    gcp_region: Mapped[str] = mapped_column(String(50), nullable=False, default="us-central1")
    declared_at: Mapped[str | None] = mapped_column(String(50), nullable=True)
