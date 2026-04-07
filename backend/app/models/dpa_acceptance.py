"""DPA acceptance model for GDPR data processing agreement records."""

from sqlalchemy import Boolean, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin, TenantMixin


class DpaAcceptance(Base, UUIDMixin, TimestampMixin, TenantMixin):
    __tablename__ = "dpa_acceptances"

    dpa_version: Mapped[str] = mapped_column(String(20), nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    accepted_by_user_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    accepted_at: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
