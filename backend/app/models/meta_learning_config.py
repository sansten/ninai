"""Meta-learning configuration model (Phase 74)."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, UUIDMixin


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MetaLearningConfig(Base, UUIDMixin):
    __tablename__ = "meta_learning_configs"

    org_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    ema_alpha: Mapped[float] = mapped_column(Float, nullable=False, default=0.25)
    noise_threshold: Mapped[float] = mapped_column(Float, nullable=False, default=0.85)
    confidence_floor: Mapped[float] = mapped_column(Float, nullable=False, default=0.4)
    decay_half_life_days: Mapped[int] = mapped_column(Integer, nullable=False, default=30)
    calibration_window: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    last_tuned: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utc_now,
        server_default=func.now(),
        onupdate=func.now(),
    )
    tuning_iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ux_meta_learning_configs_org", "org_id", unique=True),
    )
