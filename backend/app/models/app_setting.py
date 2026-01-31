"""Application settings model.

Stores runtime-editable configuration that can be managed from an admin UI.
Values are stored as JSON so we can version/extend settings without schema churn.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class AppSetting(Base, TimestampMixin):
    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(
        String(length=100),
        primary_key=True,
        doc="Setting key (unique)",
    )

    value: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="JSON value for this setting",
    )

    updated_by_user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        nullable=True,
        doc="User who last updated this setting",
    )
