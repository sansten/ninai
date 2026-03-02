"""
PR-7: Compositional Generalization Engine models.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List

from sqlalchemy import Float, Index, Integer, JSON, String, TIMESTAMP
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class AnalogyApplicability(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    METAPHORICAL = "metaphorical"


class AbstractProcedure(Base, UUIDMixin, TimestampMixin):
    """Abstraction of a concrete procedure/playbook."""

    __tablename__ = "abstract_procedures"

    organization_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    concrete_playbook_id: Mapped[str] = mapped_column(String(36), nullable=False)
    abstraction_level: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    parameters: Mapped[Dict[str, str]] = mapped_column(JSON, default={})
    prerequisites: Mapped[List[str]] = mapped_column(JSON, default=[])
    postconditions: Mapped[List[str]] = mapped_column(JSON, default=[])
    invariants: Mapped[List[str]] = mapped_column(JSON, default=[])
    instances: Mapped[List[str]] = mapped_column(JSON, default=[])

    __table_args__ = (
        Index("idx_abstract_procedures_org_level", "organization_id", "abstraction_level"),
    )


class Analogy(Base, UUIDMixin):
    """Structural similarity between two problem domains."""

    __tablename__ = "analogies"

    organization_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    source_domain: Mapped[str] = mapped_column(String(255), nullable=False)
    target_domain: Mapped[str] = mapped_column(String(255), nullable=False)
    structural_similarity: Mapped[float] = mapped_column(Float, default=0.5)
    mapped_concepts: Mapped[Dict[str, str]] = mapped_column(JSON, default={})
    constraints: Mapped[List[str]] = mapped_column(JSON, default=[])
    applicability: Mapped[str] = mapped_column(String(50), default=AnalogyApplicability.PARTIAL.value)
    discovered_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP, nullable=False)

    __table_args__ = (
        Index("idx_analogies_org_domains", "organization_id", "source_domain", "target_domain"),
    )
