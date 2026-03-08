"""OntologyEntry model — Phase 7: Enterprise Ontology + Entity Resolution.

Stores canonical entity definitions and their aliases, scoped per organization.
Enables cross-silo entity normalization (e.g. "client" in billing = "customer" in CRM).
"""

from __future__ import annotations

from sqlalchemy import String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin, UUIDMixin


class OntologyEntry(Base, UUIDMixin, TimestampMixin):
    """Canonical entity type with aliases for cross-silo resolution.

    Each row represents one canonical concept (e.g. "customer") with:
    - aliases: alternate names across departments (["client", "account", "buyer"])
    - entity_type: ontology type ("role", "metric", "event", "document", etc.)
    - domains: list of business domains where this entity commonly appears
    """

    __tablename__ = "ontology_entries"

    organization_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        nullable=False,
        index=True,
        doc="Organization scope; NULL-equivalent entries are shared (builtin).",
    )

    canonical_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
        doc="Canonical (preferred) name for this entity.",
    )

    entity_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="concept",
        doc="Ontology type: role, metric, event, document, organization, time_period, concept.",
    )

    # JSONB list of alias strings (lowercased)
    aliases: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Alternate names for this entity across departments.",
    )

    # JSONB list of domain strings where this entity appears
    domains: Mapped[list] = mapped_column(
        JSONB,
        nullable=False,
        default=list,
        doc="Business domains where this entity is commonly referenced.",
    )

    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "canonical_name",
            name="uq_ontology_entry_org_canonical",
        ),
    )
