"""
PR-10: Federated Memory & Collective Intelligence Models

Privacy-preserving knowledge sharing across organizations.
Enables collective learning while maintaining data sovereignty.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from app.models.base import Base
import uuid


class FederatedKnowledgeSummary(Base):
    """
    Anonymized, aggregated knowledge shared across organizations.
    
    Represents collective intelligence extracted from multiple orgs' memories
    without exposing individual organization data. Used for cross-org learning
    and best practice sharing.
    
    Attributes:
        id: Unique identifier
        summary_type: Type of aggregated knowledge (pattern/playbook/best_practice/benchmark)
        domain: Knowledge domain (e.g., "customer_support", "deployment", "incident_response")
        content: Anonymized aggregated content (domain-specific structure)
        aggregated_from_org_ids: Hashed/anonymized org identifiers (for traceability)
        count_contributing_orgs: Number of organizations that contributed
        quality_score: Validation score (0-1) based on multi-org validation
        created_at: When this summary was first created
        updated_at: Last update timestamp
    """
    __tablename__ = "federated_knowledge_summaries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    summary_type = Column(String, nullable=False, index=True)  # "pattern" | "playbook" | "best_practice" | "benchmark"
    domain = Column(String, nullable=False, index=True)
    content = Column(JSON, nullable=False, default=dict)
    aggregated_from_org_ids = Column(JSON, default=list)  # anonymized org identifiers
    count_contributing_orgs = Column(Integer, default=0)
    quality_score = Column(Float, default=0.0)  # 0-1
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class OrgBenchmark(Base):
    """
    Benchmarking data comparing organization against peer group.
    
    Tracks key performance metrics and provides percentile rankings
    relative to similar organizations (by size, industry, etc).
    
    Attributes:
        id: Unique identifier
        organization_id: Organization being benchmarked
        metric_type: Type of metric (memory_quality/response_time/customer_satisfaction)
        value: Actual metric value for this org
        percentile: Where this org ranks (0-1) vs peers
        peer_count: Number of peer organizations in comparison
        trend: Current trajectory (improving/stable/declining)
        last_computed_at: When benchmark was last calculated
    """
    __tablename__ = "org_benchmarks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    metric_type = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    percentile = Column(Float)  # 0-1
    peer_count = Column(Integer, default=0)
    trend = Column(String)  # "improving" | "stable" | "declining"
    last_computed_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('organization_id', 'metric_type', name='uq_org_metric'),
    )


class PrivacyPolicy(Base):
    """
    Organization-level privacy policy for federated knowledge sharing.
    
    Controls what data can be shared in federated learning and at what
    anonymization level. Organizations maintain full control over their
    data contribution.
    
    Attributes:
        id: Unique identifier
        organization_id: Organization this policy belongs to
        allow_pattern_sharing: Share behavioral patterns (anonymized)?
        allow_playbook_contribution: Contribute playbooks to federation?
        allow_benchmark_participation: Include in peer benchmarking?
        minimum_anonymization_level: Required anonymization (full/partial/none)
        excluded_domains: Topics/domains to never share
        created_at: Policy creation timestamp
        updated_at: Last policy update
    """
    __tablename__ = "privacy_policies"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, unique=True, index=True)
    allow_pattern_sharing = Column(Boolean, default=False)
    allow_playbook_contribution = Column(Boolean, default=False)
    allow_benchmark_participation = Column(Boolean, default=False)
    minimum_anonymization_level = Column(String, default="full")  # "full" | "partial" | "none"
    excluded_domains = Column(JSON, default=list)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class FederatedContribution(Base):
    """
    Tracks individual contributions to federated knowledge.
    
    Records when organizations contribute data to the federation,
    enabling credit attribution and contribution analytics.
    
    Attributes:
        id: Unique identifier
        organization_id: Contributing organization
        contribution_type: Type of contribution (pattern/playbook/benchmark_data)
        federated_knowledge_id: Reference to created federated knowledge
        anonymization_applied: Level of anonymization applied
        contribution_date: When contribution was made
        impact_score: How valuable has this contribution been? (0-1)
    """
    __tablename__ = "federated_contributions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    contribution_type = Column(String, nullable=False)
    federated_knowledge_id = Column(UUID(as_uuid=True), index=True)
    anonymization_applied = Column(String, default="full")
    contribution_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    impact_score = Column(Float, default=0.0)  # 0-1, updated as others use it
