"""
PR-10: Federated Memory Pydantic Schemas

Request/response models for federated knowledge and benchmarking APIs.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum


# Enums

class SummaryType(str, Enum):
    """Type of federated knowledge summary."""
    PATTERN = "pattern"
    PLAYBOOK = "playbook"
    BEST_PRACTICE = "best_practice"
    BENCHMARK = "benchmark"


class PublicityLevel(str, Enum):
    """Playbook sharing scope."""
    PRIVATE = "private"
    ORG_ONLY = "org_only"
    PEER_GROUP = "peer_group"
    PUBLIC = "public"


class AnonymizationLevel(str, Enum):
    """Data anonymization level."""
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class TrendType(str, Enum):
    """Performance trend direction."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


# Federated Knowledge Schemas

class FederatedKnowledgeBase(BaseModel):
    """Base schema for federated knowledge."""
    summary_type: SummaryType
    domain: str
    content: Dict[str, Any]
    
    model_config = ConfigDict(from_attributes=True)


class FederatedKnowledgeResponse(FederatedKnowledgeBase):
    """Response schema for federated knowledge."""
    id: str
    count_contributing_orgs: int
    quality_score: float
    created_at: datetime
    updated_at: datetime


class ContributePatternRequest(BaseModel):
    """Request to contribute organization patterns to federation."""
    domain: str = Field(..., description="Knowledge domain")
    min_contributors: int = Field(default=3, ge=1, description="Minimum orgs needed for aggregate")


class ContributePlaybookRequest(BaseModel):
    """Request to share a playbook with federation."""
    playbook_id: str = Field(..., description="ID of playbook to contribute")
    publicity_level: PublicityLevel = Field(default=PublicityLevel.PEER_GROUP)
    domain: str = Field(default="general", description="Playbook domain")


class ContributePlaybookResponse(BaseModel):
    """Response after contributing playbook."""
    success: bool
    federated_knowledge_id: Optional[str] = None
    message: str


class GetFederatedKnowledgeRequest(BaseModel):
    """Request to retrieve federated knowledge."""
    domain: str
    problem_type: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)


class ValidateFederatedKnowledgeRequest(BaseModel):
    """Request to validate federated knowledge."""
    knowledge_id: str
    is_helpful: bool
    feedback: Optional[str] = None


class ContributionStatsResponse(BaseModel):
    """Response with contribution statistics."""
    total_contributions: int
    by_type: Dict[str, int]
    average_impact_score: float
    first_contribution: Optional[datetime]
    last_contribution: Optional[datetime]


# Benchmark Schemas

class OrgBenchmarkBase(BaseModel):
    """Base schema for organization benchmark."""
    metric_type: str
    value: float
    
    model_config = ConfigDict(from_attributes=True)


class OrgBenchmarkResponse(OrgBenchmarkBase):
    """Response schema for organization benchmark."""
    id: str
    organization_id: str
    percentile: float
    peer_count: int
    trend: TrendType
    last_computed_at: datetime


class ComputeBenchmarksRequest(BaseModel):
    """Request to compute organization benchmarks."""
    metrics: Optional[List[str]] = Field(
        default=None,
        description="Specific metrics to compute (None = all)"
    )


class BenchmarkReportResponse(BaseModel):
    """Response with comprehensive benchmark report."""
    organization_id: str
    report_date: str
    overall_health_percentile: float
    overall_ranking: str
    metrics: Dict[str, Dict[str, Any]]
    improvements: List[str]
    recommendations: List[str]
    peer_comparison_count: int


class BenchmarkHistoryResponse(BaseModel):
    """Response with historical benchmark data."""
    metric_type: str
    history: List[Dict[str, Any]]


class TargetComparisonRequest(BaseModel):
    """Request to compare against target percentile."""
    target_percentile: float = Field(default=0.75, ge=0.0, le=1.0)


class GapAnalysis(BaseModel):
    """Gap analysis for a single metric."""
    metric: str
    current_percentile: float
    target_percentile: float
    gap: float
    priority: str
    recommendation: str


class TargetComparisonResponse(BaseModel):
    """Response with gap analysis vs target."""
    organization_id: str
    target_percentile: float
    gaps: List[GapAnalysis]
    met_target_count: int
    total_metrics: int


# Privacy Policy Schemas

class PrivacyPolicyBase(BaseModel):
    """Base schema for privacy policy."""
    allow_pattern_sharing: bool = Field(default=False)
    allow_playbook_contribution: bool = Field(default=False)
    allow_benchmark_participation: bool = Field(default=False)
    minimum_anonymization_level: AnonymizationLevel = Field(default=AnonymizationLevel.FULL)
    excluded_domains: List[str] = Field(default_factory=list)


class PrivacyPolicyCreate(PrivacyPolicyBase):
    """Request to create or update privacy policy."""
    pass


class PrivacyPolicyResponse(PrivacyPolicyBase):
    """Response schema for privacy policy."""
    id: str
    organization_id: str
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# Differential Privacy Schemas

class AddNoiseRequest(BaseModel):
    """Request to add differential privacy noise."""
    data: List[float]
    sensitivity: float = Field(default=1.0, ge=0.0)
    epsilon: Optional[float] = Field(default=None, ge=0.0)


class AddNoiseResponse(BaseModel):
    """Response with noisy data."""
    noisy_data: List[float]
    epsilon_used: float
    noise_scale: float


class PrivateAggregateRequest(BaseModel):
    """Request to compute private aggregate."""
    aggregation_type: str = Field(
        default="mean",
        description="Type: mean, median, sum, count, variance"
    )
    epsilon: Optional[float] = Field(default=None, ge=0.0)


class PrivateAggregateResponse(BaseModel):
    """Response with privatized aggregate."""
    result: float
    aggregation_type: str
    epsilon_used: float
    org_count: int


class ReidentificationRiskRequest(BaseModel):
    """Request to estimate re-identification risk."""
    org_count: int = Field(..., ge=1)
    data_dimension: int = Field(..., ge=1)
    epsilon: Optional[float] = None


class ReidentificationRiskResponse(BaseModel):
    """Response with risk assessment."""
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk of re-identification")
    org_count: int
    data_dimension: int
    epsilon: float
    risk_level: str  # "low" | "medium" | "high"
    recommendation: str


# Federation Health Schemas

class FederationHealthResponse(BaseModel):
    """Response with federation health metrics."""
    total_organizations: int
    participating_orgs: int
    participation_rate: float
    total_federated_knowledge: int
    by_type: Dict[str, int]
    average_quality_score: float
    total_benchmarks_computed: int
    privacy_compliance_rate: float
