"""
PR-10: Federated Memory & Collective Intelligence API Endpoints

REST API for privacy-preserving knowledge sharing and benchmarking.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List
from datetime import datetime

from app.core.database import get_db
from app.core.config import settings
from app.models.federated_knowledge import PrivacyPolicy
from app.services.federated_knowledge_service import FederatedKnowledgeService
from app.services.org_benchmark_service import OrgBenchmarkService
from app.services.differential_privacy_service import DifferentialPrivacyService
from app.schemas.federated_memory_pr10 import (
    FederatedKnowledgeResponse,
    ContributePatternRequest,
    ContributePlaybookRequest,
    ContributePlaybookResponse,
    GetFederatedKnowledgeRequest,
    ValidateFederatedKnowledgeRequest,
    ContributionStatsResponse,
    OrgBenchmarkResponse,
    ComputeBenchmarksRequest,
    BenchmarkReportResponse,
    BenchmarkHistoryResponse,
    TargetComparisonRequest,
    TargetComparisonResponse,
    PrivacyPolicyCreate,
    PrivacyPolicyResponse,
    AddNoiseRequest,
    AddNoiseResponse,
    PrivateAggregateRequest,
    PrivateAggregateResponse,
    ReidentificationRiskRequest,
    ReidentificationRiskResponse,
    FederationHealthResponse
)
from sqlalchemy import select, func


def require_org_federated_login_enabled() -> None:
    """Gate PR-10 federated routes behind explicit config opt-in."""
    if not settings.ORG_FEDERATED_LOGIN_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Organization federated login is disabled",
        )


router = APIRouter(
    prefix="/federated",
    tags=["federated-memory"],
    dependencies=[Depends(require_org_federated_login_enabled)],
)


# Federated Knowledge Endpoints

@router.post("/contribute-pattern", response_model=FederatedKnowledgeResponse)
async def contribute_pattern(
    request: ContributePatternRequest,
    org_id: str,  # In production: extract from auth token
    db: AsyncSession = Depends(get_db)
):
    """
    Contribute organization's anonymized patterns to federation.
    
    Requires privacy policy allowing pattern_sharing.
    Patterns are anonymized and aggregated with peer organizations.
    """
    service = FederatedKnowledgeService(db)
    
    summary = await service.anonymize_and_aggregate_patterns(
        org_id=org_id,
        domain=request.domain,
        min_contributors=request.min_contributors
    )
    
    if not summary:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Privacy policy does not allow pattern sharing or domain is excluded"
        )
    
    return summary


@router.post("/contribute-playbook", response_model=ContributePlaybookResponse)
async def contribute_playbook(
    request: ContributePlaybookRequest,
    org_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Share a playbook with the federation.
    
    Publicity levels:
    - private: Not shared
    - org_only: Within organization
    - peer_group: Similar organizations
    - public: All federation members
    """
    service = FederatedKnowledgeService(db)
    
    federated_id = await service.contribute_playbook(
        org_id=org_id,
        playbook_id=request.playbook_id,
        publicity_level=request.publicity_level.value,
        domain=request.domain
    )
    
    if not federated_id:
        return ContributePlaybookResponse(
            success=False,
            message="Privacy policy does not allow playbook contribution"
        )
    
    return ContributePlaybookResponse(
        success=True,
        federated_knowledge_id=federated_id,
        message="Playbook successfully contributed to federation"
    )


@router.get("/knowledge", response_model=List[FederatedKnowledgeResponse])
async def get_federated_knowledge(
    domain: str,
    problem_type: str = None,
    limit: int = 10,
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve relevant aggregated knowledge from federation.
    
    Returns cross-organizational insights, best practices, and patterns
    relevant to the specified domain and problem type.
    """
    service = FederatedKnowledgeService(db)
    
    summaries = await service.get_relevant_federated_knowledge(
        org_id=org_id,
        domain=domain,
        problem_type=problem_type,
        limit=limit
    )
    
    return summaries


@router.post("/knowledge/{knowledge_id}/validate")
async def validate_knowledge(
    knowledge_id: str,
    request: ValidateFederatedKnowledgeRequest,
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Provide feedback on federated knowledge usefulness.
    
    Quality scores improve when multiple organizations validate.
    """
    service = FederatedKnowledgeService(db)
    
    success = await service.validate_federated_knowledge(
        org_id=org_id,
        knowledge_id=knowledge_id,
        is_helpful=request.is_helpful,
        feedback=request.feedback
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Federated knowledge not found"
        )
    
    return {"success": True, "message": "Validation recorded"}


@router.get("/contributions", response_model=ContributionStatsResponse)
async def get_contribution_stats(
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics about organization's contributions to federation.
    
    Shows impact of shared knowledge and contribution trends.
    """
    service = FederatedKnowledgeService(db)
    stats = await service.get_contribution_stats(org_id)
    return stats


# Benchmark Endpoints

@router.post("/benchmarks/compute", response_model=List[OrgBenchmarkResponse])
async def compute_benchmarks(
    request: ComputeBenchmarksRequest,
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Compute organization's performance benchmarks vs peer group.
    
    Metrics:
    - memory_quality: Overall memory system health
    - response_time: Average response latency
    - customer_satisfaction: User satisfaction scores
    - tool_success_rate: Tool execution success
    - knowledge_coverage: Breadth of domains
    """
    service = OrgBenchmarkService(db)
    
    benchmarks = await service.compute_org_benchmarks(
        org_id=org_id,
        metrics=request.metrics
    )
    
    if not benchmarks:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Privacy policy does not allow benchmark participation"
        )
    
    return benchmarks


@router.get("/benchmarks/report", response_model=BenchmarkReportResponse)
async def get_benchmark_report(
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Generate comprehensive benchmarking report.
    
    Includes:
    - Overall health score and percentile ranking
    - Per-metric performance vs peers
    - Trend analysis (improving/stable/declining)
    - Actionable recommendations
    """
    service = OrgBenchmarkService(db)
    report = await service.get_benchmark_report(org_id)
    return report


@router.get("/benchmarks/history/{metric_type}", response_model=BenchmarkHistoryResponse)
async def get_benchmark_history(
    metric_type: str,
    lookback_days: int = 90,
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Get historical benchmark data for trend analysis.
    
    Shows how organization's performance has evolved over time.
    """
    service = OrgBenchmarkService(db)
    
    history = await service.get_benchmark_history(
        org_id=org_id,
        metric_type=metric_type,
        lookback_days=lookback_days
    )
    
    return BenchmarkHistoryResponse(
        metric_type=metric_type,
        history=history
    )


@router.post("/benchmarks/compare-target", response_model=TargetComparisonResponse)
async def compare_with_target(
    request: TargetComparisonRequest,
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Compare organization's performance against target percentile.
    
    Identifies priority gaps and provides improvement roadmap.
    """
    service = OrgBenchmarkService(db)
    
    comparison = await service.compare_with_target(
        org_id=org_id,
        target_percentile=request.target_percentile
    )
    
    return comparison


# Privacy Policy Endpoints

@router.post("/privacy-policy", response_model=PrivacyPolicyResponse)
async def create_or_update_privacy_policy(
    policy: PrivacyPolicyCreate,
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Create or update organization's privacy policy for federation.
    
    Controls:
    - What data can be shared
    - Anonymization requirements
    - Excluded domains
    - Benchmark participation
    """
    # Check if policy exists
    stmt = select(PrivacyPolicy).where(PrivacyPolicy.organization_id == org_id)
    result = await db.execute(stmt)
    existing_policy = result.scalar_one_or_none()
    
    if existing_policy:
        # Update existing policy
        for key, value in policy.model_dump().items():
            setattr(existing_policy, key, value)
        existing_policy.updated_at = datetime.utcnow()
        await db.commit()
        await db.refresh(existing_policy)
        return existing_policy
    else:
        # Create new policy
        new_policy = PrivacyPolicy(
            organization_id=org_id,
            **policy.model_dump()
        )
        db.add(new_policy)
        await db.commit()
        await db.refresh(new_policy)
        return new_policy


@router.get("/privacy-policy", response_model=PrivacyPolicyResponse)
async def get_privacy_policy(
    org_id: str = "default_org",
    db: AsyncSession = Depends(get_db)
):
    """
    Retrieve organization's current privacy policy.
    """
    stmt = select(PrivacyPolicy).where(PrivacyPolicy.organization_id == org_id)
    result = await db.execute(stmt)
    policy = result.scalar_one_or_none()
    
    if not policy:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Privacy policy not found. Create one first."
        )
    
    return policy


# Differential Privacy Endpoints

@router.post("/privacy/add-noise", response_model=AddNoiseResponse)
async def add_differential_privacy_noise(request: AddNoiseRequest):
    """
    Add differential privacy noise to data.
    
    Implements Laplace mechanism for ε-differential privacy.
    """
    dp_service = DifferentialPrivacyService(epsilon=request.epsilon or 1.0)
    
    noisy_data = dp_service.add_laplace_noise(
        data=request.data,
        sensitivity=request.sensitivity,
        epsilon=request.epsilon
    )
    
    epsilon_used = request.epsilon or 1.0
    noise_scale = request.sensitivity / epsilon_used
    
    return AddNoiseResponse(
        noisy_data=noisy_data,
        epsilon_used=epsilon_used,
        noise_scale=noise_scale
    )


@router.post("/privacy/private-aggregate", response_model=PrivateAggregateResponse)
async def compute_private_aggregate(
    request: PrivateAggregateRequest,
    org_data: dict,  # In production: fetch from secure source
):
    """
    Compute aggregate statistic with differential privacy.
    
    Supported aggregations: mean, median, sum, count, variance
    """
    dp_service = DifferentialPrivacyService(epsilon=request.epsilon or 1.0)
    
    result = dp_service.aggregate_with_privacy(
        org_data_dict=org_data,
        aggregation_type=request.aggregation_type,
        epsilon=request.epsilon
    )
    
    return PrivateAggregateResponse(
        result=result,
        aggregation_type=request.aggregation_type,
        epsilon_used=request.epsilon or 1.0,
        org_count=len(org_data)
    )


@router.post("/privacy/reidentification-risk", response_model=ReidentificationRiskResponse)
async def estimate_reidentification_risk(request: ReidentificationRiskRequest):
    """
    Estimate risk of re-identifying an organization from aggregated data.
    
    Factors:
    - Number of contributing orgs (more = lower risk)
    - Data dimensionality (higher = higher risk)
    - Privacy budget epsilon (larger = higher risk)
    """
    dp_service = DifferentialPrivacyService(epsilon=request.epsilon or 1.0)
    
    risk_score = dp_service.estimate_reidentification_risk(
        org_count=request.org_count,
        data_dimension=request.data_dimension,
        epsilon=request.epsilon
    )
    
    # Classify risk level
    if risk_score < 0.1:
        risk_level = "low"
        recommendation = "Privacy risk is acceptable. Proceed with data sharing."
    elif risk_score < 0.3:
        risk_level = "medium"
        recommendation = "Moderate risk. Consider increasing org count or reducing epsilon."
    else:
        risk_level = "high"
        recommendation = "High re-identification risk. Increase k-anonymity or apply stricter privacy."
    
    return ReidentificationRiskResponse(
        risk_score=risk_score,
        org_count=request.org_count,
        data_dimension=request.data_dimension,
        epsilon=request.epsilon or 1.0,
        risk_level=risk_level,
        recommendation=recommendation
    )


# Federation Health Endpoint

@router.get("/health", response_model=FederationHealthResponse)
async def get_federation_health(db: AsyncSession = Depends(get_db)):
    """
    Get overall federation health metrics.
    
    Shows participation rates, knowledge quality, and privacy compliance.
    """
    from app.models.federated_knowledge import FederatedKnowledgeSummary
    
    # Count total federated knowledge
    stmt_total_knowledge = select(func.count()).select_from(FederatedKnowledgeSummary)
    result = await db.execute(stmt_total_knowledge)
    total_knowledge = result.scalar()
    
    # Count by type
    stmt_by_type = select(
        FederatedKnowledgeSummary.summary_type,
        func.count()
    ).group_by(FederatedKnowledgeSummary.summary_type)
    result = await db.execute(stmt_by_type)
    by_type_counts = dict(result.all())
    
    # Average quality score
    stmt_avg_quality = select(func.avg(FederatedKnowledgeSummary.quality_score))
    result = await db.execute(stmt_avg_quality)
    avg_quality = result.scalar() or 0.0
    
    # Count organizations with privacy policies
    stmt_policies = select(func.count()).select_from(PrivacyPolicy)
    result = await db.execute(stmt_policies)
    participating_orgs = result.scalar()
    
    # Simulated total orgs (in production: query from organizations table)
    total_orgs = 100
    
    return FederationHealthResponse(
        total_organizations=total_orgs,
        participating_orgs=participating_orgs,
        participation_rate=participating_orgs / total_orgs if total_orgs > 0 else 0.0,
        total_federated_knowledge=total_knowledge,
        by_type=by_type_counts,
        average_quality_score=avg_quality,
        total_benchmarks_computed=0,  # Would query OrgBenchmark table
        privacy_compliance_rate=1.0  # All participating orgs have policies
    )
