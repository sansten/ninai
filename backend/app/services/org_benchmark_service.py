"""
PR-10: Organization Benchmark Service

Compare organization performance against peer groups.
Provides percentile rankings and trend analysis.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from backend.app.models.federated_knowledge import OrgBenchmark, PrivacyPolicy
import statistics


class OrgBenchmarkService:
    """
    Compare organization against peer group.
    
    Provides benchmarking for:
    - Memory quality metrics
    - Response time performance
    - Customer satisfaction scores
    - Tool success rates
    - Knowledge coverage
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def compute_org_benchmarks(
        self,
        org_id: str,
        metrics: Optional[List[str]] = None
    ) -> List[OrgBenchmark]:
        """
        Compute key metrics for organization and compare against peers.
        
        Default metrics:
        - memory_quality: Overall memory system health
        - response_time: Average response time
        - customer_satisfaction: User satisfaction score
        - tool_success_rate: Success rate of tool executions
        - knowledge_coverage: Breadth of knowledge domains
        
        Args:
            org_id: Organization to benchmark
            metrics: Specific metrics to compute (None = all)
        
        Returns:
            List of OrgBenchmark objects with percentile rankings
        """
        # Check if org allows benchmark participation
        policy = await self._get_privacy_policy(org_id)
        if not policy or not policy.allow_benchmark_participation:
            return []
        
        if metrics is None:
            metrics = [
                "memory_quality",
                "response_time",
                "customer_satisfaction",
                "tool_success_rate",
                "knowledge_coverage"
            ]
        
        benchmarks = []
        for metric_type in metrics:
            benchmark = await self._compute_single_metric(org_id, metric_type)
            if benchmark:
                benchmarks.append(benchmark)
        
        return benchmarks
    
    async def _compute_single_metric(
        self,
        org_id: str,
        metric_type: str
    ) -> Optional[OrgBenchmark]:
        """
        Compute single metric and percentile rank.
        
        Args:
            org_id: Organization ID
            metric_type: Metric to compute
        
        Returns:
            OrgBenchmark with percentile ranking
        """
        # Get or create benchmark record
        stmt = select(OrgBenchmark).where(
            and_(
                OrgBenchmark.organization_id == org_id,
                OrgBenchmark.metric_type == metric_type
            )
        )
        result = await self.db.execute(stmt)
        benchmark = result.scalar_one_or_none()
        
        # Compute metric value (in production: fetch from actual data)
        value = await self._fetch_metric_value(org_id, metric_type)
        
        # Get peer values for comparison
        peer_values = await self._fetch_peer_metric_values(org_id, metric_type)
        
        # Compute percentile
        percentile = self._compute_percentile(value, peer_values)
        
        # Determine trend (compare to previous value)
        trend = "stable"
        if benchmark:
            if value > benchmark.value * 1.05:
                trend = "improving"
            elif value < benchmark.value * 0.95:
                trend = "declining"
        
        if benchmark:
            # Update existing benchmark
            benchmark.value = value
            benchmark.percentile = percentile
            benchmark.peer_count = len(peer_values)
            benchmark.trend = trend
            benchmark.last_computed_at = datetime.utcnow()
        else:
            # Create new benchmark
            benchmark = OrgBenchmark(
                organization_id=org_id,
                metric_type=metric_type,
                value=value,
                percentile=percentile,
                peer_count=len(peer_values),
                trend=trend
            )
            self.db.add(benchmark)
        
        await self.db.commit()
        await self.db.refresh(benchmark)
        return benchmark
    
    async def get_benchmark_report(self, org_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive benchmarking report.
        
        Args:
            org_id: Organization ID
        
        Returns:
            Dict with metrics, comparisons, trends, and recommendations
        """
        # Fetch all benchmarks for this org
        stmt = select(OrgBenchmark).where(
            OrgBenchmark.organization_id == org_id
        )
        result = await self.db.execute(stmt)
        benchmarks = result.scalars().all()
        
        if not benchmarks:
            return {
                "status": "no_data",
                "message": "Enable benchmark participation to see comparisons"
            }
        
        # Build report
        metrics = {}
        improvements = []
        recommendations = []
        
        for benchmark in benchmarks:
            metrics[benchmark.metric_type] = {
                "value": benchmark.value,
                "percentile": benchmark.percentile,
                "peer_count": benchmark.peer_count,
                "trend": benchmark.trend,
                "ranking": self._percentile_to_ranking(benchmark.percentile)
            }
            
            # Identify improvements
            if benchmark.trend == "improving":
                improvements.append(
                    f"{benchmark.metric_type} improved 
to {benchmark.percentile:.0%} percentile"
                )
            
            # Generate recommendations for low performers
            if benchmark.percentile < 0.5:
                recommendations.append(
                    self._generate_recommendation(benchmark.metric_type, benchmark.percentile)
                )
        
        # Overall health score
        overall_score = statistics.mean([b.percentile for b in benchmarks])
        
        return {
            "organization_id": org_id,
            "report_date": datetime.utcnow().isoformat(),
            "overall_health_percentile": overall_score,
            "overall_ranking": self._percentile_to_ranking(overall_score),
            "metrics": metrics,
            "improvements": improvements,
            "recommendations": recommendations,
            "peer_comparison_count": max([b.peer_count for b in benchmarks] if benchmarks else [0])
        }
    
    async def get_benchmark_history(
        self,
        org_id: str,
        metric_type: str,
        lookback_days: int = 90
    ) -> List[Dict[str, Any]]:
        """
        Get historical benchmark data for trend analysis.
        
        Args:
            org_id: Organization ID
            metric_type: Metric to retrieve history for
            lookback_days: How many days back to look
        
        Returns:
            List of historical benchmark snapshots
        """
        # In production: query historical snapshots table
        # For now: return current benchmark
        stmt = select(OrgBenchmark).where(
            and_(
                OrgBenchmark.organization_id == org_id,
                OrgBenchmark.metric_type == metric_type
            )
        )
        result = await self.db.execute(stmt)
        benchmark = result.scalar_one_or_none()
        
        if not benchmark:
            return []
        
        # Return as single point (would be multiple in production)
        return [{
            "date": benchmark.last_computed_at.isoformat(),
            "value": benchmark.value,
            "percentile": benchmark.percentile,
            "trend": benchmark.trend
        }]
    
    async def compare_with_target(
        self,
        org_id: str,
        target_percentile: float = 0.75
    ) -> Dict[str, Any]:
        """
        Compare organization's performance against target percentile.
        
        Identify gaps and priority areas for improvement.
        
        Args:
            org_id: Organization ID
            target_percentile: Desired percentile (default: 75th)
        
        Returns:
            Gap analysis with priorities
        """
        stmt = select(OrgBenchmark).where(
            OrgBenchmark.organization_id == org_id
        )
        result = await self.db.execute(stmt)
        benchmarks = result.scalars().all()
        
        gaps = []
        for benchmark in benchmarks:
            if benchmark.percentile < target_percentile:
                gap = target_percentile - benchmark.percentile
                gaps.append({
                    "metric": benchmark.metric_type,
                    "current_percentile": benchmark.percentile,
                    "target_percentile": target_percentile,
                    "gap": gap,
                    "priority": "high" if gap > 0.3 else "medium" if gap > 0.15 else "low",
                    "recommendation": self._generate_recommendation(benchmark.metric_type, benchmark.percentile)
                })
        
        # Sort by gap size
        gaps.sort(key=lambda x: x["gap"], reverse=True)
        
        return {
            "organization_id": org_id,
            "target_percentile": target_percentile,
            "gaps": gaps,
            "met_target_count": len([b for b in benchmarks if b.percentile >= target_percentile]),
            "total_metrics": len(benchmarks)
        }
    
    # Private helper methods
    
    async def _get_privacy_policy(self, org_id: str) -> Optional[PrivacyPolicy]:
        """Retrieve organization's privacy policy."""
        stmt = select(PrivacyPolicy).where(PrivacyPolicy.organization_id == org_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def _fetch_metric_value(self, org_id: str, metric_type: str) -> float:
        """
        Fetch actual metric value from organization's data.
        
        In production: query memory_metadata, playbooks, etc.
        For now: return simulated values
        """
        # Simulated values for demonstration
        simulations = {
            "memory_quality": 0.82,
            "response_time": 1.5,  # seconds
            "customer_satisfaction": 4.2,  # 0-5 scale
            "tool_success_rate": 0.88,
            "knowledge_coverage": 0.75
        }
        return simulations.get(metric_type, 0.5)
    
    async def _fetch_peer_metric_values(self, org_id: str, metric_type: str) -> List[float]:
        """
        Fetch metric values from peer organizations.
        
        Peers determined by: industry, size, plan tier
        
        In production: aggregate from other orgs with privacy policies allowing benchmark_participation
        For now: return simulated peer data
        """
        # Simulated peer distribution
        import random
        random.seed(hash(org_id + metric_type))  # Deterministic for demo
        
        # Generate ~50 peer values with normal distribution
        peer_count = 50
        mean = 0.7
        std_dev = 0.15
        
        return [max(0.0, min(1.0, random.gauss(mean, std_dev))) for _ in range(peer_count)]
    
    def _compute_percentile(self, value: float, peer_values: List[float]) -> float:
        """
        Compute percentile rank of value within peer distribution.
        
        Args:
            value: Organization's value
            peer_values: Peer organization values
        
        Returns:
            Percentile (0-1) where org ranks
        """
        if not peer_values:
            return 0.5
        
        # Count how many peers are below this value
        below_count = sum(1 for pv in peer_values if pv < value)
        return below_count / len(peer_values)
    
    def _percentile_to_ranking(self, percentile: float) -> str:
        """Convert percentile to human-readable ranking."""
        if percentile >= 0.9:
            return "top 10%"
        elif percentile >= 0.75:
            return "top 25%"
        elif percentile >= 0.5:
            return "above average"
        elif percentile >= 0.25:
            return "below average"
        else:
            return "bottom 25%"
    
    def _generate_recommendation(self, metric_type: str, percentile: float) -> str:
        """Generate actionable recommendation for low-performing metrics."""
        recommendations = {
            "memory_quality": "Consider running memory consolidation more frequently and validating fact quality",
            "response_time": "Optimize retrieval budget and consider caching frequently accessed memories",
            "customer_satisfaction": "Review emotional trajectories and enable empathetic response generation",
            "tool_success_rate": "Analyze tool failure patterns and update self-model capabilities",
            "knowledge_coverage": "Expand memory capture across more domains and enable autonomous learning goals"
        }
        
        base_rec = recommendations.get(metric_type, "Review this metric and compare with peer best practices")
        
        if percentile < 0.25:
            return f"PRIORITY: {base_rec}"
        else:
            return base_rec
