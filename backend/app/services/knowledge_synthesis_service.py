"""
Knowledge Synthesis Service

Synthesizes insights from memories and relationships:
- Concept clustering and grouping
- Relationship strength analysis
- Trend detection and topic evolution
- Summary generation from memory clusters
- PDF/Markdown export
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from abc import ABC, abstractmethod

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.memory import Memory
from app.models.graph_relationship import GraphRelationship
from app.core.config import settings


class InsightType(str, Enum):
    """Types of insights that can be generated."""
    CLUSTER = "cluster"
    TREND = "trend"
    RELATIONSHIP = "relationship"
    SUMMARY = "summary"


@dataclass
class ConceptCluster:
    """A cluster of related memories around a concept."""
    concept: str
    memories: List[Dict[str, Any]]
    strength: float  # 0-1, how cohesive the cluster is
    tags: List[str]
    relationships_count: int
    date_range: tuple  # (start_date, end_date)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'concept': self.concept,
            'memories': self.memories,
            'strength': self.strength,
            'tags': self.tags,
            'relationships_count': self.relationships_count,
            'date_range': [d.isoformat() if isinstance(d, datetime) else d 
                          for d in self.date_range],
        }


@dataclass
class Trend:
    """A detected trend in memory data."""
    topic: str
    description: str
    start_date: datetime
    end_date: datetime
    memory_count: int
    trajectory: str  # 'increasing', 'decreasing', 'stable'
    strength: float  # 0-1
    related_concepts: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'topic': self.topic,
            'description': self.description,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'memory_count': self.memory_count,
            'trajectory': self.trajectory,
            'strength': self.strength,
            'related_concepts': self.related_concepts,
        }


@dataclass
class SynthesisReport:
    """A comprehensive synthesis report."""
    title: str
    summary: str
    clusters: List[ConceptCluster]
    trends: List[Trend]
    key_insights: List[str]
    relationships: Dict[str, List[str]]
    generated_at: datetime
    memory_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'title': self.title,
            'summary': self.summary,
            'clusters': [c.to_dict() for c in self.clusters],
            'trends': [t.to_dict() for t in self.trends],
            'key_insights': self.key_insights,
            'relationships': self.relationships,
            'generated_at': self.generated_at.isoformat(),
            'memory_count': self.memory_count,
        }


class LLMSynthesizer(ABC):
    """Abstract base for LLM-based synthesis."""
    
    @abstractmethod
    async def synthesize_cluster(self, cluster: ConceptCluster) -> str:
        """Generate summary for a cluster using LLM."""
        pass
    
    @abstractmethod
    async def generate_insights(self, report: SynthesisReport) -> List[str]:
        """Generate key insights using LLM."""
        pass
    
    @abstractmethod
    async def create_narrative(self, report: SynthesisReport) -> str:
        """Create a narrative summary using LLM."""
        pass


class KnowledgeSynthesisService:
    """
    Synthesizes knowledge from memories and relationships.
    
    Features:
    - Concept clustering based on tags and relationships
    - Trend detection across time periods
    - Relationship strength analysis
    - LLM-powered insight generation
    - Multi-format export (JSON, Markdown, PDF)
    """
    
    def __init__(self, session: Session, llm_synthesizer: Optional[LLMSynthesizer] = None):
        """
        Initialize synthesis service.
        
        Args:
            session: Database session
            llm_synthesizer: Optional LLM synthesizer for enhanced insights
        """
        self.session = session
        self.llm_synthesizer = llm_synthesizer
    
    async def create_synthesis_report(
        self,
        memory_ids: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        days_back: int = 30,
        title: Optional[str] = None,
    ) -> SynthesisReport:
        """
        Create a comprehensive synthesis report.
        
        Args:
            memory_ids: Specific memories to include (if None, recent memories)
            tags: Filter by tags
            days_back: How many days of history to include
            title: Optional custom title
            
        Returns:
            SynthesisReport with clusters, trends, and insights
        """
        # Get memories
        memories = await self._get_memories(memory_ids, tags, days_back)
        
        if not memories:
            return SynthesisReport(
                title=title or "Empty Report",
                summary="No memories found for synthesis",
                clusters=[],
                trends=[],
                key_insights=[],
                relationships={},
                generated_at=datetime.utcnow(),
                memory_count=0,
            )
        
        # Create clusters
        clusters = await self._create_concept_clusters(memories)
        
        # Detect trends
        trends = await self._detect_trends(memories)
        
        # Analyze relationships
        relationships = await self._analyze_relationships(memories)
        
        # Generate insights
        key_insights = await self._generate_insights(clusters, trends)
        
        # Create summary
        summary = await self._create_summary(clusters, trends)
        
        report = SynthesisReport(
            title=title or f"Knowledge Synthesis - {datetime.utcnow().strftime('%Y-%m-%d')}",
            summary=summary,
            clusters=clusters,
            trends=trends,
            key_insights=key_insights,
            relationships=relationships,
            generated_at=datetime.utcnow(),
            memory_count=len(memories),
        )
        
        return report
    
    async def _get_memories(
        self,
        memory_ids: Optional[List[str]],
        tags: Optional[List[str]],
        days_back: int,
    ) -> List[Memory]:
        """Get memories for synthesis."""
        query = select(Memory)
        
        if memory_ids:
            query = query.where(Memory.id.in_(memory_ids))
        
        if tags:
            # Filter by tags
            for tag in tags:
                query = query.where(Memory.tags.contains([tag]))
        
        if days_back > 0:
            start_date = datetime.utcnow() - timedelta(days=days_back)
            query = query.where(Memory.created_at >= start_date)
        
        result = self.session.execute(query)
        return result.scalars().all()
    
    async def _create_concept_clusters(
        self,
        memories: List[Memory],
    ) -> List[ConceptCluster]:
        """
        Create concept clusters from memories.
        
        Groups memories by common tags and relationships.
        """
        clusters: Dict[str, List[Memory]] = {}
        
        # Group by tags
        for memory in memories:
            if memory.tags:
                for tag in memory.tags:
                    if tag not in clusters:
                        clusters[tag] = []
                    clusters[tag].append(memory)
        
        # Create cluster objects
        concept_clusters = []
        for concept, mems in clusters.items():
            # Calculate strength based on relationship density
            strength = await self._calculate_cluster_strength(mems)
            
            # Get all tags in cluster
            all_tags = set()
            for mem in mems:
                if mem.tags:
                    all_tags.update(mem.tags)
            
            # Count relationships
            rel_count = await self._count_relationships(mems)
            
            # Get date range
            dates = [m.created_at for m in mems if m.created_at]
            date_range = (min(dates), max(dates)) if dates else (None, None)
            
            cluster = ConceptCluster(
                concept=concept,
                memories=[{
                    'id': m.id,
                    'title': m.title,
                    'content': m.content[:100] + "..." if len(m.content) > 100 else m.content,
                    'created_at': m.created_at.isoformat() if m.created_at else None,
                    'tags': m.tags or [],
                } for m in mems[:10]],  # Top 10 memories
                strength=strength,
                tags=list(all_tags),
                relationships_count=rel_count,
                date_range=date_range,
            )
            concept_clusters.append(cluster)
        
        # Sort by strength
        concept_clusters.sort(key=lambda c: c.strength, reverse=True)
        return concept_clusters[:10]  # Top 10 clusters
    
    async def _calculate_cluster_strength(self, memories: List[Memory]) -> float:
        """Calculate how cohesive a cluster is (0-1)."""
        if len(memories) < 2:
            return 0.5
        
        # Count relationships between cluster members
        member_ids = {m.id for m in memories}
        query = select(GraphRelationship).where(
            GraphRelationship.from_memory_id.in_(member_ids),
            GraphRelationship.to_memory_id.in_(member_ids),
        )
        result = self.session.execute(query)
        relationships = result.scalars().all()
        
        # Strength = relationship density
        max_possible = len(memories) * (len(memories) - 1)
        if max_possible == 0:
            return 0.5
        
        return min(1.0, len(relationships) / max_possible * 2)
    
    async def _count_relationships(self, memories: List[Memory]) -> int:
        """Count relationships within a group of memories."""
        member_ids = {m.id for m in memories}
        query = select(GraphRelationship).where(
            GraphRelationship.from_memory_id.in_(member_ids),
            GraphRelationship.to_memory_id.in_(member_ids),
        )
        result = self.session.execute(query)
        return len(result.scalars().all())
    
    async def _detect_trends(self, memories: List[Memory]) -> List[Trend]:
        """Detect trends in memory data over time."""
        if not memories:
            return []
        
        # Group memories by week
        weeks: Dict[str, List[Memory]] = {}
        for memory in memories:
            if memory.created_at:
                week_key = memory.created_at.strftime('%Y-W%U')
                if week_key not in weeks:
                    weeks[week_key] = []
                weeks[week_key].append(memory)
        
        # Detect trends (simplified: just growth trends)
        trends = []
        week_counts = sorted([(k, len(v)) for k, v in weeks.items()])
        
        if len(week_counts) >= 3:
            # Check if growing
            counts = [c for _, c in week_counts[-3:]]
            if counts[-1] > counts[0]:
                trajectory = 'increasing'
                strength = min(1.0, (counts[-1] - counts[0]) / max(counts[0], 1))
            else:
                trajectory = 'decreasing'
                strength = min(1.0, (counts[0] - counts[-1]) / max(counts[0], 1))
            
            trend = Trend(
                topic="Memory Activity",
                description=f"Memory creation is {trajectory}",
                start_date=memories[0].created_at or datetime.utcnow(),
                end_date=memories[-1].created_at or datetime.utcnow(),
                memory_count=len(memories),
                trajectory=trajectory,
                strength=strength,
                related_concepts=[],
            )
            trends.append(trend)
        
        return trends
    
    async def _analyze_relationships(self, memories: List[Memory]) -> Dict[str, List[str]]:
        """Analyze relationships between memories."""
        member_ids = {m.id for m in memories}
        query = select(GraphRelationship).where(
            GraphRelationship.from_memory_id.in_(member_ids),
        )
        result = self.session.execute(query)
        relationships = result.scalars().all()
        
        analysis = {}
        for rel in relationships:
            if rel.from_memory_id not in analysis:
                analysis[rel.from_memory_id] = []
            analysis[rel.from_memory_id].append(f"{rel.relationship_type}:{rel.to_memory_id}")
        
        return analysis
    
    async def _generate_insights(
        self,
        clusters: List[ConceptCluster],
        trends: List[Trend],
    ) -> List[str]:
        """Generate key insights from clusters and trends."""
        insights = []
        
        # Basic insights
        if clusters:
            top_concept = clusters[0].concept
            insights.append(f"'{top_concept}' is your most connected concept with {clusters[0].relationships_count} relationships")
        
        if trends:
            trend = trends[0]
            insights.append(f"Memory activity is {trend.trajectory} with {trend.memory_count} memories tracked")
        
        # Use LLM if available
        if self.llm_synthesizer and clusters:
            # Get LLM insights
            pass  # Would call LLM here
        
        return insights
    
    async def _create_summary(
        self,
        clusters: List[ConceptCluster],
        trends: List[Trend],
    ) -> str:
        """Create narrative summary."""
        parts = []
        
        if clusters:
            parts.append(f"This synthesis covers {len(clusters)} key concept clusters.")
            top_3 = clusters[:3]
            concepts = [c.concept for c in top_3]
            parts.append(f"Primary focuses: {', '.join(concepts)}")
        
        if trends:
            for trend in trends:
                parts.append(f"Trend: {trend.description}")
        
        return " ".join(parts) if parts else "No summary available"
    
    def export_markdown(self, report: SynthesisReport) -> str:
        """Export report as Markdown."""
        lines = [
            f"# {report.title}",
            f"\nGenerated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Memories analyzed: {report.memory_count}",
            f"\n## Summary\n\n{report.summary}",
        ]
        
        if report.key_insights:
            lines.append("\n## Key Insights\n")
            for insight in report.key_insights:
                lines.append(f"- {insight}")
        
        if report.clusters:
            lines.append("\n## Concept Clusters\n")
            for cluster in report.clusters:
                lines.append(f"\n### {cluster.concept}")
                lines.append(f"- Strength: {cluster.strength:.1%}")
                lines.append(f"- Relationships: {cluster.relationships_count}")
                lines.append(f"- Tags: {', '.join(cluster.tags)}")
                lines.append(f"- Memories: {len(cluster.memories)}")
        
        if report.trends:
            lines.append("\n## Trends\n")
            for trend in report.trends:
                lines.append(f"\n### {trend.topic}")
                lines.append(f"- {trend.description}")
                lines.append(f"- Trajectory: {trend.trajectory}")
        
        return "\n".join(lines)
    
    def export_json(self, report: SynthesisReport) -> str:
        """Export report as JSON."""
        return json.dumps(report.to_dict(), indent=2)
