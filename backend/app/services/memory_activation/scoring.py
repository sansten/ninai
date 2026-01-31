"""Memory Activation Scoring Library

This module implements the core activation scoring algorithm with 8 components:
1. Relevance (Rel) - Vector similarity to query
2. Recency (Rec) - Exponential decay based on time since access
3. Frequency (Freq) - Saturating curve of access count
4. Importance (Imp) - User-provided importance with feedback delta
5. Confidence (Conf) - Confidence adjusted for contradictions
6. Context Gate (Ctx) - Scope, episode, goal affinity
7. Provenance (Prov) - Evidence link density
8. Risk (Risk) - Risk classification factor

All components normalized to [0, 1].
Final activation computed via sigmoid logit with weights.
Optional neighbor boost from co-activation edges.
"""

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from uuid import UUID

from pydantic import BaseModel, Field


class ActivationScorerConfig(BaseModel):
    """Configuration for activation scorer weights and decay parameters."""

    # Decay parameters
    lambda_recency: float = Field(default=0.1, description="Recency decay rate")
    lambda_frequency: float = Field(default=0.5, description="Frequency saturation rate")
    lambda_importance: float = Field(default=0.05, description="Importance decay rate")
    lambda_edge: float = Field(default=0.2, description="Co-activation edge decay")

    # Contradiction penalty
    rho_contradiction: float = Field(default=0.5, description="Contradiction penalty factor")

    # Component weights (sum should be ~1.0, but flexible)
    w_rel: float = Field(default=0.25, description="Relevance weight")
    w_rec: float = Field(default=0.15, description="Recency weight")
    w_freq: float = Field(default=0.10, description="Frequency weight")
    w_imp: float = Field(default=0.20, description="Importance weight")
    w_conf: float = Field(default=0.15, description="Confidence weight")
    w_ctx: float = Field(default=0.10, description="Context weight")
    w_prov: float = Field(default=0.03, description="Provenance weight")
    w_risk: float = Field(default=0.02, description="Risk weight (usually negative)")

    bias: float = Field(default=0.0, description="Logit bias term")

    # Neighbor boost
    eta_neighbor_boost: float = Field(default=0.1, description="Neighbor boost scaling")
    max_neighbor_activation: float = Field(default=1.0, description="Max neighbor activation contribution")

    class Config:
        frozen = False


@dataclass
class ActivationComponents:
    """Container for all 8 activation components per memory."""

    rel: float  # Relevance
    rec: float  # Recency
    freq: float  # Frequency
    imp: float  # Importance
    conf: float  # Confidence
    ctx: float  # Context gate
    prov: float  # Provenance
    risk: float  # Risk

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "rel": self.rel,
            "rec": self.rec,
            "freq": self.freq,
            "imp": self.imp,
            "conf": self.conf,
            "ctx": self.ctx,
            "prov": self.prov,
            "risk": self.risk,
        }

    def validate(self) -> bool:
        """Validate all components in [0, 1]."""
        for value in [self.rel, self.rec, self.freq, self.imp, self.conf, self.ctx, self.prov, self.risk]:
            if not (0.0 <= value <= 1.0):
                return False
        return True


class ActivationScorer:
    """Core activation scoring engine.

    Computes activation score for a memory given:
    - Query context (vector similarity)
    - Memory metadata (importance, confidence, risk)
    - Access history (access_count, last_accessed)
    - Relationships (evidence links, contradictions)
    - Episode/scope/goal context

    All component computation is synchronous and fast (<10ms).
    Async updates to access counts and edges happen separately.
    """

    def __init__(self, config: Optional[ActivationScorerConfig] = None):
        """Initialize scorer with configuration.

        Args:
            config: Activation scorer configuration. Defaults to ActivationScorerConfig().
        """
        self.config = config or ActivationScorerConfig()

    def compute_relevance(self, similarity: float) -> float:
        """Compute relevance component from vector similarity.

        Args:
            similarity: Vector similarity in [0, 1]

        Returns:
            Relevance score in [0, 1]
        """
        # Similarity is already normalized by Qdrant
        return max(0.0, min(1.0, similarity))

    def compute_recency(
        self,
        last_accessed_at: Optional[datetime],
        current_time: Optional[datetime] = None,
    ) -> float:
        """Compute recency component via exponential decay.

        Recent accesses get high scores. Score decays exponentially.

        Args:
            last_accessed_at: Most recent access timestamp
            current_time: Current time (defaults to now)

        Returns:
            Recency score in [0, 1]
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        if last_accessed_at is None:
            # Never accessed: minimum recency
            return 0.01

        # Ensure timezone awareness
        if last_accessed_at.tzinfo is None:
            last_accessed_at = last_accessed_at.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # Time since last access in seconds
        delta_seconds = max(0, (current_time - last_accessed_at).total_seconds())

        # Exponential decay: exp(-λ * Δt)
        recency = math.exp(-self.config.lambda_recency * delta_seconds / 86400.0)  # Normalize to days

        return max(0.0, min(1.0, recency))

    def compute_frequency(self, access_count: int) -> float:
        """Compute frequency component from access count.

        More accesses → higher score, with saturation.

        Args:
            access_count: Number of times accessed

        Returns:
            Frequency score in [0, 1]
        """
        # Saturating exponential: 1 - exp(-λ * count)
        frequency = 1.0 - math.exp(-self.config.lambda_frequency * access_count)

        return max(0.0, min(1.0, frequency))

    def compute_importance(
        self,
        base_importance: float,
        feedback_delta: float = 0.0,
        age_days: float = 0.0,
    ) -> float:
        """Compute importance component with optional feedback and age decay.

        Args:
            base_importance: Base importance (0-1)
            feedback_delta: User feedback adjustment (-1 to +1)
            age_days: How old the memory is (days)

        Returns:
            Importance score in [0, 1]
        """
        # Apply feedback delta
        adjusted = base_importance + (feedback_delta * 0.2)  # Feedback limited to ±0.2

        # Apply age decay
        decay = math.exp(-self.config.lambda_importance * age_days)

        importance = adjusted * decay

        return max(0.0, min(1.0, importance))

    def compute_confidence(self, confidence: float, contradicted: bool = False) -> float:
        """Compute confidence component.

        Reduces confidence if memory is marked as contradicted.

        Args:
            confidence: Base confidence (0-1)
            contradicted: Whether memory is contradicted by evidence

        Returns:
            Adjusted confidence in [0, 1]
        """
        if contradicted:
            # Reduce by factor ρ
            confidence = confidence * (1.0 - self.config.rho_contradiction)

        return max(0.0, min(1.0, confidence))

    def compute_context_gate(
        self,
        scope_match: float = 0.5,
        episode_match: float = 0.5,
        goal_match: float = 0.5,
    ) -> float:
        """Compute context gate component.

        Combines scope, episode, and goal affinity.

        Args:
            scope_match: How well scope matches (0-1)
            episode_match: How well episode matches (0-1)
            goal_match: How well goal matches (0-1)

        Returns:
            Context gate score in [0, 1]
        """
        # Simple average of affinity scores
        context = (scope_match + episode_match + goal_match) / 3.0

        return max(0.0, min(1.0, context))

    def compute_provenance(self, evidence_link_count: int = 0) -> float:
        """Compute provenance component from evidence links.

        More evidence links → higher provenance.

        Args:
            evidence_link_count: Number of supporting evidence links

        Returns:
            Provenance score in [0, 1]
        """
        # Saturating: 1 - exp(-λ * count)
        provenance = 1.0 - math.exp(-self.config.lambda_edge * evidence_link_count)

        return max(0.0, min(1.0, provenance))

    def compute_risk(self, risk_factor: float) -> float:
        """Compute risk component.

        Higher risk → lower activation.

        Args:
            risk_factor: Risk classification (0-1, where 1 is highest risk)

        Returns:
            Risk adjustment in [0, 1] (inverted: 1-risk)
        """
        # Invert: high risk → low score
        risk = 1.0 - max(0.0, min(1.0, risk_factor))

        return risk

    def compute_activation(
        self,
        components: ActivationComponents,
        neighbor_activation: Optional[float] = None,
    ) -> float:
        """Compute final activation score from components.

        Uses logit combination with weights and neighbor boost.

        Args:
            components: All 8 activation components
            neighbor_activation: Max activation of co-activated neighbors (optional)

        Returns:
            Final activation score in [0, 1]
        """
        # Validate components
        if not components.validate():
            raise ValueError("All components must be in [0, 1]")

        # Logit: z = Σ(w_i * component_i) + bias
        z = (
            self.config.w_rel * components.rel
            + self.config.w_rec * components.rec
            + self.config.w_freq * components.freq
            + self.config.w_imp * components.imp
            + self.config.w_conf * components.conf
            + self.config.w_ctx * components.ctx
            + self.config.w_prov * components.prov
            - self.config.w_risk * components.risk  # Risk reduces activation
            + self.config.bias
        )

        # Sigmoid activation
        activation = 1.0 / (1.0 + math.exp(-z))

        # Optional neighbor boost
        if neighbor_activation is not None:
            boost = self.config.eta_neighbor_boost * min(
                neighbor_activation, self.config.max_neighbor_activation
            )
            activation = min(1.0, activation + boost)

        return max(0.0, min(1.0, activation))

    def score_memory(
        self,
        # Query context
        similarity: float,
        # Memory metadata
        base_importance: float,
        confidence: float,
        contradicted: bool = False,
        risk_factor: float = 0.0,
        # Access history
        access_count: int = 0,
        last_accessed_at: Optional[datetime] = None,
        # Relationships
        evidence_link_count: int = 0,
        # Context
        scope_match: float = 0.5,
        episode_match: float = 0.5,
        goal_match: float = 0.5,
        # Neighbor info
        neighbor_activation: Optional[float] = None,
        # Timing
        current_time: Optional[datetime] = None,
        age_days: float = 0.0,
    ) -> tuple[float, ActivationComponents]:
        """Score a single memory comprehensively.

        This is the main entry point. Computes all 8 components and final activation.

        Args:
            similarity: Vector similarity to query
            base_importance: User-provided importance
            confidence: Confidence in memory
            contradicted: Whether contradicted
            risk_factor: Risk classification
            access_count: Number of times accessed
            last_accessed_at: Most recent access time
            evidence_link_count: Number of evidence links
            scope_match: Scope affinity (0-1)
            episode_match: Episode affinity (0-1)
            goal_match: Goal affinity (0-1)
            neighbor_activation: Max neighbor activation (optional)
            current_time: Current time (defaults to now)
            age_days: Memory age in days

        Returns:
            Tuple of (activation_score, components_breakdown)
        """
        # Compute all 8 components
        rel = self.compute_relevance(similarity)
        rec = self.compute_recency(last_accessed_at, current_time)
        freq = self.compute_frequency(access_count)
        imp = self.compute_importance(base_importance, feedback_delta=0.0, age_days=age_days)
        conf = self.compute_confidence(confidence, contradicted)
        ctx = self.compute_context_gate(scope_match, episode_match, goal_match)
        prov = self.compute_provenance(evidence_link_count)
        risk = self.compute_risk(risk_factor)

        # Bundle components
        components = ActivationComponents(
            rel=rel,
            rec=rec,
            freq=freq,
            imp=imp,
            conf=conf,
            ctx=ctx,
            prov=prov,
            risk=risk,
        )

        # Compute final activation
        activation = self.compute_activation(components, neighbor_activation)

        return activation, components


# Global singleton instance
_scorer_instance: Optional[ActivationScorer] = None


def get_activation_scorer(config: Optional[ActivationScorerConfig] = None) -> ActivationScorer:
    """Get or create global activation scorer instance.

    Args:
        config: Optional custom configuration

    Returns:
        ActivationScorer instance
    """
    global _scorer_instance
    if _scorer_instance is None:
        _scorer_instance = ActivationScorer(config)
    return _scorer_instance
