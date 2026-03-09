"""Agent registry.

Central place to map agent names to implementations.
"""

from __future__ import annotations

from typing import Optional

from app.agents.base import BaseAgent
from app.agents.classification_agent import ClassificationAgent
from app.agents.feedback_learning_agent import FeedbackLearningAgent
from app.agents.graph_linking_agent import GraphLinkingAgent
from app.agents.logseq_export_agent import LogseqExportAgent
from app.agents.metadata_extraction_agent import MetadataExtractionAgent
from app.agents.pattern_detection_agent import PatternDetectionAgent
from app.agents.promotion_agent import PromotionAgent
from app.agents.context_amplifier_agent import ContextAmplifierAgent
from app.agents.entity_resolution_agent import EntityResolutionAgent
from app.agents.semantic_normalization_agent import SemanticNormalizationAgent
from app.agents.silo_propagation_agent import SiloPropagationAgent
from app.agents.org_attention_agent import OrgAttentionAgent
from app.agents.causal_reasoning_agent import CausalReasoningAgent
from app.agents.conflict_detection_agent import ConflictDetectionAgent
from app.agents.adaptive_conflict_resolution_agent import AdaptiveConflictResolutionAgent
from app.agents.memory_decay_agent import MemoryDecayAgent
from app.agents.memory_consolidation_agent import MemoryConsolidationAgent
from app.agents.proactive_memory_push_agent import ProactiveMemoryPushAgent
from app.agents.temporal_reasoning_agent import TemporalReasoningAgent
from app.agents.episodic_grouping_agent import EpisodicGroupingAgent
from app.agents.predictive_monitor_agent import PredictiveMonitorAgent
from app.agents.topic_modeling_agent import TopicModelingAgent
from app.agents.world_model_agent import WorldModelAgent
from app.agents.credibility_agent import CredibilityAgent
from app.agents.playbook_agent import PlaybookAgent
from app.agents.goal_decomposition_agent import GoalDecompositionAgent
from app.agents.uncertainty_reporting_agent import UncertaintyReportingAgent
from app.agents.narrative_synthesis_agent import NarrativeSynthesisAgent
from app.agents.feedback_integration_agent import FeedbackIntegrationAgent
from app.agents.anomaly_detection_agent import AnomalyDetectionAgent


def get_agent(agent_name: str) -> Optional[BaseAgent]:
    name = (agent_name or "").strip().lower()

    if name in {"classification", "classificationagent"}:
        return ClassificationAgent()

    if name in {"metadata", "metadataextraction", "metadataextractionagent"}:
        return MetadataExtractionAgent()

    if name in {"topics", "topic", "topicmodeling", "topicmodelingagent"}:
        return TopicModelingAgent()

    if name in {"patterns", "pattern", "patterndetection", "patterndetectionagent"}:
        return PatternDetectionAgent()

    if name in {"promotion", "promotionagent"}:
        return PromotionAgent()

    if name in {"graph", "graphlinking", "graphlinkingagent"}:
        return GraphLinkingAgent()

    if name in {"logseq", "logseq_export", "logseqexport", "logseqexportagent"}:
        return LogseqExportAgent()

    if name in {"feedback", "feedbacklearning", "feedbacklearningagent"}:
        return FeedbackLearningAgent()

    if name in {"semantic_normalization", "semanticnormalization", "semanticnormalizationagent"}:
        return SemanticNormalizationAgent()

    if name in {"entity_resolution", "entityresolution", "entityresolutionagent"}:
        return EntityResolutionAgent()

    if name in {"context_amplifier", "contextamplifier", "contextamplifieragent"}:
        return ContextAmplifierAgent()

    if name in {"silo_propagation", "silopropagation", "silopropagationagent"}:
        return SiloPropagationAgent()

    if name in {"org_attention", "orgattention", "orgattentionagent"}:
        return OrgAttentionAgent()

    if name in {"proactive_push", "proactivepush", "proactivememorpush",
                "proactivememory", "proactivememorypush", "proactivememorypushagent"}:
        return ProactiveMemoryPushAgent()

    if name in {"world_model", "worldmodel", "worldmodelagent"}:
        return WorldModelAgent()

    if name in {"predictive_monitor", "predictivemonitor", "predictivemonitoragent"}:
        return PredictiveMonitorAgent()

    if name in {"causal_reasoning", "causalreasoning", "causalreasoningagent"}:
        return CausalReasoningAgent()

    if name in {"conflict_detection", "conflictdetection", "conflictdetectionagent"}:
        return ConflictDetectionAgent()

    if name in {"adaptive_conflict_resolution", "adaptiveconflictresolution",
                "adaptiveconflictresolutionagent"}:
        return AdaptiveConflictResolutionAgent()

    if name in {"memory_decay", "memorydecay", "memorydecayagent"}:
        return MemoryDecayAgent()

    if name in {"memory_consolidation", "memoryconsolidation", "memoryconsolidationagent"}:
        return MemoryConsolidationAgent()

    if name in {"temporal_reasoning", "temporalreasoning", "temporalreasoningagent"}:
        return TemporalReasoningAgent()

    if name in {"episodic_grouping", "episodicgrouping", "episodicgroupingagent"}:
        return EpisodicGroupingAgent()

    if name in {"credibility", "credibility_agent", "credibilityagent"}:
        return CredibilityAgent()

    if name in {"playbook", "playbook_agent", "playbookagent"}:
        return PlaybookAgent()

    if name in {"goal_decomposition", "goaldecomposition", "goaldecompositionagent"}:
        return GoalDecompositionAgent()

    if name in {"uncertainty_reporting", "uncertaintyreporting", "uncertaintyreportingagent",
                "uncertainty_agent"}:
        return UncertaintyReportingAgent()

    if name in {"narrative_synthesis", "narrativesynthesis", "narrativesynthesisagent",
                "narrative_agent"}:
        return NarrativeSynthesisAgent()

    if name in {"feedback_integration", "feedbackintegration", "feedbackintegrationagent",
                "feedback_agent"}:
        return FeedbackIntegrationAgent()

    if name in {"anomaly_detection", "anomalydetection", "anomalydetectionagent",
                "anomaly_agent"}:
        return AnomalyDetectionAgent()

    return None
