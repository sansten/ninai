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
from app.agents.proactive_memory_push_agent import ProactiveMemoryPushAgent
from app.agents.topic_modeling_agent import TopicModelingAgent


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

    return None
