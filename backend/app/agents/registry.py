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

    return None
