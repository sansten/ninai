"""Agent registry.

Central place to map agent names to implementations.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from typing import Optional

from app.agents.active_knowledge_seeker_agent import ActiveKnowledgeSeekerAgent
from app.agents.auto_research_agent import AutoResearchAgent
from app.agents.adaptive_enrichment_budget_agent import AdaptiveEnrichmentBudgetAgent
from app.agents.adaptive_persona_agent import AdaptivePersonaAgent
from app.agents.analogical_reasoning_agent import AnalogicalReasoningAgent
from app.agents.anomaly_detection_agent import AnomalyDetectionAgent
from app.agents.base import BaseAgent
from app.agents.classification_agent import ClassificationAgent
from app.agents.compositional_generalization_agent import CompositionalGeneralizationAgent
from app.agents.context_amplifier_agent import ContextAmplifierAgent
from app.agents.cross_modal_reasoning_agent import CrossModalReasoningAgent
from app.agents.debate_ensemble_agent import DebateEnsembleAgent
from app.agents.emotional_affective_memory_agent import EmotionalAffectiveMemoryAgent
from app.agents.entity_resolution_agent import EntityResolutionAgent
from app.agents.episodic_grouping_agent import EpisodicGroupingAgent
from app.agents.error_remediation_agent import ErrorRemediationAgent
from app.agents.federated_memory_agent import FederatedMemoryAgent
from app.agents.hierarchical_goal_planner_agent import HierarchicalGoalPlannerAgent
from app.agents.logseq_export_agent import LogseqExportAgent
from app.agents.memory_sleep_agent import MemorySleepAgent
from app.agents.memory_tier_manager_agent import MemoryTierManagerAgent
from app.agents.metadata_extraction_agent import MetadataExtractionAgent
from app.agents.multimodal_deep_memory_agent import MultimodalDeepMemoryAgent
from app.agents.multi_turn_goal_tracking_agent import MultiTurnGoalTrackingAgent
from app.agents.narrative_compression_agent import NarrativeCompressionAgent
from app.agents.narrative_synthesis_agent import NarrativeSynthesisAgent
from app.agents.orchestration_bus_agent import OrchestrationBusAgent
from app.agents.org_attention_agent import OrgAttentionAgent
from app.agents.pattern_detection_agent import PatternDetectionAgent
from app.agents.playbook_auto_synthesis_agent import PlaybookAutoSynthesisAgent
from app.agents.playbook_execution_tracker_agent import PlaybookExecutionTrackerAgent
from app.agents.proactive_memory_push_agent import ProactiveMemoryPushAgent
from app.agents.prospective_memory_agent import ProspectiveMemoryAgent
from app.agents.query_intelligence_agent import QueryIntelligenceAgent
from app.agents.semantic_normalization_agent import SemanticNormalizationAgent
from app.agents.semantic_role_inference_agent import SemanticRoleInferenceAgent
from app.agents.silo_propagation_agent import SiloPropagationAgent
from app.agents.social_memory_agent import SocialMemoryAgent
from app.agents.temporal_pattern_miner_agent import TemporalPatternMinerAgent
from app.agents.temporal_reasoning_agent import TemporalReasoningAgent
from app.agents.theory_of_mind_agent import TheoryOfMindAgent
from app.agents.topic_modeling_agent import TopicModelingAgent


def _load_optional_agent(module_path: str, class_name: str) -> type[BaseAgent] | None:
    try:
        if find_spec(module_path) is None:
            return None
    except ModuleNotFoundError:
        return None

    try:
        module = import_module(module_path)
    except ModuleNotFoundError:
        return None

    agent_class = getattr(module, class_name, None)
    if isinstance(agent_class, type) and issubclass(agent_class, BaseAgent):
        return agent_class
    return None


def _instantiate_agent(agent_class: type[BaseAgent] | None) -> Optional[BaseAgent]:
    return agent_class() if agent_class is not None else None


AdaptiveConflictResolutionAgent = _load_optional_agent(
    "ninai_enterprise.agents.adaptive_conflict_resolution_agent",
    "AdaptiveConflictResolutionAgent",
)
AuditTrailAgent = _load_optional_agent("ninai_enterprise.agents.audit_trail_agent", "AuditTrailAgent")
AutonomousActionAgent = _load_optional_agent(
    "ninai_enterprise.agents.autonomous_action_agent",
    "AutonomousActionAgent",
)
AutonomousGoalGenerationAgent = _load_optional_agent(
    "ninai_enterprise.agents.autonomous_goal_generation_agent",
    "AutonomousGoalGenerationAgent",
)
CausalReasoningAgent = _load_optional_agent(
    "ninai_enterprise.agents.causal_reasoning_agent",
    "CausalReasoningAgent",
)
ConceptLearningAgent = _load_optional_agent(
    "ninai_enterprise.agents.concept_learning_agent",
    "ConceptLearningAgent",
)
ConflictDetectionAgent = _load_optional_agent(
    "ninai_enterprise.agents.conflict_detection_agent",
    "ConflictDetectionAgent",
)
CounterfactualMemoryAgent = _load_optional_agent(
    "ninai_enterprise.agents.counterfactual_memory_agent",
    "CounterfactualMemoryAgent",
)
CredibilityAgent = _load_optional_agent("ninai_enterprise.agents.credibility_agent", "CredibilityAgent")
EpisodicFutureSimulationAgent = _load_optional_agent(
    "ninai_enterprise.agents.episodic_future_simulation_agent",
    "EpisodicFutureSimulationAgent",
)
ErrorRecoveryAgent = _load_optional_agent(
    "ninai_enterprise.agents.error_recovery_agent",
    "ErrorRecoveryAgent",
)
FeedbackIntegrationAgent = _load_optional_agent(
    "ninai_enterprise.agents.feedback_integration_agent",
    "FeedbackIntegrationAgent",
)
FeedbackLearningAgent = _load_optional_agent(
    "ninai_enterprise.agents.feedback_learning_agent",
    "FeedbackLearningAgent",
)
GoalDecompositionAgent = _load_optional_agent(
    "ninai_enterprise.agents.goal_decomposition_agent",
    "GoalDecompositionAgent",
)
GraphLinkingAgent = _load_optional_agent(
    "ninai_enterprise.agents.graph_linking_agent",
    "GraphLinkingAgent",
)
HumanReviewQueueAgent = _load_optional_agent(
    "ninai_enterprise.agents.human_review_queue_agent",
    "HumanReviewQueueAgent",
)
MemoryConsolidationAgent = _load_optional_agent(
    "ninai_enterprise.agents.memory_consolidation_agent",
    "MemoryConsolidationAgent",
)
MemoryDecayAgent = _load_optional_agent("ninai_enterprise.agents.memory_decay_agent", "MemoryDecayAgent")
MetaCognitivePlanningAgent = _load_optional_agent(
    "ninai_enterprise.agents.meta_cognitive_planning_agent",
    "MetaCognitivePlanningAgent",
)
PlaybookAgent = _load_optional_agent("ninai_enterprise.agents.playbook_agent", "PlaybookAgent")
PredictiveMonitorAgent = _load_optional_agent(
    "ninai_enterprise.agents.predictive_monitor_agent",
    "PredictiveMonitorAgent",
)
PromotionAgent = _load_optional_agent("ninai_enterprise.agents.promotion_agent", "PromotionAgent")
SelfImprovementPlannerAgent = _load_optional_agent(
    "ninai_enterprise.agents.self_improvement_planner_agent",
    "SelfImprovementPlannerAgent",
)
SemanticChangeDetectionAgent = _load_optional_agent(
    "ninai_enterprise.agents.semantic_change_detection_agent",
    "SemanticChangeDetectionAgent",
)
UncertaintyReportingAgent = _load_optional_agent(
    "ninai_enterprise.agents.uncertainty_reporting_agent",
    "UncertaintyReportingAgent",
)
WorldModelAgent = _load_optional_agent("ninai_enterprise.agents.world_model_agent", "WorldModelAgent")


OSS_AGENT_CLASSES: tuple[type[BaseAgent], ...] = (
    ClassificationAgent,
    MetadataExtractionAgent,
    TopicModelingAgent,
    PatternDetectionAgent,
    LogseqExportAgent,
    SemanticNormalizationAgent,
    EntityResolutionAgent,
    ContextAmplifierAgent,
    SiloPropagationAgent,
    OrgAttentionAgent,
    ProactiveMemoryPushAgent,
    TemporalReasoningAgent,
    EpisodicGroupingAgent,
    NarrativeSynthesisAgent,
    AnomalyDetectionAgent,
    QueryIntelligenceAgent,
    OrchestrationBusAgent,
    PlaybookExecutionTrackerAgent,
    MultiTurnGoalTrackingAgent,
    AdaptiveEnrichmentBudgetAgent,
    TheoryOfMindAgent,
    MemorySleepAgent,
    CompositionalGeneralizationAgent,
    EmotionalAffectiveMemoryAgent,
    MultimodalDeepMemoryAgent,
    FederatedMemoryAgent,
    AdaptivePersonaAgent,
    ProspectiveMemoryAgent,
    AnalogicalReasoningAgent,
    NarrativeCompressionAgent,
    TemporalPatternMinerAgent,
    ActiveKnowledgeSeekerAgent,
    HierarchicalGoalPlannerAgent,
    SocialMemoryAgent,
    SemanticRoleInferenceAgent,
    CrossModalReasoningAgent,
    MemoryTierManagerAgent,
    DebateEnsembleAgent,
    PlaybookAutoSynthesisAgent,
    ErrorRemediationAgent,
    AutoResearchAgent,
)

ENTERPRISE_AGENT_CLASSES: tuple[type[BaseAgent], ...] = tuple(
    agent_class
    for agent_class in (
        PromotionAgent,
        GraphLinkingAgent,
        FeedbackLearningAgent,
        WorldModelAgent,
        PredictiveMonitorAgent,
        CausalReasoningAgent,
        ConflictDetectionAgent,
        AdaptiveConflictResolutionAgent,
        MemoryDecayAgent,
        MemoryConsolidationAgent,
        CredibilityAgent,
        PlaybookAgent,
        GoalDecompositionAgent,
        UncertaintyReportingAgent,
        FeedbackIntegrationAgent,
        AuditTrailAgent,
        HumanReviewQueueAgent,
        MetaCognitivePlanningAgent,
        AutonomousGoalGenerationAgent,
        AutonomousActionAgent,
        CounterfactualMemoryAgent,
        SemanticChangeDetectionAgent,
        ConceptLearningAgent,
        SelfImprovementPlannerAgent,
        EpisodicFutureSimulationAgent,
        ErrorRecoveryAgent,
    )
    if agent_class is not None
)

AGENT_CLASSES: tuple[type[BaseAgent], ...] = OSS_AGENT_CLASSES + ENTERPRISE_AGENT_CLASSES


def list_registered_agent_classes() -> tuple[type[BaseAgent], ...]:
    return AGENT_CLASSES


def list_registered_agents() -> list[BaseAgent]:
    return [agent_class() for agent_class in AGENT_CLASSES]


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
        return _instantiate_agent(PromotionAgent)

    if name in {"graph", "graphlinking", "graphlinkingagent"}:
        return _instantiate_agent(GraphLinkingAgent)

    if name in {"logseq", "logseq_export", "logseqexport", "logseqexportagent"}:
        return LogseqExportAgent()

    if name in {"feedback", "feedbacklearning", "feedbacklearningagent"}:
        return _instantiate_agent(FeedbackLearningAgent)

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

    if name in {
        "proactive_push",
        "proactivepush",
        "proactivememorpush",
        "proactivememory",
        "proactivememorypush",
        "proactivememorypushagent",
    }:
        return ProactiveMemoryPushAgent()

    if name in {"world_model", "worldmodel", "worldmodelagent"}:
        return _instantiate_agent(WorldModelAgent)

    if name in {"predictive_monitor", "predictivemonitor", "predictivemonitoragent"}:
        return _instantiate_agent(PredictiveMonitorAgent)

    if name in {"causal_reasoning", "causalreasoning", "causalreasoningagent"}:
        return _instantiate_agent(CausalReasoningAgent)

    if name in {"conflict_detection", "conflictdetection", "conflictdetectionagent"}:
        return _instantiate_agent(ConflictDetectionAgent)

    if name in {
        "adaptive_conflict_resolution",
        "adaptiveconflictresolution",
        "adaptiveconflictresolutionagent",
    }:
        return _instantiate_agent(AdaptiveConflictResolutionAgent)

    if name in {"memory_decay", "memorydecay", "memorydecayagent"}:
        return _instantiate_agent(MemoryDecayAgent)

    if name in {"memory_consolidation", "memoryconsolidation", "memoryconsolidationagent"}:
        return _instantiate_agent(MemoryConsolidationAgent)

    if name in {"temporal_reasoning", "temporalreasoning", "temporalreasoningagent"}:
        return TemporalReasoningAgent()

    if name in {"episodic_grouping", "episodicgrouping", "episodicgroupingagent"}:
        return EpisodicGroupingAgent()

    if name in {"credibility", "credibility_agent", "credibilityagent"}:
        return _instantiate_agent(CredibilityAgent)

    if name in {"playbook", "playbook_agent", "playbookagent"}:
        return _instantiate_agent(PlaybookAgent)

    if name in {"goal_decomposition", "goaldecomposition", "goaldecompositionagent"}:
        return _instantiate_agent(GoalDecompositionAgent)

    if name in {
        "uncertainty_reporting",
        "uncertaintyreporting",
        "uncertaintyreportingagent",
        "uncertainty_agent",
    }:
        return _instantiate_agent(UncertaintyReportingAgent)

    if name in {
        "narrative_synthesis",
        "narrativesynthesis",
        "narrativesynthesisagent",
        "narrative_agent",
    }:
        return NarrativeSynthesisAgent()

    if name in {
        "feedback_integration",
        "feedbackintegration",
        "feedbackintegrationagent",
        "feedback_agent",
    }:
        return _instantiate_agent(FeedbackIntegrationAgent)

    if name in {"anomaly_detection", "anomalydetection", "anomalydetectionagent", "anomaly_agent"}:
        return AnomalyDetectionAgent()

    if name in {"query_intelligence", "queryintelligence", "queryintelligenceagent", "query_agent"}:
        return QueryIntelligenceAgent()

    if name in {"orchestration_bus", "orchestrationbus", "orchestrationbusagent"}:
        return OrchestrationBusAgent()

    if name in {"audit_trail", "audittrail", "audittrailagent", "audit_agent"}:
        return _instantiate_agent(AuditTrailAgent)

    if name in {
        "human_review_queue",
        "humanreviewqueue",
        "humanreviewqueueagent",
        "human_review_queue_agent",
    }:
        return _instantiate_agent(HumanReviewQueueAgent)

    if name in {"playbook_execution_tracker", "playbookexecutiontracker", "playbookexecutiontrackeragent"}:
        return PlaybookExecutionTrackerAgent()

    if name in {
        "multi_turn_goal_tracking",
        "multiturngoaltracking",
        "multiturngoaltracingagent",
        "multiturngoaltrackinagent",
        "multiturngoaltrackingagent",
        "multi_turn_goal",
    }:
        return MultiTurnGoalTrackingAgent()

    if name in {
        "adaptive_enrichment_budget",
        "adaptiveenrichmentbudget",
        "adaptiveenrichmentbudgetagent",
        "enrichment_budget",
    }:
        return AdaptiveEnrichmentBudgetAgent()

    if name in {
        "meta_cognitive_planning",
        "metacognitiveplanning",
        "metacognitiveplanningagent",
        "meta_cognitive",
    }:
        return _instantiate_agent(MetaCognitivePlanningAgent)

    if name in {"autonomous_goal_generation", "autonomousgoalgeneration", "autonomousgoalgenerationagent"}:
        return _instantiate_agent(AutonomousGoalGenerationAgent)

    if name in {"theory_of_mind", "theoryofmind", "theoryofmindagent"}:
        return TheoryOfMindAgent()

    if name in {"memory_sleep", "memorysleep", "memorysleepagent"}:
        return MemorySleepAgent()

    if name in {
        "compositional_generalization",
        "compositionalgeneralization",
        "compositionalgeneralizationagent",
    }:
        return CompositionalGeneralizationAgent()

    if name in {
        "emotional_affective_memory",
        "emotionalaffectivememory",
        "emotionalaffectivememoryagent",
        "emotional_affective",
    }:
        return EmotionalAffectiveMemoryAgent()

    if name in {
        "multimodal_deep_memory",
        "multimodaldeepmemory",
        "multimodaldeepmemoryagent",
        "multimodal_memory",
        "multimodalmemory",
    }:
        return MultimodalDeepMemoryAgent()

    if name in {
        "federated_memory",
        "federatedmemory",
        "federatedmemoryagent",
        "collective_intelligence",
        "collectiveintelligence",
    }:
        return FederatedMemoryAgent()

    if name in {
        "autonomous_action",
        "autonomousaction",
        "autonomousactionagent",
        "autonomous_action_agent",
        "action_engine",
        "actionengine",
    }:
        return _instantiate_agent(AutonomousActionAgent)

    if name in {
        "counterfactual_memory",
        "counterfactualmemory",
        "counterfactualmemoryagent",
        "counterfactual",
    }:
        return _instantiate_agent(CounterfactualMemoryAgent)

    if name in {"adaptive_persona", "adaptivepersona", "adaptivepersonaagent", "persona"}:
        return AdaptivePersonaAgent()

    if name in {
        "prospective_memory",
        "prospectivememory",
        "prospectivememoryagent",
        "deadline_tracker",
        "deadlinetracker",
    }:
        return ProspectiveMemoryAgent()

    if name in {
        "analogical_reasoning",
        "analogicalreasoning",
        "analogicalreasoningagent",
        "analogy",
        "skill_transfer",
        "skilltransfer",
    }:
        return AnalogicalReasoningAgent()

    if name in {
        "narrative_compression",
        "narrativecompression",
        "narrativecompressionagent",
        "compression",
        "retrospective_compression",
        "retrospectivecompression",
    }:
        return NarrativeCompressionAgent()

    if name in {
        "semantic_change_detection",
        "semanticchangedetection",
        "semanticchangedetectionagent",
        "change_detection",
        "semantic_change",
    }:
        return _instantiate_agent(SemanticChangeDetectionAgent)

    if name in {
        "concept_learning",
        "conceptlearning",
        "conceptlearningagent",
        "concepts",
        "concept_registry",
        "conceptregistry",
    }:
        return _instantiate_agent(ConceptLearningAgent)

    if name in {
        "self_improvement_planner",
        "selfimprovementplanner",
        "selfimprovementplanneragent",
        "self_improvement",
        "improvement_planner",
        "improvementplanner",
    }:
        return _instantiate_agent(SelfImprovementPlannerAgent)

    if name in {
        "temporal_pattern_miner",
        "temporalpatternminer",
        "temporalpatternmineragent",
        "temporal_patterns",
        "pattern_miner",
        "patternminer",
    }:
        return TemporalPatternMinerAgent()

    if name in {
        "active_knowledge_seeker",
        "activeknowledgeseeker",
        "activeknowledgeseekeragent",
        "knowledge_seeker",
        "knowledgeseeker",
        "knowledge_gap_seeker",
    }:
        return ActiveKnowledgeSeekerAgent()

    if name in {
        "hierarchical_goal_planner",
        "hierarchicalgoalplanner",
        "hierarchicalgoalplanneragent",
        "goal_hierarchy",
        "goalhierarchy",
        "hierarchical_planner",
    }:
        return HierarchicalGoalPlannerAgent()

    if name in {"social_memory", "socialmemory", "socialmemoryagent", "team_dynamics", "social_graph"}:
        return SocialMemoryAgent()

    if name in {
        "episodic_future_simulation",
        "episodicfuturesimulation",
        "episodicfuturesimulationagent",
        "future_simulation",
        "future_rehearsal",
        "episodic_simulation",
    }:
        return _instantiate_agent(EpisodicFutureSimulationAgent)

    if name in {"error_recovery", "errorrecovery", "errorrecoveryagent", "replan", "recovery", "recovery_agent"}:
        return _instantiate_agent(ErrorRecoveryAgent)

    if name in {
        "semantic_role_inference",
        "semanticroleinference",
        "semanticroleinferenceagent",
        "role_inference",
        "roleinference",
        "inferred_roles",
    }:
        return SemanticRoleInferenceAgent()

    if name in {
        "cross_modal_reasoning",
        "crossmodalreasoning",
        "crossmodalreasoningagent",
        "cross_modal",
        "crossmodal",
        "multimodal_reasoning",
    }:
        return CrossModalReasoningAgent()

    if name in {
        "memory_tier_manager",
        "memorytiermanager",
        "memorytiermanageragent",
        "memory_tiers",
        "working_set_manager",
    }:
        return MemoryTierManagerAgent()

    if name in {"debate_ensemble", "debateensemble", "debateensembleagent", "debate"}:
        return DebateEnsembleAgent()

    if name in {"playbook_auto_synthesis", "playbookautosynthesis", "playbookautosynthesisagent", "playbook_synthesis"}:
        return PlaybookAutoSynthesisAgent()

    if name in {"error_remediation", "errorremediation", "errorremediationagent", "remediation"}:
        return ErrorRemediationAgent()

    if name in {"auto_research", "autoresearch", "autoresearchagent"}:
        return AutoResearchAgent()

    return None
