"""
Database Models
===============

SQLAlchemy models for the Ninai memory operating system.
All models include organization_id for RLS-based multi-tenant isolation.
"""

from app.models.base import (
    Base,
    TimestampMixin,
    UUIDMixin,
    generate_uuid,
)
from app.models.organization import (
    Organization,
    OrganizationHierarchy,
)
from app.models.user import (
    User,
    Role,
    UserRole,
)
from app.models.admin import AdminRole, AdminSession, AdminIPWhitelist
from app.models.team import (
    Team,
    TeamMember,
)
from app.models.agent import Agent
from app.models.memory import (
    Memory,
    MemoryMetadata,
    MemorySharing,
)
from app.models.memory_attachment import MemoryAttachment
from app.models.audit import (
    AuditEvent,
    MemoryAccessLog,
)
from app.models.app_setting import AppSetting
from app.models.agent_run import AgentRun
from app.models.agent_run_event import AgentRunEvent
from app.models.agent_process import AgentProcess
from app.models.agent_result_cache import AgentResultCache
from app.models.memory_feedback import MemoryFeedback
from app.models.memory_edge import MemoryEdge
from app.models.memory_promotion_history import MemoryPromotionHistory
from app.models.memory_topic import MemoryTopic
from app.models.memory_topic_membership import MemoryTopicMembership
from app.models.memory_pattern import MemoryPattern
from app.models.memory_pattern_evidence import MemoryPatternEvidence
from app.models.memory_logseq_export import MemoryLogseqExport
from app.models.logseq_export_file import LogseqExportFile
from app.models.org_feedback_learning_config import OrgFeedbackLearningConfig
from app.models.org_logseq_export_config import OrgLogseqExportConfig
from app.models.knowledge_item import KnowledgeItem
from app.models.knowledge_item_version import KnowledgeItemVersion
from app.models.knowledge_review_request import KnowledgeReviewRequest
from app.models.api_key import ApiKey
from app.models.webhook import WebhookSubscription, WebhookOutboxEvent, WebhookDelivery
from app.models.export_job import ExportJob
from app.models.cognitive_session import CognitiveSession
from app.models.cognitive_iteration import CognitiveIteration
from app.models.tool_call_log import ToolCallLog
from app.models.evaluation_report import EvaluationReport
from app.models.goal import Goal, GoalActivityLog, GoalEdge, GoalMemoryLink, GoalNode
from app.models.self_model import SelfModelEvent, SelfModelProfile
from app.models.simulation_report import SimulationReport
from app.models.meta_agent import (
    MetaAgentRun,
    MetaConflictRegistry,
    BeliefStore,
    CalibrationProfile,
)
from app.models.capability_token import CapabilityToken
from app.models.knowledge import Knowledge
from app.models.event import Event
from app.models.snapshot import Snapshot
from app.models.mfa import (
    TOTPDevice,
    SMSDevice,
    WebAuthnDevice,
    MFAEnrollment,
)
from app.models.backup import (
    BackupTask,
    BackupSchedule,
    BackupRestore,
)
from app.models.memory_consolidation import MemoryConsolidation
from app.models.memory_consolidation_session import ConsolidationSession
from app.models.memory_arc import MemoryArc
from app.models.memory_episode import MemoryEpisode
from app.models.memory_episode_membership import MemoryEpisodeMembership
from app.models.memory_semantic_node import MemorySemanticNode
from app.models.memory_semantic_node_topic_history import MemorySemanticNodeTopicHistory
from app.models.navigation_edge import NavigationEdge
from app.models.episode import Episode, EpisodeStatus, EpisodeScopeType
from app.models.episode_event import EpisodeEvent, EpisodeEventType, EpisodeActorType
from app.models.episode_link import EpisodeLink, EpisodeLinkRelation
from app.models.memory_fact import MemoryFact, MemoryFactStatus
from app.models.autonomous_goal import AutonomousGoal
from app.models.knowledge_gap import KnowledgeGap
from app.models.autonomous_goal_outcome import AutonomousGoalOutcome
from app.models.tool_capability import ToolCapability, StrategyAdaptation, CapabilityDiscovery, ToolType
from app.models.temporal_reasoning import (
    TemporalFact,
    TemporalSequence,
    TemporalTrajectory,
    TemporalChangetype,
    SequenceType,
    PatternType,
    TrendDirection,
)
from app.models.meta_cognitive import CognitiveStrategy, EpistemicState, StrategySelected
from app.models.contradiction import Contradiction, ContradictionSeverity
from app.models.playbook import Playbook, PlaybookScopeType
from app.models.run_checkpoint import RunCheckpoint
from app.models.eval_suite import EvalSuite
from app.models.eval_run import EvalRun
from app.models.drift_report import DriftReport
from app.models.compositional_generalization import (
    AbstractProcedure,
    Analogy,
    AnalogyApplicability,
)
from app.models.affective_memory import (
    AffectiveMemory,
    EmotionalTrajectory,
    EmotionalInteractionEvent,
    EmotionalTag,
    EmotionalTrend,
    DeEscalationStrategy,
)
from app.models.federated_knowledge import (
    FederatedKnowledgeSummary,
    OrgBenchmark,
    PrivacyPolicy,
    FederatedContribution,
)
from app.models.strategy_library import StrategyLibraryEntry
from app.models.feature_readiness import FeatureReadinessConfig
from app.models.action_execution_record import ActionExecutionRecord
from app.models.connector_registration import ConnectorRegistration
from app.models.persona_profile import PersonaProfile
from app.models.prospective_reminder import ProspectiveReminder
from app.models.load_snapshot import LoadSnapshot
from app.models.learned_concept import LearnedConcept
from app.models.improvement_proposal import ImprovementProposal
from app.models.working_memory_item import WorkingMemoryItem
from app.models.temporal_pattern import TemporalPattern
from app.models.goal_hierarchy_node import GoalHierarchyNode

__all__ = [
    # Base
    "Base",
    "TimestampMixin",
    "UUIDMixin",
    "generate_uuid",
    # Organization
    "Organization",
    "OrganizationHierarchy",
    # User
    "User",
    "Role",
    "UserRole",
    # Admin
    "AdminRole",
    "AdminSession",
    "AdminIPWhitelist",
    # Team
    "Team",
    "TeamMember",
    # Agent
    "Agent",
    # Memory
    "Memory",
    "MemoryMetadata",
    "MemorySharing",
    "MemoryAttachment",
    # Audit
    "AuditEvent",
    "MemoryAccessLog",
    # Settings
    "AppSetting",
    # Agent runs
    "AgentRun",
    "AgentRunEvent",
    "AgentProcess",
    # Feedback
    "MemoryFeedback",
    # Graph edges
    "MemoryEdge",
    # Promotion history
    "MemoryPromotionHistory",
    # Topics
    "MemoryTopic",
    "MemoryTopicMembership",
    # Patterns
    "MemoryPattern",
    "MemoryPatternEvidence",
    # Logseq exports
    "MemoryLogseqExport",
    "LogseqExportFile",
    "OrgLogseqExportConfig",
    # Feedback learning config
    "OrgFeedbackLearningConfig",
    "PersonaProfile",
    "ProspectiveReminder",
    "LoadSnapshot",
    "LearnedConcept",
    "ImprovementProposal",
    "WorkingMemoryItem",
    "TemporalPattern",
    "GoalHierarchyNode",

    # HITL knowledge review
    "KnowledgeItem",
    "KnowledgeItemVersion",
    "KnowledgeReviewRequest",

    # API keys
    "ApiKey",

    # Webhooks
    "WebhookSubscription",
    "WebhookOutboxEvent",
    "WebhookDelivery",

    # Export jobs
    "ExportJob",

    # Cognitive loop
    "CognitiveSession",
    "CognitiveIteration",
    "ToolCallLog",
    "EvaluationReport",

    # GoalGraph
    "Goal",
    "GoalNode",
    "GoalEdge",
    "GoalMemoryLink",
    "GoalActivityLog",

    # SelfModel
    "SelfModelProfile",
    "SelfModelEvent",

    # Simulation
    "SimulationReport",

    # Meta agent supervision & calibration
    "MetaAgentRun",
    "MetaConflictRegistry",
    "BeliefStore",
    "CalibrationProfile",
    
    # Phase 2: Memory Syscall Surface
    "CapabilityToken",
    "Knowledge",
    
    # Phase 7: Event Publishing & Batch Operations
    "Event",
    "WebhookSubscription",
    "Snapshot",
    
    # Week 2: MFA
    "TOTPDevice",
    "SMSDevice",
    "WebAuthnDevice",
    "MFAEnrollment",
    
    # Week 2: Backup
    "BackupTask",
    "BackupSchedule",
    "BackupRestore",
    # Consolidations
    "MemoryConsolidation",
    "ConsolidationSession",
    "MemoryArc",
    # GAP-1: Four-Level Hierarchy
    "MemoryEpisode",
    "MemoryEpisodeMembership",
    "MemorySemanticNode",
    # GAP-5: Retroactive Restructuring
    "MemorySemanticNodeTopicHistory",
    # GAP-6: kNN Navigation Graph
    "NavigationEdge",
    # PR1: Episode Case Continuity (Advanced Memory Features)
    "Episode",
    "EpisodeStatus",
    "EpisodeScopeType",
    "EpisodeEvent",
    "EpisodeEventType",
    "EpisodeActorType",
    "EpisodeLink",
    "EpisodeLinkRelation",
    # PR3: Facts + Contradictions
    "MemoryFact",
    "MemoryFactStatus",
    "Contradiction",
    "ContradictionSeverity",
    # PR3: Autonomous Goals & Intrinsic Motivation
    "AutonomousGoal",
    "KnowledgeGap",
    "AutonomousGoalOutcome",
    # PR4: Tool Capability Learning & Adaptive Strategy Selection
    "ToolCapability",
    "StrategyAdaptation",
    "CapabilityDiscovery",
    "ToolType",
    # PR5: Temporal Reasoning Engine
    "TemporalFact",
    "TemporalSequence",
    "TemporalTrajectory",
    "TemporalChangetype",
    "SequenceType",
    "PatternType",
    "TrendDirection",
    # PR6: Meta-Cognitive Planning
    "CognitiveStrategy",
    "EpistemicState",
    "StrategySelected",
    # PR4: Procedural/Skill Memory (Playbooks)
    "Playbook",
    "PlaybookScopeType",
    # PR5: Checkpoints (Replayability)
    "RunCheckpoint",
    # PR6: Eval Harness + Drift Detection
    "EvalSuite",
    "EvalRun",
    "DriftReport",
    # PR-7: Compositional Generalization Engine
    "AbstractProcedure",
    "Analogy",
    "AnalogyApplicability",
    # PR-8: Emotional & Affective Memory
    "AffectiveMemory",
    "EmotionalTrajectory",
    "EmotionalInteractionEvent",
    "EmotionalTag",
    "EmotionalTrend",
    "DeEscalationStrategy",
    # PR-10: Federated Memory & Collective Intelligence
    "FederatedKnowledgeSummary",
    "OrgBenchmark",
    "PrivacyPolicy",
    "FederatedContribution",
    # Strategy Learning (Outcome-Driven Strategy Learning)
    "StrategyLibraryEntry",
    # Phase 28: Feature Readiness Engine
    "FeatureReadinessConfig",
    # Phase 47: Autonomous Action Engine
    "ActionExecutionRecord",
    # Phase 48: Environment Connector Hub
    "ConnectorRegistration",
]
