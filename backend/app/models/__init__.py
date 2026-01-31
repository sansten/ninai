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
from app.models.webhook_subscription import WebhookSubscription
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
]
