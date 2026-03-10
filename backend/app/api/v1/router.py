"""
API v1 Router
=============

Main router that combines all API v1 endpoints.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    agent_runs,
    admin_settings,
    admin_knowledge,
    review_knowledge,
    knowledge,
    memories,
    consolidations,
    memory_stream,
    organizations,
    teams,
    users,
    audit,
    logseq,
    api_keys,
    webhooks,
    exports,
    cognitive_loop,
    meta_agent,
    goals,
    self_model,
    simulation_reports,
    tools,
    llm,
    agent_processes,
    pipeline_tasks,
    dead_letter_queue,
    health,
    memory_syscall,
    mfa,
    backups,
    capability_tokens,
    snapshots,
    dlq_management,
    policy_versions,
    memory_backups,
    admission_control,
    graph,
    recommendations,
    graph_relationships,
    topics,
    knowledge_reports,
    advanced_search,
    memory_activation,
    episodes,
    evaluation,
    consolidation_sleep,
    intrinsic_motivation_pr3,
    adaptive_strategy_pr4,
    temporal_reasoning_pr5,
    meta_cognitive_pr6,
    compositional_generalization_pr7,
    affective_memory_pr8,
    federated_memory_pr10,
    memory_enrichment,
    feature_readiness,
    knowledge_graph,
    review_queue,
)
from app.api.v1.admin import routes as admin_routes
from app.api.v1.endpoints import event_publishing_batch
from app.api.v1 import features

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"],
)

api_router.include_router(
    admin_routes.router,
)

api_router.include_router(
    agent_runs.router,
    prefix="/agents",
    tags=["Agents"],
)

api_router.include_router(
    admin_settings.router,
    prefix="/admin",
    tags=["Admin"],
)

api_router.include_router(
    api_keys.router,
    prefix="/admin",
    tags=["Admin"],
)

api_router.include_router(
    webhooks.router,
    prefix="/admin",
    tags=["Admin"],
)

api_router.include_router(
    exports.admin_router,
    prefix="/admin",
    tags=["Admin"],
)

api_router.include_router(
    admin_knowledge.router,
    prefix="/admin/knowledge",
    tags=["Admin - Knowledge"],
)

api_router.include_router(
    knowledge.router,
    prefix="/knowledge",
    tags=["Knowledge"],
)

api_router.include_router(
    review_knowledge.router,
    prefix="/review/knowledge",
    tags=["Review - Knowledge"],
)

api_router.include_router(
    memory_stream.router,
    prefix="/memories",
    tags=["Memories"],
)

api_router.include_router(
    memories.router,
    prefix="/memories",
    tags=["Memories"],
)

api_router.include_router(
    episodes.router,
   prefix="/episodes",
    tags=["Episodes"],
)

api_router.include_router(
    evaluation.router,
    prefix="/eval",
    tags=["Evaluation"],
)

api_router.include_router(
    recommendations.router,
    tags=["Recommendations"],
)

api_router.include_router(
    graph_relationships.router,
    tags=["Graph"],
)

api_router.include_router(
    topics.router,
    tags=["Topics"],
)

api_router.include_router(
    knowledge_reports.router,
    tags=["Knowledge"],
)

api_router.include_router(
    advanced_search.router,
    prefix="/memories",
    tags=["Advanced Search"],
)

api_router.include_router(
    memory_activation.router,
    prefix="/memory-activation",
    tags=["Memory Activation"],
)

api_router.include_router(
    capability_tokens.router,
    prefix="/api/v1",
    tags=["Capability Tokens"],
)

api_router.include_router(
    snapshots.router,
    prefix="/api/v1",
    tags=["Memory Snapshots"],
)

api_router.include_router(
    dlq_management.router,
    prefix="/api/v1",
    tags=["DLQ Management"],
)

api_router.include_router(
    policy_versions.router,
    prefix="/api/v1",
    tags=["Policy Versioning"],
)

api_router.include_router(
    memory_backups.router,
    prefix="/api/v1",
    tags=["Memory Backup"],
)

api_router.include_router(
    organizations.router,
    prefix="/organizations",
    tags=["Organizations"],
)

api_router.include_router(
    teams.router,
    prefix="/teams",
    tags=["Teams"],
)

api_router.include_router(
    users.router,
    prefix="/users",
    tags=["Users"],
)

api_router.include_router(
    audit.router,
    prefix="/audit",
    tags=["Audit"],
)

api_router.include_router(
    logseq.router,
    prefix="/logseq",
    tags=["Logseq"],
)

api_router.include_router(
    exports.public_router,
    tags=["Exports"],
)

api_router.include_router(
    cognitive_loop.router,
    prefix="/cognitive",
    tags=["Cognitive"],
)

api_router.include_router(
    meta_agent.router,
    prefix="/meta",
    tags=["Meta"],
)

api_router.include_router(
    goals.router,
    prefix="/goals",
    tags=["Goals"],
)

api_router.include_router(
    self_model.router,
    prefix="/selfmodel",
    tags=["SelfModel"],
)

# Alias for compatibility with docs/tests that use hyphenated prefix.
api_router.include_router(
    self_model.router,
    prefix="/self-model",
    tags=["SelfModel"],
)

api_router.include_router(
    simulation_reports.router,
    prefix="/simulation-reports",
    tags=["Simulation"],
)

api_router.include_router(
    tools.router,
    prefix="/tools",
    tags=["Tools"],
)

api_router.include_router(
    llm.router,
    prefix="/llm",
    tags=["LLM"],
)

api_router.include_router(
    agent_processes.router,
    prefix="/ops",
    tags=["Operations"],
)

api_router.include_router(
    graph.router,
    prefix="/graph",
    tags=["Graph"],
)

api_router.include_router(
    pipeline_tasks.router,
    prefix="/admin",
    tags=["Admin - Pipelines"],
)

api_router.include_router(
    dead_letter_queue.router,
    prefix="/admin/dlq",
    tags=["Admin - DLQ"],
)

api_router.include_router(
    health.router,
    prefix="/health",
    tags=["Health"],
)

api_router.include_router(
    memory_syscall.router,
    tags=["Memory Syscall"],

)

api_router.include_router(
    consolidations.router,
    prefix="/api/v1",
    tags=["Consolidations"],
)

api_router.include_router(
    consolidation_sleep.router,
    tags=["Consolidation PR-2"],
)

api_router.include_router(
    intrinsic_motivation_pr3.router,
    tags=["Autonomous Goals PR-3"],
)

api_router.include_router(
    adaptive_strategy_pr4.router,
    tags=["Tool Learning PR-4"],
)

api_router.include_router(
    temporal_reasoning_pr5.router,
    tags=["Temporal Reasoning PR-5"],
)

api_router.include_router(
    meta_cognitive_pr6.router,
    tags=["Meta-Cognitive Planning PR-6"],
)

api_router.include_router(
    compositional_generalization_pr7.router,
)

api_router.include_router(
    affective_memory_pr8.router,
    tags=["Emotional & Affective Memory PR-8"],
)

api_router.include_router(
    federated_memory_pr10.router,
    tags=["Federated Memory & Collective Intelligence PR-10"],
)

api_router.include_router(
    event_publishing_batch.router,
    tags=["Events & Batch Operations"],
)

api_router.include_router(
    mfa.router,
    tags=["MFA"],
)

api_router.include_router(
    backups.router,
    tags=["Backups"],
)

# Memory OS Phase 2+ Endpoints
api_router.include_router(
    admission_control.router,
    tags=["Admission Control"],
)

# Feature detection endpoint
api_router.include_router(
    features.router,
    tags=["Features"],
)

api_router.include_router(
    memory_enrichment.router,
    prefix="/memories",
    tags=["Memory Enrichment"],
)

api_router.include_router(
    feature_readiness.router,
    prefix="/features",
    tags=["Feature Readiness"],
)

api_router.include_router(
    knowledge_graph.router,
    prefix="/graph",
    tags=["Knowledge Graph"],
)

api_router.include_router(
    review_queue.router,
    prefix="/review",
    tags=["Human Review Queue"],
)
