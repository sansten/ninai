"""
API v1 Router
=============

Main router that combines all API v1 endpoints.
"""

from fastapi import APIRouter

from app.api.v1.endpoints import (
    auth,
    actions,
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
    cognitive_session_protocol,
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
    causal,
    memory_insights,
    compliance,
    digests,
    proof,
    ws_stream as ws_stream_endpoint,
    sse_stream,
    connectors,
    cognitive_gateway,
    cognitive_gateway_stream,
    cognitive_diff,
    cognitive_federated,
    cognitive_forget,
    cognitive_simulate,
    feedback,
    plugins,
    provenance,
    explainability,
    scim,
    billing,
    onboarding,
    usage,
    data_residency,
    benchmarks,
    a2a,
    openai_tool_schema,
)
from app.api.v1.endpoints import cognitive_schedules
from app.api.v1.admin import routes as admin_routes
from app.api.v1.endpoints import event_publishing_batch
from app.api.v1 import features

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(
    scim.router,
    prefix="/scim/v2",
    tags=["SCIM"],
)

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

# memory_insights must be registered before memories to prevent /memories/insights
# being matched by memories' /{memory_id} parametric route.
api_router.include_router(
    memory_insights.router,
    prefix="/memories",
    tags=["Memory Insights"],
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
    actions.router,
    prefix="/actions",
    tags=["Actions"],
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
    cognitive_session_protocol.router,
    prefix="/cognitive",
    tags=["Cognitive Session Protocol"],
)

api_router.include_router(
    cognitive_diff.router,
    prefix="/cognitive/diff",
    tags=["Cognitive Diff"],
)

api_router.include_router(
    feedback.router,
    prefix="/feedback",
    tags=["Feedback"],
)

api_router.include_router(
    plugins.router,
    prefix="/plugins",
    tags=["Plugins"],
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

# Cognitive OS Phase 2+ Endpoints
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

api_router.include_router(
    causal.router,
    tags=["Causal Reasoning"],
)

api_router.include_router(
    compliance.router,
    tags=["Compliance (GDPR)"],
)

api_router.include_router(
    digests.router,
    prefix="/digests",
    tags=["Intelligence Digest"],
)

api_router.include_router(
    proof.router,
    prefix="/proof",
    tags=["Proof"],
)

api_router.include_router(
    ws_stream_endpoint.router,
    tags=["Real-Time Event Stream"],
)

api_router.include_router(
    sse_stream.router,
    prefix="/sse",
    tags=["Cognitive Streaming API"],
)

api_router.include_router(
    connectors.router,
    prefix="/integrations",
    tags=["Connector Hub"],
)

api_router.include_router(
    cognitive_gateway.router,
    prefix="/cognitive/gateway",
    tags=["Cognitive Gateway"],
)

api_router.include_router(
    cognitive_federated.router,
    prefix="/cognitive/federated",
    tags=["Cognitive Federated Learning"],
)

api_router.include_router(
    cognitive_simulate.router,
    prefix="/cognitive",
    tags=["Cognitive Simulation"],
)

api_router.include_router(
    cognitive_forget.router,
    prefix="/cognitive",
    tags=["Cognitive Forgetting"],
)

api_router.include_router(
    cognitive_gateway_stream.router,
    prefix="/cognitive/gateway/stream",
    tags=["Cognitive Gateway Stream"],
)

api_router.include_router(
    explainability.router,
    prefix="/explain",
    tags=["Explainability"],
)

api_router.include_router(
    provenance.router,
    prefix="/provenance",
    tags=["Provenance"],
)

api_router.include_router(
    benchmarks.router,
    tags=["Admin - Benchmarks"],
)

api_router.include_router(
    a2a.router,
    prefix="/a2a",
    tags=["A2A Protocol"],
)

api_router.include_router(
    openai_tool_schema.router,
    tags=["OpenAI Tool Schema"],
)

api_router.include_router(
    billing.router,
    tags=["Billing"],
)

api_router.include_router(
    onboarding.router,
    tags=["Onboarding"],
)

api_router.include_router(
    usage.router,
    tags=["Usage"],
)

api_router.include_router(
    data_residency.router,
    tags=["Data Residency"],
)

api_router.include_router(
    cognitive_schedules.router,
    prefix="/cognitive/schedules",
    tags=["Cognitive Schedules"],
)



