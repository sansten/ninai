# Ninai Backend

## Stack
- FastAPI + SQLAlchemy async (PostgreSQL, multi-tenant RLS)
- Redis (cache/broker), Qdrant (vector), Celery (tasks)
- Local LLM: Ollama (qwen2.5:0.5b default), `AGENT_STRATEGY=llm|heuristic`
- Python 3.12+, pytest-asyncio

## Run Tests
```
python -m pytest tests/ -x -q
```

## Contribution Style — ALWAYS follow these rules
- **No AI attribution** — never add `Co-Authored-By: Claude` or any AI tool credit to commits
- **No AI branding in PRs** — no "Generated with Claude Code" or similar footers in PR bodies
- **Write like a human developer** — code, comments, commit messages, and PR descriptions must read as natural human engineering work; no overly structured AI-style prose
- **Commit messages** — concise, imperative, lowercase subject line (e.g. `feat: add org attention model`); no meta-commentary about tools used
- **PR bodies** — plain summary + test plan; no emoji-heavy lists, no tool signatures

## Phase Git Workflow — ALWAYS do this after completing a phase

After all tests pass for a phase, run these steps without asking for confirmation:

1. **Create a branch** from the current HEAD:
   ```
   git checkout -b agi/phase-{N}-{short-slug}
   ```
   e.g. `agi/phase-19-credibility-scoring`

2. **Stage only the phase files** (agent + tests + registry + CLAUDE.md):
   ```
   git add app/agents/{agent_file}.py tests/test_{agent_file}.py \
           app/agents/registry.py CLAUDE.md
   ```

3. **Commit** with a concise human-style message:
   ```
   git commit -m "feat: add {phase name} agent (phase {N})"
   ```
   No AI attribution, no tool signatures anywhere in the message.

4. **Push** the branch:
   ```
   git push -u origin agi/phase-{N}-{short-slug}
   ```

5. **Open a PR** to `main` using admin privileges so it can merge without a reviewer:
   ```
   gh pr create --title "feat: phase {N} — {Phase Name}" \
     --body "..." \
     --base main \
     --head agi/phase-{N}-{short-slug}
   ```
   Then immediately merge it:
   ```
   gh pr merge --admin --squash --delete-branch
   ```

   PR body format (plain, no emoji lists, no tool footers):
   ```
   Add {AgentName} for phase {N}.

   {One or two sentences describing what the agent does and what it feeds into.}

   Test plan:
   - {N} unit tests covering heuristic path, LLM path, fallback, validate_outputs, registry
   - Full suite passes ({total} passed, {skipped} skipped)
   ```

## Key Paths
```
app/
  services/
    cognitive_loop/
      orchestrator.py       # LoopOrchestrator — central AGI loop
      planner_agent.py      # PlannerAgent — LLM/heuristic planner
      critic_agent.py       # CriticAgent — evaluation
      executor_agent.py     # ExecutorAgent — tool execution
      evidence_service.py   # Evidence retrieval (limit=5, summary 300 chars)
    cognitive_context_aggregator.py   # Phase 3: cross-service intel bus
    strategy_learning_service.py      # Phase 2: outcome-driven strategy learning
    adaptive_strategy_service.py      # Tool EMA success rates
    self_model_service.py             # Tool/domain confidence EMA
    meta_cognitive_service.py         # Epistemic state (partial stub)
    intrinsic_motivation_service.py   # Knowledge gap detection (partial stub)
    causal_reasoning_service.py       # Causal BFS (partial stub)
    memory_activation/scoring.py      # 8-component sigmoid scorer
  models/
    strategy_library.py     # StrategyLibraryEntry (Phase 2 model)
  prompts/cognitive_loop/
    planner_v1.txt           # Planner prompt — 7 sections incl. cognitive_context
  tasks/
    cognitive_loop.py        # Celery entrypoint — wires all services
tests/
  test_cognitive_context_aggregator.py   # Phase 3 — 10 tests
  test_strategy_learning_service.py      # Phase 2 — 18 tests
  test_planner_agent_strategy_wiring.py  # Phase 2 planner — 6 tests
  test_loop_orchestrator_strategy_wiring.py  # Phase 2 orch — 11 tests
  test_world_model_agent.py                  # Phase 4 — 47 tests
  test_predictive_monitor_agent.py           # Phase 5 — 63 tests
  test_entity_resolution_agent.py            # Phase 7 — 44 tests
  test_silo_propagation_agent.py             # Phase 9 — 47 tests
  test_causal_reasoning_agent.py             # Phase 12 — 47 tests
  test_conflict_detection_agent.py           # Phase 13 — 61 tests
  test_adaptive_conflict_resolution_agent.py # Phase 14 — 55 tests
  test_memory_decay_agent.py                 # Phase 15 — 57 tests
  test_memory_consolidation_agent.py        # Phase 16 — 72 tests
  test_temporal_reasoning_agent.py          # Phase 17 — 82 tests
  test_episodic_grouping_agent.py           # Phase 18 — 89 tests
  test_credibility_agent.py                # Phase 19 — 63 tests
  test_playbook_agent.py                   # Phase 20 — 94 tests
  test_goal_decomposition_agent.py         # Phase 21 — 112 tests
  test_uncertainty_reporting_agent.py     # Phase 22 — 100 tests
  test_narrative_synthesis_agent.py      # Phase 23 — 108 tests
  test_feedback_integration_agent.py    # Phase 24 — 84 tests
  test_anomaly_detection_agent.py       # Phase 25 — 89 tests
  test_memory_enrichment_endpoints.py   # Phase 26 — 26 tests
  test_orchestration_bus_agent.py       # Phase 29 — 65 tests
  test_knowledge_graph_endpoints.py     # Phase 30 — 33 tests
  test_audit_trail_agent.py            # Phase 31 — 108 tests
  test_human_review_queue_agent.py     # Phase 32 — 65 tests
  test_playbook_execution_tracker_agent.py  # Phase 33 — 69 tests
  test_multi_turn_goal_tracking_agent.py    # Phase 34 — 68 tests
  test_meta_cognitive_planning_agent.py          # Phase 36 — 70 tests
  test_autonomous_goal_generation_agent.py       # Phase 37 — 87 tests
  test_theory_of_mind_agent.py                   # Phase 38 — 96 tests
  test_compositional_generalization_agent.py     # Phase 41 — 53 tests
  test_emotional_affective_memory_agent.py       # Phase 42 — 87 tests
  test_multimodal_deep_memory_agent.py           # Phase 43 — 74 tests
  test_ws_stream.py                              # Phase 45 — 20 tests
  test_digest_pipeline.py                        # Phase 46 — 41 tests
  test_temporal_pattern_miner_agent.py           # Phase 62 — 29 tests
  test_active_knowledge_seeker_agent.py          # Phase 63 — 38 tests
  test_uncertainty_propagation_service.py        # Phase 64 — 34 tests
  test_hierarchical_goal_planner_agent.py        # Phase 65 — 39 tests
  test_social_memory_agent.py                    # Phase 66 — 40 tests
  test_episodic_future_simulation_agent.py       # Phase 67 — 39 tests
  test_error_recovery_agent.py                   # Phase 68 — 41 tests
  test_semantic_role_inference_agent.py          # Phase 69 — 34 tests
  test_confidence_ensemble_service.py            # Phase 70 — 25 tests
  test_memory_importance_ranker.py               # Phase 71 — 25 tests
  e2e/
    data.py                                      # Kaggle helpdesk fixture loader
    test_realworld_decay_credibility.py          # E2E — 15 tests
    test_realworld_anomaly_conflict.py           # E2E — 14 tests
    test_realworld_pipeline.py                   # E2E — 6 tests
```

## AGI Implementation Status

| Phase | Feature | Status |
|-------|---------|--------|
| 1a | Uncertainty gating early-stop bug fix | Done |
| 1b | AdaptiveStrategy → Planner wiring | Done |
| 2 | Outcome-Driven Strategy Learning | Done (StrategyLearningService) |
| 3 | Cross-Service Intelligence Bus (CognitiveContextAggregator) | Done |
| 4 | Unified World Model Graph | Done (WorldModelAgent) |
| 5 | Predictive World State Monitor | Done (PredictiveMonitorAgent) |
| 6 | Semantic Normalization Layer | Done (SemanticNormalizationAgent) |
| 7 | Enterprise Ontology + Entity Resolution | Done (EntityResolutionAgent) |
| 8 | Expert Context Amplifier | Done (ContextAmplifierAgent) |
| 9 | Cross-Silo Signal Propagation | Done (SiloPropagationAgent) |
| 10 | Organizational Attention Model | Done (OrgAttentionAgent) |
| 11 | Proactive Memory Push | Done (ProactiveMemoryPushAgent) |
| 12 | Causal Reasoning Engine | Done (CausalReasoningAgent) |
| 13 | Cross-Silo Conflict Detection | Done (ConflictDetectionAgent) |
| 14 | Adaptive Conflict Resolution | Done (AdaptiveConflictResolutionAgent) |
| 15 | Memory Decay & Relevance Aging | Done (MemoryDecayAgent) |
| 16 | Memory Consolidation Engine | Done (MemoryConsolidationAgent) |
| 17 | Temporal Reasoning Engine | Done (TemporalReasoningAgent) |
| 18 | Episodic Memory Grouping | Done (EpisodicGroupingAgent) |
| 19 | Credibility Scoring | Done (CredibilityAgent) |
| 20 | Playbook / Skill Memory | Done (PlaybookAgent) |
| 21 | Goal Decomposition | Done (GoalDecompositionAgent) |
| 22 | Uncertainty Reporting | Done (UncertaintyReportingAgent) |
| 23 | Narrative Synthesis | Done (NarrativeSynthesisAgent) |
| 24 | Feedback Integration & Self-Correction | Done (FeedbackIntegrationAgent) |
| 25 | Anomaly Detection | Done (AnomalyDetectionAgent) |
| 26 | Enrichment API Surface | Done (memory_enrichment endpoints) |
| 27 | Query Intelligence Layer | Done (QueryIntelligenceAgent) |
| 28 | Feature Readiness Engine | Done (FeatureReadinessService, FeatureReadinessConfig, /features/readiness) |
| 29 | Cross-Agent Orchestration Bus | Done (OrchestrationBusAgent) |
| 30 | Tenant-Scoped Knowledge Graph API | Done (knowledge_graph endpoints: /graph/neighbors, /graph/entity-path, /graph/changes) |
| 31 | Audit & Explainability Trail | Done (AuditTrailAgent) |
| 32 | Human Review Queue | Done (HumanReviewQueueAgent, /review/queue, /review/claim, /review/resolve) |
| 33 | Playbook Execution Tracker | Done (PlaybookExecutionTrackerAgent) |
| 34 | Multi-Turn Goal Tracking | Done (MultiTurnGoalTrackingAgent) |
| 35 | Adaptive Enrichment Budget | Done (AdaptiveEnrichmentBudgetAgent) |
| 36 | Meta-Cognitive Planning | Done (MetaCognitivePlanningAgent) |
| 37 | Autonomous Goal Generation & Intrinsic Motivation | Done (AutonomousGoalGenerationAgent) |
| 38 | Theory of Mind & Multi-Agent Modeling | Done (TheoryOfMindAgent) |
| 39 | Causal API Surface | Done (/causal/edges, /causal/explain, /causal/counterfactual, /causal/predict, /causal/do-calculus, /causal/validate) |
| 40 | Memory Sleep Cycle | Done (MemorySleepAgent, SleepCycleReport, memory_sleep_pipeline) |
| 41 | Compositional Generalization Engine | Done (CompositionalGeneralizationAgent) |
| 42 | Emotional & Affective Memory | Done (EmotionalAffectiveMemoryAgent) |
| 43 | Multimodal Deep Memory | Done (MultimodalDeepMemoryAgent) |
| 44 | Federated Memory & Collective Intelligence | Done (FederatedMemoryAgent + PR-10 federated services/endpoints) |
| 45 | Real-Time WebSocket Event Stream | Done (ws_stream.py, /ws/stream endpoint) |
| 46 | Intelligence Digest, GDPR Compliance & Memory Insights | Done (DigestService, DigestReport, /digest endpoints, ComplianceService, /export endpoints) |
| 51 | Counterfactual Memory Simulation | Done (CounterfactualMemoryAgent, 37 tests, 4554 total passing) |
| 52 | Adaptive Persona Engine | Done (AdaptivePersonaAgent, PersonaProfile model, PersonaProfileService, 38 tests, 4592 total passing) |
| 53 | Prospective Memory & Deadline Tracking | Done (ProspectiveMemoryAgent, ProspectiveReminder model, prospective_memory_pipeline Celery task, 47 tests, 4639 total passing) |
| 54 | Skill Transfer & Analogical Reasoning | Done (AnalogicalReasoningAgent, registry wiring, 40 tests, 4679 total passing) |
| 55 | Cognitive Load Balancer | Done (CognitiveLoadBalancer service, LoadSnapshot model, 37 tests, 4716 total passing) |
| 56 | Narrative Memory Compression | Done (NarrativeCompressionAgent, NarrativeCompressionService, 36 tests, 4752 total passing) |
| 57 | Semantic Change Detection | Done (SemanticChangeDetectionAgent, 33 tests, 4785 total passing) |
| 58 | Attention-Weighted Memory Retrieval | Done (AttentionRetrievalService, 25 tests, 4810 total passing) |
| 59 | Self-Supervised Concept Learning | Done (ConceptLearningAgent, LearnedConcept model, ConceptRegistryService, 36 tests, 4846 total passing) |
| 60 | Recursive Self-Improvement Planner | Done (SelfImprovementPlannerAgent, ImprovementProposal model, 30 tests, 4876 total passing) |
| 61 | Working Memory Manager | Done (WorkingMemoryService, WorkingMemoryItem model, 32 tests, 4908 total passing) |
| 62 | Temporal Pattern Miner | Done (TemporalPatternMinerAgent, TemporalPattern model, 29 tests, 4937 total passing) |
| 63 | Active Knowledge Seeker | Done (ActiveKnowledgeSeekerAgent, KnowledgeGap model updates, 38 tests, 4975 total passing) |
| 64 | Uncertainty Propagation Engine | Done (UncertaintyPropagationService, 34 tests, 5009 total passing) |
| 65 | Hierarchical Goal Planner | Done (HierarchicalGoalPlannerAgent, GoalHierarchyNode model, 39 tests, 5048 total passing) |
| 66 | Social Memory & Team Dynamics Agent | Done (SocialMemoryAgent, SocialGraphEdge model, 40 tests, 5088 total passing) |
| 67 | Episodic Future Simulation | Done (EpisodicFutureSimulationAgent, 39 tests, 5127 total passing) |
| 68 | Error Recovery & Replan Agent | Done (ErrorRecoveryAgent, 41 tests, 5168 total passing) |
| 69 | Semantic Role Inference Agent | Done (SemanticRoleInferenceAgent, InferredRole model, 34 tests, 5202 total passing) |
| 70 | Confidence Ensemble Service | Done (ConfidenceEnsembleService, 25 tests, 5227 total passing) |
| 71 | Memory Importance Ranker | Done (MemoryImportanceRanker, 25 tests, 5252 total passing) |

## Cognitive OS Vision

Ninai is not a cache — **apps write raw data, Ninai makes sense of it**. The system enriches writes,
links entities across the enterprise, assembles full context on read, reasons causally, plans
proactively, and models its own cognition. No silo runs in isolation.

### Open Phases (Cognitive OS Roadmap)

**Phase 36 — Meta-Cognitive Planning** *(DONE — MetaCognitivePlanningAgent)*

**Phase 37 — Autonomous Goal Generation & Intrinsic Motivation** *(DONE — AutonomousGoalGenerationAgent)*

**Phase 38 — Theory of Mind & Multi-Agent Modeling** *(DONE — TheoryOfMindAgent)*

**Phase 39 — Causal API Surface** *(DONE)*
- Exposes CausalReasoningService via REST endpoints; adds counterfactual model and do-calculus
- Fixed CausalReasoningService (get_edges, validate_edge signature), CausalEdge.to_dict(), wired /causal router
- Endpoints: `/causal/edges`, `/causal/explain/{effect_id}`, `/causal/counterfactual`, `/causal/predict`, `/causal/do-calculus`, `/causal/validate`, `/causal/discover/{episode_id}`
- 34 tests, 3415 total passing

**Phase 40 — Memory Sleep Cycle** *(medium priority)*
- Offline nightly consolidation: strengthens weak memories, merges redundant facts, prunes stale knowledge
- Phase 16 built the consolidation agent; this adds the batch pipeline and creative-association pass
- New: `memory_sleep_pipeline.py` Celery beat task (02:00 UTC), `SleepCycleReport` model
- Builds on: MemoryConsolidationAgent, MemoryDecayAgent, EpisodicGroupingAgent
- Ref: AGI_PATH_REQUIREMENTS.md PR-2

**Phase 41 — Compositional Generalization Engine** *(DONE — CompositionalGeneralizationAgent)*
- Remixes known playbooks to solve novel problems the agent has never seen exactly
- "Deploy to on-prem?" → adapts AWS playbook using abstract procedure primitives
- Outputs: `composed_procedure`, `source_playbooks[]`, `adaptation_confidence`, `novel_steps[]`
- Builds on: PlaybookAgent, PlaybookExecutionTrackerAgent, GoalDecompositionAgent
- 53 tests, 3540 total passing

**Phase 42 — Emotional & Affective Memory** *(DONE — EmotionalAffectiveMemoryAgent)*
- Tags memories with emotional valence; escalates emotionally-charged situations; adapts agent tone
- Outputs: `emotional_valence`, `arousal_level`, `empathy_flag`, `escalation_recommended`, `tone_guidance`
- Builds on: FeedbackIntegrationAgent, HumanReviewQueueAgent, NarrativeSynthesisAgent
- 87 tests, 3627 total passing

**Phase 43 — Multimodal Deep Memory** *(DONE — MultimodalDeepMemoryAgent)*
- Semantic understanding of attachments: images, diagrams, screenshots, audio, video
- "The screenshot where the button was red" → retrievable by visual content
- Classifies modality, extracts objects/colours/text from VisionMemoryService output, emits searchable_tags
- Builds on: VisionMemoryService (PR-9), SpatialMemory, MemoryAttachment, PlaybookAgent
- 74 tests, 3701 total passing

**Phase 44 — Federated Memory & Collective Intelligence** *(DONE — FederatedMemoryAgent)*
- Privacy-preserving cross-org knowledge synthesis over federated candidates and peer benchmarks
- Outputs: `federated_summary`, `benchmark_insight`, `sharing_recommendation`, `privacy_risk_score`, `federated_confidence`
- Builds on: FederatedKnowledgeService, DifferentialPrivacyService, OrgBenchmarkService, PR-10 `/federated/*` endpoints
- 25 tests (`test_federated_memory_agent.py`) passing

**Phase 45 — Real-Time WebSocket Event Stream** *(DONE)*
- Authenticated WebSocket endpoint `/ws/stream` streams memory lifecycle events (created, updated, decayed, conflict, anomaly) to connected clients in real time
- EventPublishingService publishes to Redis pub/sub; ws_stream.py relays to tenant-scoped socket connections
- 20 tests (`test_ws_stream.py`) passing

**Phase 46 — Intelligence Digest, GDPR Compliance & Memory Insights** *(DONE)*
- DigestService aggregates daily/weekly memory activity into a narrative summary; `/digest/latest`, `/digest/history`, `/digest/trigger`
- ComplianceService handles GDPR data export and deletion; `/export/request`, `/export/status`, `/export/download`
- MemoryInsightsService surfaces per-tenant health metrics (decay distribution, credibility histogram, anomaly rate)
- 41 tests (`test_digest_pipeline.py`) passing

## Key Architecture Patterns
- **EMA**: `new_rate = 0.75 * prev + 0.25 * outcome` (α=0.25)
- **Fault isolation**: every service call wrapped in `try/except`, returns None on failure
- **Fire-and-forget**: post-session ingests wrapped in `try/except pass`
- **RLS**: all queries scoped via `set_tenant_context()` before DB ops
- **Planner prompt sections**: goal, evidence_json, tools_json, self_model_json, tool_recommendations_json, strategy_context_json, cognitive_context_json
- **Evidence cards**: compact format — id, summary(≤300chars), title, tags, score (no raw content)
- **Goal type classification**: 7 buckets via keyword match — information_retrieval, analysis, generation, evaluation, planning, self_improvement, general

## Known Stubs
None — all partial implementations have been completed.

