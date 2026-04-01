# AGI Advancement Plan — Ninai Backend

This document is the single source of truth for the next wave of AGI capability phases.
Each phase is self-contained: it lists the exact files to create/modify, the data model,
the service/agent logic, and the test requirements.  GitHub Copilot should implement each
phase in order, run `python -m pytest tests/ -x -q` after every phase, then follow the
git workflow in CLAUDE.md (branch → commit → push → PR → merge).

---

## Ground Rules for Implementation

- `AGENT_STRATEGY=heuristic` for all tests — no DB, no Ollama, no Qdrant required.
- All new agents extend `app/agents/base.py` and register in `app/agents/registry.py`.
- All new Celery tasks are wired into `app/core/celery_app.py` (includes + routes + beat).
- No AI attribution in commits or PRs.
- Every phase ends with **all tests green** before moving to the next phase.

---

## Phase 51 — Counterfactual Memory Simulation

### What / Why
The system can explain what happened (causal) and predict what will happen (predictive
monitor). It cannot answer "what would have happened if we had acted differently?"
Counterfactual reasoning is required for root-cause post-mortems and what-if planning.

### Files to Create
- `app/agents/counterfactual_memory_agent.py`
- `tests/test_counterfactual_memory_agent.py`

### Agent: `CounterfactualMemoryAgent`

```python
# Inputs (via AgentInput / enrichment dict):
#   - memory_id: str          — the memory being counterfactually examined
#   - intervention: dict      — {"field": "severity", "from": "low", "to": "high"}
#   - causal_graph: list[dict] — edges from CausalReasoningService.get_edges()
#   - related_memories: list[dict]  — fetched by caller
#
# Outputs (AgentOutput extras dict):
#   - counterfactual_outcome: str  — narrative of what would have happened
#   - affected_nodes: list[str]    — memory IDs that would change
#   - confidence: float            — 0.0–1.0
#   - counterfactual_delta: dict   — {"probability_change": +0.35, "severity_shift": "low→critical"}
#   - assumptions: list[str]       — caveats (e.g. "assumes no concurrent incidents")
```

**Heuristic logic:**
1. Walk causal_graph via BFS from the intervened node.
2. For each reachable edge, compute `edge_weight * intervention_magnitude` as the
   probability delta.  `intervention_magnitude` = 1.0 if changing a boolean field,
   or `abs(numeric_to - numeric_from) / max(numeric_to, 1)` for numeric.
3. `affected_nodes` = all nodes reachable within 3 hops.
4. `counterfactual_outcome` = template string: "Had {field} been {to} instead of {from},
   {len(affected_nodes)} downstream effects would likely have occurred including
   {affected_nodes[:2]}."
5. `confidence` = `min(0.9, 0.5 + 0.1 * len(affected_nodes))` clamped.

**Tests (≥30):**
- Empty causal graph → affected_nodes=[], confidence low.
- Single-edge graph → 1 affected node.
- 3-hop BFS stops at depth 3.
- intervention_magnitude numeric calculation.
- confidence clamped at 0.9.
- counterfactual_outcome string contains field and values.
- assumptions list non-empty.
- validate_outputs passes.
- LLM path: mocked LLM returns valid JSON with all required keys.

---

## Phase 52 — Adaptive Persona Engine

### What / Why
All agents respond uniformly regardless of which user or org they are serving.
An adaptive persona engine calibrates tone, verbosity, and domain vocabulary to each
user's observed interaction style — making responses more useful to a novice vs. an expert.

### Files to Create
- `app/agents/adaptive_persona_agent.py`
- `app/services/persona_profile_service.py`
- `app/models/persona_profile.py` (SQLAlchemy model)
- `tests/test_adaptive_persona_agent.py`

### Model: `PersonaProfile`
```python
# Table: persona_profiles
# Columns:
#   id: UUID PK
#   user_id: UUID FK → users
#   org_id: UUID FK → organizations
#   expertise_level: str  ("novice", "intermediate", "expert")
#   preferred_verbosity: str  ("brief", "normal", "detailed")
#   domain_vocabulary: dict  (JSONB — {"acronyms": [...], "preferred_terms": {...}})
#   interaction_count: int  default 0
#   last_updated: datetime
```

### Service: `PersonaProfileService`
```python
# Methods:
#   async def get_or_create(*, db, user_id, org_id) -> PersonaProfile
#   async def update_from_interaction(*, db, user_id, org_id, signal: dict) -> PersonaProfile
#     — signal: {"query_length": int, "used_jargon": bool, "requested_detail": bool}
#     — updates expertise_level via EMA of signals, increments interaction_count
#   async def get_style_hints(*, db, user_id, org_id) -> dict
#     — returns {"tone": "...", "verbosity": "...", "vocabulary_hints": [...]}
```

### Agent: `AdaptivePersonaAgent`
```python
# Inputs:
#   - content: str             — raw response to adapt
#   - persona: dict            — from PersonaProfileService.get_style_hints()
#   - context_type: str        — "memory_read" | "decision" | "plan" | "alert"
#
# Outputs:
#   - adapted_content: str     — tone/verbosity adjusted version
#   - persona_applied: str     — "expert_brief" | "novice_detailed" | etc.
#   - changes_made: list[str]  — e.g. ["expanded acronyms", "added context"]
```

**Heuristic logic:**
- novice + detailed: expand acronyms in content, add "What this means:" suffix.
- expert + brief: strip parenthetical explanations `\(.*?\)`, truncate to 2 sentences max.
- intermediate + normal: no transformation, pass through.
- `persona_applied` = f"{expertise_level}_{preferred_verbosity}".
- `changes_made` = list of transformations applied.

**Tests (≥35):**
- novice profile expands acronyms.
- expert profile strips parentheticals.
- intermediate pass-through.
- PersonaProfileService.update_from_interaction increments interaction_count.
- EMA expertise shift: repeated expert signals move expertise toward "expert".
- get_style_hints returns correct dict shape.
- context_type="alert" always uses brief verbosity regardless of profile.
- validate_outputs passes.

---

## Phase 53 — Prospective Memory & Deadline Tracking

### What / Why
The system reacts to past events. It has no mechanism to set future reminders — "check if
the DB migration completed in 2 hours", "escalate if no human review by Friday".
Prospective memory is a key component of deliberate cognition.

### Files to Create
- `app/agents/prospective_memory_agent.py`
- `app/models/prospective_reminder.py`
- `app/tasks/prospective_memory_pipeline.py`
- `tests/test_prospective_memory_agent.py`

### Model: `ProspectiveReminder`
```python
# Table: prospective_reminders
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   user_id: UUID FK (nullable)
#   trigger_type: str  ("time", "event", "condition")
#   trigger_at: datetime (nullable — for time triggers)
#   trigger_condition: dict (JSONB — for condition triggers)
#   memory_id: UUID FK (nullable — memory this is anchored to)
#   goal_id: UUID FK (nullable)
#   reminder_content: str
#   status: str  ("pending", "fired", "cancelled")
#   fired_at: datetime (nullable)
#   created_at: datetime
```

### Agent: `ProspectiveMemoryAgent`
```python
# Inputs:
#   - content: str                — memory or goal content to analyse
#   - existing_reminders: list    — already set reminders for this org
#   - current_time: datetime      — injected for testability
#
# Outputs:
#   - reminders_suggested: list[dict]  — each: {trigger_type, trigger_at_offset_hours,
#                                        reminder_content, urgency}
#   - deadline_detected: bool
#   - deadline_tokens: list[str]   — e.g. ["by Friday", "within 2 hours"]
#   - confidence: float
```

**Heuristic logic:**
- Scan content for deadline tokens: ["by", "before", "within", "due", "deadline",
  "expires", "until", "no later than"].
- For each match, infer offset_hours: "2 hours" → 2, "Friday" → hours until next Friday
  from current_time, "tomorrow" → 24, "end of week" → hours until Friday 17:00.
- `deadline_detected` = len(deadline_tokens) > 0.
- `reminders_suggested` includes one entry per detected deadline with urgency="high" if
  offset_hours < 4, "medium" if < 48, "low" otherwise.

### Celery Task: `prospective_memory_scan_task`
- Runs every 5 minutes (beat schedule: `60 * 5`).
- Loads all ProspectiveReminder rows with status="pending" and trigger_at <= now().
- Fires each by publishing to EventPublishingService ("prospective_reminder_fired").
- Updates status="fired", fired_at=now().

**Tests (≥35):**
- "deploy by Friday" → deadline_detected=True.
- offset_hours correct for "within 2 hours".
- urgency="high" for < 4 hours.
- No deadline tokens → deadline_detected=False, empty suggestions.
- Celery task fires overdue reminders (mock DB returning expired rows).
- Celery task skips non-expired reminders.
- status set to "fired" after task runs.
- validate_outputs passes.

---

## Phase 54 — Skill Transfer & Analogical Reasoning

### What / Why
Phase 41 (Compositional Generalization) remixes known playbooks.  It does not perform
analogical mapping: "we solved database latency with index tuning — is this API latency
problem structurally similar?  Apply the same abstract solution."
Analogical reasoning is central to human expert problem-solving.

### Files to Create
- `app/agents/analogical_reasoning_agent.py`
- `tests/test_analogical_reasoning_agent.py`

### Agent: `AnalogicalReasoningAgent`
```python
# Inputs:
#   - source_problem: str          — problem description
#   - candidate_analogues: list[dict]  — past solved problems (memory/playbook records)
#   - structural_features: list[str]   — ["latency", "database", "timeout"]
#
# Outputs:
#   - best_analogue: dict | None       — the most structurally similar past problem
#   - analogy_score: float             — structural similarity 0.0–1.0
#   - transferred_solution: str        — abstract solution mapped to current context
#   - mapping: list[dict]              — [{"source_term": "index", "target_term": "cache TTL"}]
#   - confidence: float
#   - novel_elements: list[str]        — features in current problem not in analogue
```

**Heuristic logic:**
1. For each candidate_analogue, compute token Jaccard similarity between its tags/content
   and structural_features of source_problem.
2. `best_analogue` = candidate with highest Jaccard.
3. `analogy_score` = that Jaccard value.
4. `transferred_solution` = best_analogue["solution"] with domain-specific substitutions:
   substitute known domain pairs (e.g. "postgres" → "redis", "index" → "cache key")
   from a static substitution map.
5. `mapping` = list of (source_term, target_term) pairs that were substituted.
6. `novel_elements` = structural_features not found in best_analogue tokens.
7. `confidence` = analogy_score * (1 - len(novel_elements) / max(len(structural_features), 1)).

**Substitution map (static):**
```python
_DOMAIN_SUBSTITUTIONS = {
    "database": "cache", "index": "cache_key", "query": "request",
    "table": "bucket", "row": "entry", "postgres": "redis",
    "mysql": "memcached", "latency": "response_time", "timeout": "ttl",
}
```

**Tests (≥30):**
- Empty candidates → best_analogue=None, score=0.
- Exact match → score=1.0.
- Partial match → score between 0 and 1.
- transferred_solution substitutes known terms.
- novel_elements contains features absent from analogue.
- confidence penalised for novel elements.
- mapping list correct length.
- validate_outputs passes.

---

## Phase 55 — Cognitive Load Balancer

### What / Why
All 50+ agents run at flat priority. There is no mechanism to shed load when the system is
overloaded, or to prioritise high-value tasks over routine enrichment.  A cognitive load
balancer prevents cascading latency under heavy write traffic.

### Files to Create
- `app/services/cognitive_load_balancer.py`
- `app/models/load_snapshot.py`
- `tests/test_cognitive_load_balancer.py`

### Model: `LoadSnapshot`
```python
# Table: load_snapshots
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   queue_depths: dict  (JSONB — {"q.agent_enrich": 120, "q.agent_graph": 45, ...})
#   active_workers: int
#   load_level: str  ("low", "medium", "high", "critical")
#   sampled_at: datetime
```

### Service: `CognitiveLoadBalancer`
```python
# Constants:
#   THRESHOLDS = {"low": 50, "medium": 200, "high": 500}   (total queue depth)
#
# Methods:
#   def classify_load(*, queue_depths: dict[str, int]) -> str
#     — sum all queue depths, compare against thresholds, return level
#
#   def should_skip(*, task_name: str, load_level: str) -> bool
#     — returns True if task should be skipped at this load level
#     — rules:
#         critical: skip all except q.memory_ingest and q.cognitive_loop
#         high: skip q.agent_topics, q.agent_patterns, q.agent_graph
#         medium: skip q.agent_topics only
#         low: skip nothing
#
#   def throttle_factor(*, load_level: str) -> float
#     — returns multiplier for sleep/delay: low=0.0, medium=0.5, high=2.0, critical=5.0
#
#   async def record_snapshot(*, db, org_id, queue_depths, active_workers) -> LoadSnapshot
#     — writes a LoadSnapshot row
#
#   async def get_recent_load(*, db, org_id, limit=10) -> list[LoadSnapshot]
#     — returns recent snapshots ordered by sampled_at desc
```

**Tests (≥25):**
- classify_load: empty dict → "low".
- classify_load: sum=501 → "critical".
- should_skip: "q.agent_topics" at critical → True.
- should_skip: "q.memory_ingest" at critical → False.
- should_skip: "q.agent_topics" at low → False.
- throttle_factor returns correct values per level.
- All four load levels tested for should_skip.
- record_snapshot writes correct load_level.

---

## Phase 56 — Narrative Memory Compression

### What / Why
The memory store grows without bound. Phase 40 (Sleep Cycle) prunes stale/redundant
memories. There is no mechanism to compress a long episodic sequence into a short
retrospective narrative — "Q1 incident log" → 3-sentence summary stored as a new memory,
with the source episodes archived (not deleted).

### Files to Create
- `app/agents/narrative_compression_agent.py`
- `app/services/narrative_compression_service.py`
- `tests/test_narrative_compression_agent.py`

### Agent: `NarrativeCompressionAgent`
```python
# Inputs:
#   - episodes: list[dict]    — ordered list of memory/episode dicts (content, created_at, tags)
#   - topic: str              — unifying topic (e.g. "database incidents Q1")
#   - max_sentences: int      — default 3
#
# Outputs:
#   - compressed_narrative: str    — multi-sentence retrospective
#   - compression_ratio: float     — len(episodes) / max(1, sentence_count)
#   - key_events: list[str]        — most important episode content snippets (≤5)
#   - time_span: dict              — {"from": iso_str, "to": iso_str}
#   - archived_ids: list[str]      — episode IDs that should be archived
#   - confidence: float
```

**Heuristic logic:**
1. Sort episodes by `created_at`.
2. `time_span` = first and last `created_at` as ISO strings.
3. `key_events` = episodes with highest tag overlap with topic tokens (top 5 by score).
4. `compressed_narrative` = template:
   "From {time_span.from} to {time_span.to}, {len(episodes)} events occurred related to
   {topic}. Key events: {key_events[0][:100]}. {key_events[1][:100] if len>1 else ''}
   Overall pattern: {dominant_tag}."
5. `dominant_tag` = most common tag across all episodes.
6. `archived_ids` = episode IDs NOT in key_events (the rest get archived).
7. `compression_ratio` = len(episodes) / max_sentences.
8. `confidence` = min(0.9, 0.5 + 0.05 * len(episodes)).

### Service: `NarrativeCompressionService`
```python
# Methods:
#   async def compress_and_archive(*, db, org_id, user_id, episodes, topic,
#                                  max_sentences=3) -> dict
#     — runs NarrativeCompressionAgent
#     — creates a new Memory row with compressed_narrative as content
#     — updates source episode memories: sets is_archived=True, archived_at=now()
#     — returns {"new_memory_id": str, "archived_count": int, "narrative": str}
```

**Tests (≥30):**
- Single episode → compression_ratio=0.33, key_events has 1 item.
- 10 episodes → archived_ids has 5 non-key items.
- time_span.from < time_span.to.
- dominant_tag is the most frequent tag.
- compressed_narrative contains topic.
- confidence clamped at 0.9 for large episode sets.
- compress_and_archive sets is_archived on source records (mock DB).
- empty episodes list → graceful return with empty fields.
- validate_outputs passes.

---

## Phase 57 — Semantic Change Detection

### What / Why
The system stores knowledge about the world but has no way to detect when a stored fact
has become stale because the real world changed.  A semantic change detector compares
new inbound content against stored memories to flag "this contradicts what we knew".

### Files to Create
- `app/agents/semantic_change_detection_agent.py`
- `tests/test_semantic_change_detection_agent.py`

### Agent: `SemanticChangeDetectionAgent`
```python
# Inputs:
#   - new_content: str           — newly written memory content
#   - existing_memories: list[dict]  — candidate existing memories to compare against
#   - change_threshold: float    — default 0.4 (below this similarity = significant change)
#
# Outputs:
#   - change_detected: bool
#   - changed_memories: list[dict]   — {memory_id, old_content_snippet, similarity, change_type}
#   - change_type: str               — "contradiction", "update", "extension", "unrelated"
#   - semantic_drift_score: float    — 0.0 (no change) to 1.0 (complete reversal)
#   - recommended_action: str        — "supersede" | "flag_review" | "append" | "ignore"
#   - confidence: float
```

**Heuristic logic:**
1. For each existing_memory, compute Jaccard similarity of token sets with new_content.
2. If similarity < change_threshold AND token overlap on _NEGATION_TOKENS is detected
   → change_type = "contradiction", recommended_action = "supersede".
3. If similarity >= change_threshold but < 0.8
   → change_type = "update", recommended_action = "flag_review".
4. If similarity >= 0.8
   → change_type = "extension", recommended_action = "append".
5. `semantic_drift_score` = 1.0 - max(similarity across all memories) (clamped 0–1).
6. `change_detected` = any(similarity < change_threshold).
7. `changed_memories` = memories where similarity < change_threshold.
8. `confidence` = 0.75 if change_detected else 0.9.

**Tests (≥30):**
- New content identical to existing → change_detected=False, semantic_drift_score≈0.
- New content with negations vs existing → change_type="contradiction".
- Low similarity without negation → change_type="update".
- High similarity → change_type="extension".
- recommended_action logic per change_type.
- semantic_drift_score approaches 1.0 for completely different content.
- empty existing_memories → change_detected=False.
- validate_outputs passes.

---

## Phase 58 — Attention-Weighted Memory Retrieval

### What / Why
All memory retrieval uses flat scoring (recency + relevance).  Human cognition weights
retrieval by what the agent is currently "paying attention to" — recent goals, active
incidents, current user intent.  Attention-weighted retrieval surfaces more contextually
relevant memories.

### Files to Create
- `app/services/attention_retrieval_service.py`
- `tests/test_attention_retrieval_service.py`

### Service: `AttentionRetrievalService`
```python
# Constants:
#   GOAL_WEIGHT = 0.4
#   INCIDENT_WEIGHT = 0.3
#   RECENCY_WEIGHT = 0.2
#   BASE_RELEVANCE_WEIGHT = 0.1
#
# Methods:
#   def score(
#       *,
#       memory: dict,
#       active_goals: list[dict],    — list of {goal_id, title, tags}
#       active_incidents: list[dict], — list of {memory_id, content, tags}
#       query_tokens: frozenset[str],
#       now: datetime,
#   ) -> float
#     — Weighted combination:
#       goal_score = max Jaccard(memory.tags, goal.tags) across active_goals
#       incident_score = max Jaccard(memory.tokens, incident.tokens) across active_incidents
#       recency_score = exp(-days_old / 30)  where days_old = (now - memory.created_at).days
#       relevance_score = Jaccard(memory.tokens, query_tokens)
#       total = GOAL_WEIGHT*goal_score + INCIDENT_WEIGHT*incident_score +
#               RECENCY_WEIGHT*recency_score + BASE_RELEVANCE_WEIGHT*relevance_score
#
#   def rank(
#       *,
#       memories: list[dict],
#       active_goals: list[dict],
#       active_incidents: list[dict],
#       query_tokens: frozenset[str],
#       now: datetime,
#       limit: int = 10,
#   ) -> list[dict]
#     — Returns top-`limit` memories sorted by score(memory, ...) descending.
#     — Attaches "_attention_score" key to each returned memory dict.
```

**Tests (≥25):**
- Memory matching active_goal tags scores higher than unrelated memory.
- Memory matching active_incident scores higher than non-incident memory.
- Very old memory (days_old=365) has recency_score≈0.
- Fresh memory (days_old=0) has recency_score=1.0.
- rank returns at most `limit` results.
- rank attaches "_attention_score" to each result.
- Empty active_goals → goal_score=0 for all.
- Empty active_incidents → incident_score=0 for all.
- Score always in [0, 1] range.
- rank is stable (deterministic for same inputs).

---

## Phase 59 — Self-Supervised Concept Learning

### What / Why
The system detects topics (Phase 21) but concepts are not learned — the agent cannot
identify that "database latency" and "slow queries" and "connection pool exhaustion" are
all instances of the same concept cluster.  Self-supervised concept learning builds an
emergent concept vocabulary from memory patterns without requiring labeled data.

### Files to Create
- `app/agents/concept_learning_agent.py`
- `app/models/learned_concept.py`
- `app/services/concept_registry_service.py`
- `tests/test_concept_learning_agent.py`

### Model: `LearnedConcept`
```python
# Table: learned_concepts
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   concept_name: str              — e.g. "database_latency_cluster"
#   member_memory_ids: list[str]   (JSONB array)
#   canonical_terms: list[str]     (JSONB array — most frequent tokens)
#   occurrence_count: int
#   first_seen: datetime
#   last_seen: datetime
#   confidence: float
```

### Agent: `ConceptLearningAgent`
```python
# Inputs:
#   - memories: list[dict]        — pool of memories to cluster
#   - existing_concepts: list[dict] — already learned concepts for this org
#   - min_cluster_size: int       — default 3
#
# Outputs:
#   - new_concepts: list[dict]    — each: {concept_name, member_ids, canonical_terms, confidence}
#   - updated_concepts: list[dict] — existing concepts that gained new members
#   - noise_memories: list[str]   — memory IDs not assigned to any concept
#   - total_concepts_found: int
#   - confidence: float
```

**Heuristic logic:**
1. Build token sets for each memory (same tokenizer as pre_write_noise_gate).
2. Simple greedy clustering: for each memory, find existing concept whose canonical_terms
   have Jaccard >= 0.3 with this memory's tokens.  Assign to best match.
3. Memories not assigned → collect into "pending" pool.
4. In pending pool, find pairs with Jaccard >= 0.3.  Build connected components via
   union-find.  Components with size >= min_cluster_size → new concepts.
5. `concept_name` = top-2 most common tokens joined with "_".
6. `canonical_terms` = top-5 most common tokens across cluster members.
7. `confidence` = avg Jaccard within cluster (intra-cluster cohesion).
8. `noise_memories` = pending memories not in any component >= min_cluster_size.

### Service: `ConceptRegistryService`
```python
# Methods:
#   async def upsert_concepts(*, db, org_id, new_concepts, updated_concepts) -> int
#     — inserts new LearnedConcept rows, updates existing ones
#     — returns total rows affected
#   async def get_concepts_for_org(*, db, org_id, limit=50) -> list[LearnedConcept]
#   async def find_concept_for_memory(*, db, org_id, memory_id) -> LearnedConcept | None
```

**Tests (≥35):**
- 3 similar memories → 1 new concept, 0 noise_memories.
- 2 similar + 1 dissimilar → 1 concept + 1 noise.
- existing concept absorbs new member (updated_concepts non-empty).
- concept_name is top tokens joined with "_".
- canonical_terms has ≤5 items.
- min_cluster_size=5, only 3 matching → no new concept, all in noise.
- ConceptRegistryService.upsert_concepts inserts correct count (mock session).
- validate_outputs passes.

---

## Phase 60 — Recursive Self-Improvement Planner

### What / Why
The system can model itself (Phase 36, MetaCognitivePlanningAgent) and track strategy
outcomes (Phase 50).  It cannot yet propose improvements to its own agent configuration
based on observed performance gaps — "I fail at causal reasoning when the graph has >50
nodes; I should request a pruned subgraph first."

### Files to Create
- `app/agents/self_improvement_planner_agent.py`
- `app/models/improvement_proposal.py`
- `tests/test_self_improvement_planner_agent.py`

### Model: `ImprovementProposal`
```python
# Table: improvement_proposals
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   target_agent: str              — e.g. "CausalReasoningAgent"
#   proposal_type: str             — "parameter_tune" | "data_preprocessing" | "routing_change"
#   description: str
#   evidence: list[dict]           (JSONB — failure records that motivated this)
#   expected_gain: float           — estimated confidence/accuracy improvement
#   status: str                    — "proposed" | "accepted" | "rejected" | "implemented"
#   created_at: datetime
```

### Agent: `SelfImprovementPlannerAgent`
```python
# Inputs:
#   - performance_metrics: list[dict]   — {agent_name, avg_confidence, failure_rate, sample_count}
#   - failure_records: list[dict]       — {agent_name, input_summary, error_type, timestamp}
#   - current_config: dict             — current agent parameter settings
#   - improvement_threshold: float     — default 0.15 (only propose if gain >= this)
#
# Outputs:
#   - proposals: list[dict]            — each: {target_agent, proposal_type, description,
#                                          expected_gain, evidence_count}
#   - high_priority_proposals: list[dict]  — proposals with expected_gain >= 0.3
#   - system_health_score: float       — 1.0 - avg(failure_rates)
#   - confidence: float
```

**Heuristic logic:**
1. For each agent in performance_metrics where `failure_rate > 0.2`:
   - Count matching failure_records.
   - If error_type contains "timeout" or "graph_too_large" → proposal_type="data_preprocessing",
     description="pre-filter input to reduce size before agent run".
   - If error_type contains "low_confidence" → proposal_type="parameter_tune",
     description="lower confidence threshold or increase evidence limit".
   - Else → proposal_type="routing_change",
     description="route to human review when confidence < 0.4".
   - expected_gain = min(0.5, failure_rate * 1.5).
2. Filter out proposals where expected_gain < improvement_threshold.
3. `high_priority_proposals` = proposals where expected_gain >= 0.3.
4. `system_health_score` = 1.0 - mean(failure_rates) clamped to [0, 1].
5. `confidence` = 0.7 if len(proposals) > 0 else 0.5.

**Tests (≥30):**
- Agent with failure_rate=0.0 → no proposal for that agent.
- Agent with timeout errors → proposal_type="data_preprocessing".
- Agent with low_confidence errors → proposal_type="parameter_tune".
- expected_gain clamped at 0.5.
- improvement_threshold filters out low-gain proposals.
- high_priority_proposals subset of proposals.
- system_health_score = 1.0 when all failure_rates=0.
- system_health_score close to 0 when all rates=1.0.
- validate_outputs passes.

---

## Implementation Order & Dependencies

```
Phase 51 (Counterfactual)      — no deps
Phase 52 (Persona Engine)      — no deps, needs new DB model
Phase 53 (Prospective Memory)  — no deps, needs new DB model + Celery task
Phase 54 (Analogical Reasoning)— no deps
Phase 55 (Load Balancer)       — no deps, needs new DB model
Phase 56 (Narrative Compression)— depends on Phase 23 (Narrative Synthesis patterns)
Phase 57 (Semantic Change)     — no deps
Phase 58 (Attention Retrieval) — depends on Phase 10 (OrgAttentionAgent patterns)
Phase 59 (Concept Learning)    — depends on Phase 21 (GoalDecomposition token patterns)
Phase 60 (Self-Improvement)    — depends on Phase 36 + Phase 50 metrics
```

Implement phases in order 51 → 60.  Do not skip a phase.

---

## Git Workflow (repeat for each phase)

```bash
# 1. From main, create branch
git checkout main && git pull
git checkout -b agi/phase-{N}-{short-slug}

# 2. Implement the phase (agent + model + service + task as specified above)

# 3. Run tests — must all pass
python -m pytest tests/test_{agent_file}.py -x -q
python -m pytest tests/ -x -q  # full suite

# 4. Stage only this phase's files
git add app/agents/{agent_file}.py \
        app/services/{service_file}.py \   # if applicable
        app/models/{model_file}.py \        # if applicable
        app/tasks/{task_file}.py \          # if applicable
        tests/test_{agent_file}.py \
        CLAUDE.md

# 5. Commit
git commit -m "feat: phase {N} — {phase name}"

# 6. Push + PR + merge
git push -u origin agi/phase-{N}-{short-slug}
gh pr create --title "feat: phase {N} — {Phase Name}" \
  --body "..." --base main --head agi/phase-{N}-{short-slug}
gh pr merge --admin --squash --delete-branch
```

---

## CLAUDE.md Update Template

After each phase merges, add a row to the Phase Status table in CLAUDE.md:

```
| {N} | {Phase Name} | Done ({AgentName}, {N_tests} tests, {total} total passing) |
```

Update "total passing" by running `python -m pytest tests/ -q` after the full suite and
capturing the count from the summary line.

---

## Test Conventions

All test files must follow these patterns (same as existing test files):

```python
import pytest
from unittest.mock import MagicMock, patch

# Mark all tests so they run without DB/LLM
pytestmark = pytest.mark.unit  # or no mark — just no asyncio needed for pure services

class TestAgentNameHeuristic:
    def _agent(self):
        return AgentName()   # no constructor args for pure agents

    def test_basic_case(self):
        agent = self._agent()
        result = agent.run(AgentInput(...), enrichment={...})
        assert result.extras["key"] == expected_value

    def test_validate_outputs(self):
        agent = self._agent()
        result = agent.run(AgentInput(...), enrichment={...})
        agent.validate_outputs(result)  # must not raise
```

Service tests use `MagicMock` for DB session:
```python
class TestSomeService:
    def _service(self):
        db = MagicMock()
        db.execute = AsyncMock(return_value=MagicMock(scalar_one_or_none=lambda: None))
        db.flush = AsyncMock()
        return SomeService(session=db), db
```

Pure (non-async) services are tested synchronously — no `pytest.mark.asyncio` needed.

---

*Last updated: 2026-04-01*
*Covers phases 51–60.*
