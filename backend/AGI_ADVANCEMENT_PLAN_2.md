# AGI Advancement Plan — Ninai Backend — Phases 61–80

Continuation of AGI_ADVANCEMENT_PLAN.md (phases 51–60).
Same rules apply: implement in order, tests green before next phase, no AI attribution.

**Run tests after every phase:**
```bash
python -m pytest tests/test_{new_file}.py -x -q
python -m pytest tests/ -x -q
```

---

## Phase 61 — Working Memory Manager

### What / Why
The system has long-term memory (Qdrant + Postgres) and no concept of a bounded
active-context buffer.  Human cognition uses working memory — a small, fast, capacity-
limited scratchpad that holds the items currently "in play".  Without it, every reasoning
step re-queries all of long-term memory. This service maintains a per-session working
memory buffer (capacity=20), evicts least-recently-used items, and provides the cognitive
loop with a fast, pre-loaded context rather than repeated full retrievals.

### Files to Create
- `app/services/working_memory_service.py`
- `app/models/working_memory_item.py`
- `tests/test_working_memory_service.py`

### Model: `WorkingMemoryItem`
```python
# Table: working_memory_items
# Columns:
#   id: UUID PK
#   session_id: UUID  (cognitive_session FK, indexed)
#   org_id: UUID FK
#   memory_id: UUID FK (nullable — long-term memory reference)
#   content_snapshot: str   (denormalised short copy, max 512 chars)
#   item_type: str  ("memory", "goal", "hypothesis", "plan_step", "observation")
#   activation: float  default 1.0  (decays on eviction pressure)
#   inserted_at: datetime
#   last_accessed_at: datetime
```

### Service: `WorkingMemoryService`
```python
# Constants:
#   CAPACITY = 20
#   DECAY_FACTOR = 0.85   # applied to activation on each tick

# Methods:
#   async def push(*, db, session_id, org_id, item: dict) -> WorkingMemoryItem
#     — adds item to working memory for this session
#     — if len >= CAPACITY: evict item with lowest activation * recency_score
#     — recency_score = 1 / (1 + seconds_since_last_accessed)
#
#   async def access(*, db, session_id, item_id) -> WorkingMemoryItem | None
#     — marks item as accessed: last_accessed_at=now(), activation=min(1.0, activation+0.1)
#
#   async def tick_decay(*, db, session_id) -> int
#     — multiplies all activations by DECAY_FACTOR
#     — removes items with activation < 0.05
#     — returns count removed
#
#   async def snapshot(*, db, session_id) -> list[WorkingMemoryItem]
#     — returns all items for session ordered by activation desc
#
#   async def flush(*, db, session_id) -> int
#     — clears all working memory for this session, returns count cleared
#
#   def eviction_score(item: WorkingMemoryItem, now: datetime) -> float
#     — pure function: activation * recency_score (for testing without DB)
#     — recency_score = exp(-seconds_since_last_accessed / 300)
```

### Tests (≥30)
- push: capacity not exceeded when under limit.
- push: evicts lowest-activation item when at CAPACITY.
- access: increments activation, updates last_accessed_at.
- tick_decay: activation < 0.05 items removed.
- tick_decay: remaining items have activation *= DECAY_FACTOR.
- snapshot: ordered by activation desc.
- flush: returns correct count, snapshot empty after.
- eviction_score: higher activation = higher score.
- eviction_score: recently accessed = higher score.
- eviction_score: old unaccessed item has score near 0.
- push with item_type="goal" stored correctly.
- content_snapshot truncated to 512 chars.

---

## Phase 62 — Temporal Pattern Miner

### What / Why
The system stores events with timestamps but never mines recurring temporal patterns:
"database alerts always spike on Monday mornings", "deploy failures cluster between
18:00–20:00 UTC".  Temporal patterns enable proactive alerts and predictive scheduling.

### Files to Create
- `app/agents/temporal_pattern_miner_agent.py`
- `app/models/temporal_pattern.py`
- `tests/test_temporal_pattern_miner_agent.py`

### Model: `TemporalPattern`
```python
# Table: temporal_patterns
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   pattern_type: str   ("hour_of_day", "day_of_week", "day_of_month", "interval")
#   pattern_key: str    — e.g. "monday_09" or "hour_18" or "every_7_days"
#   topic_tags: list[str]  (JSONB)
#   occurrence_count: int
#   avg_severity: float
#   first_seen: datetime
#   last_seen: datetime
#   confidence: float
```

### Agent: `TemporalPatternMinerAgent`
```python
# Inputs (via enrichment dict):
#   - memories: list[dict]    — each has: content, tags, created_at (ISO string), severity
#   - analysis_window_days: int  — default 90
#   - min_occurrences: int       — default 3
#
# Outputs (extras dict):
#   - patterns: list[dict]     — each: {pattern_type, pattern_key, topic_tags,
#                                occurrence_count, avg_severity, confidence}
#   - dominant_pattern: dict | None   — highest occurrence_count pattern
#   - anomalous_times: list[str]      — hour/day slots with z_score > 2.0
#   - total_events_analysed: int
#   - confidence: float
```

**Heuristic logic:**
1. Parse `created_at` for each memory. Build frequency histograms:
   - `hour_counts[hour_of_day]` = list of memory indices
   - `dow_counts[day_of_week]` = list of memory indices  (0=Monday)
2. For each bucket with count >= min_occurrences:
   - `confidence` = min(0.95, occurrence_count / (analysis_window_days / 7))
   - `avg_severity` = mean of memory["severity"] values in bucket (default 0.5 if absent)
   - `topic_tags` = top-3 tags by frequency across bucket members
3. `anomalous_times`: compute mean and std of all hour_counts values.
   Slots where count > mean + 2*std → anomalous.
4. `dominant_pattern` = pattern dict with highest occurrence_count.
5. Top-level `confidence` = mean of per-pattern confidences (0.5 if no patterns).

**Tests (≥30):**
- 5 memories all on Monday → dominant_pattern.pattern_key = "dow_0".
- Memories spread across all days → no dominant_pattern (all counts equal).
- anomalous_times: one hour with 10x normal count → in anomalous_times.
- avg_severity computed correctly.
- topic_tags = top-3 most common tags in bucket.
- min_occurrences=5, only 3 occurrences → no pattern emitted.
- analysis_window_days affects confidence calculation.
- Empty memories list → patterns=[], confidence=0.5.
- validate_outputs passes.

---

## Phase 63 — Active Knowledge Seeker

### What / Why
The system answers questions passively.  It never identifies gaps in its own knowledge
and asks for the missing information.  Active knowledge seeking — detecting "I don't know
X, and X is critical for this goal" — is necessary for autonomous operation.

### Files to Create
- `app/agents/active_knowledge_seeker_agent.py`
- `app/models/knowledge_gap.py`
- `tests/test_active_knowledge_seeker_agent.py`

### Model: `KnowledgeGap`
```python
# Table: knowledge_gaps
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   goal_id: UUID FK (nullable)
#   gap_description: str
#   question_to_ask: str
#   required_for: str    — which goal/decision this is blocking
#   priority: str        — "critical" | "high" | "medium" | "low"
#   status: str          — "open" | "resolved" | "waived"
#   created_at: datetime
#   resolved_at: datetime (nullable)
```

### Agent: `ActiveKnowledgeSeekerAgent`
```python
# Inputs:
#   - goal: str                        — current goal text
#   - available_memories: list[dict]   — what the system already knows
#   - required_entities: list[str]     — entities the goal references
#   - confidence_threshold: float      — default 0.4 (below = gap)
#
# Outputs:
#   - knowledge_gaps: list[dict]   — each: {gap_description, question_to_ask,
#                                     required_for, priority, coverage_score}
#   - coverage_score: float        — fraction of required_entities covered by memories
#   - is_sufficient: bool          — True if coverage_score >= confidence_threshold
#   - top_question: str | None     — single most important question to ask
#   - confidence: float
```

**Heuristic logic:**
1. For each entity in `required_entities`:
   - Check if any available_memory contains the entity (case-insensitive token match).
   - If not found: create gap entry with:
     - `gap_description` = f"No information found about '{entity}'"
     - `question_to_ask` = f"What is the current status of {entity}?"
     - `priority` = "critical" if entity appears in goal text else "high"
2. `coverage_score` = covered_entities / max(1, len(required_entities)).
3. `is_sufficient` = coverage_score >= confidence_threshold.
4. `top_question` = question from the first "critical" gap, else first "high" gap.
5. `confidence` = 0.85 — gaps detected reliably via token matching.

**Tests (≥30):**
- All entities covered → knowledge_gaps=[], is_sufficient=True.
- Missing entity → gap entry created with correct question.
- Entity in goal text → priority="critical".
- Entity not in goal text → priority="high".
- coverage_score: 2 of 4 entities covered → 0.5.
- is_sufficient=False when coverage < threshold.
- top_question is the critical-priority question.
- Empty required_entities → coverage_score=1.0, is_sufficient=True.
- validate_outputs passes.

---

## Phase 64 — Uncertainty Propagation Engine

### What / Why
Phase 22 (Uncertainty Reporting) flags uncertain memories in isolation.  When memory A
(uncertain) is used as evidence for conclusion B, that uncertainty is not propagated —
B appears confident despite being built on shaky foundations.  This engine tracks and
propagates uncertainty through inference chains.

### Files to Create
- `app/services/uncertainty_propagation_service.py`
- `tests/test_uncertainty_propagation_service.py`

### Service: `UncertaintyPropagationService`
```python
# Constants:
#   PROPAGATION_DECAY = 0.7   # uncertainty attenuates with each hop
#   MIN_PROPAGATED = 0.05     # below this, uncertainty is considered resolved
#
# Methods:
#   def propagate(
#       *,
#       source_uncertainty: float,          # 0.0=certain, 1.0=fully uncertain
#       inference_hops: int,                # how many steps from source
#       corroborating_evidence_count: int,  # each piece reduces uncertainty
#   ) -> float
#     — formula:
#         attenuated = source_uncertainty * (PROPAGATION_DECAY ** inference_hops)
#         corroboration_factor = 1.0 / (1.0 + 0.2 * corroborating_evidence_count)
#         return max(MIN_PROPAGATED, attenuated * corroboration_factor)
#
#   def propagate_chain(
#       *,
#       sources: list[dict],   # each: {memory_id, uncertainty}
#       chain_length: int,
#   ) -> float
#     — combined uncertainty = geometric mean of per-source propagated values
#
#   def uncertainty_label(uncertainty: float) -> str
#     — <0.1 → "certain", <0.3 → "likely", <0.6 → "uncertain", >=0.6 → "speculative"
#
#   def should_flag_for_review(
#       *,
#       propagated_uncertainty: float,
#       decision_stakes: str,   # "low" | "medium" | "high" | "critical"
#   ) -> bool
#     — critical: flag if > 0.1
#     — high: flag if > 0.25
#     — medium: flag if > 0.5
#     — low: flag if > 0.75
```

**Tests (≥30):**
- 0 hops, 0 corroboration → propagated = source_uncertainty.
- 3 hops → attenuated by DECAY^3.
- Corroborating evidence reduces propagated uncertainty.
- propagate_chain: geometric mean of sources.
- uncertainty_label: 0.05 → "certain", 0.5 → "uncertain".
- should_flag_for_review: critical stakes flags at 0.11.
- should_flag_for_review: low stakes ignores 0.6.
- MIN_PROPAGATED floor applied.
- chain with all-certain sources → near-certain result.
- chain with one highly-uncertain source → propagated > 0.1.

---

## Phase 65 — Hierarchical Goal Planner

### What / Why
Phase 21 (GoalDecompositionAgent) breaks one goal into flat steps.  Real planning is
hierarchical: a high-level goal decomposes into sub-goals, each of which decomposes
further. Without hierarchy, planning depth is limited to 1 level.

### Files to Create
- `app/agents/hierarchical_goal_planner_agent.py`
- `app/models/goal_hierarchy_node.py`
- `tests/test_hierarchical_goal_planner_agent.py`

### Model: `GoalHierarchyNode`
```python
# Table: goal_hierarchy_nodes
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   root_goal_id: UUID FK  (top-level goal)
#   parent_node_id: UUID FK (nullable — null for root)
#   goal_id: UUID FK (nullable — links to existing Goal if persisted)
#   title: str
#   depth: int    (0=root, 1=sub-goal, 2=task, 3=action)
#   status: str   ("pending", "in_progress", "done", "blocked")
#   estimated_effort: str   ("trivial", "small", "medium", "large")
#   created_at: datetime
```

### Agent: `HierarchicalGoalPlannerAgent`
```python
# Inputs:
#   - root_goal: str           — high-level goal
#   - depth_limit: int         — default 3 (0=root only, 3=root+3 levels)
#   - domain_context: str      — optional domain hint ("infrastructure", "engineering")
#
# Outputs:
#   - hierarchy: list[dict]    — tree as flat list with {title, depth, parent_index,
#                                estimated_effort, status}
#   - total_nodes: int
#   - max_depth_reached: int
#   - critical_path: list[str] — titles of nodes on the critical (longest) path
#   - leaf_tasks: list[str]    — titles of leaf nodes (no children)
#   - confidence: float
```

**Heuristic logic (keyword-driven decomposition):**

```
Decomposition rules by depth:
  depth=0 (root): use the root_goal as-is.
  depth=1 (sub-goals): split by detecting conjunctions ("and", "then", "after",
    "also", "as well as") in root_goal. If none found, generate 2 default sub-goals:
      "Gather information for: {root_goal}"
      "Execute plan for: {root_goal}"
  depth=2 (tasks): for each sub-goal, generate:
      "Verify preconditions for: {sub_goal_title}"
      "Perform: {sub_goal_title}"
      "Validate outcome of: {sub_goal_title}"
  depth=3 (actions): for each task, generate:
      "Start: {task_title}"
      "Complete: {task_title}"
```

- `estimated_effort`: root="large", depth=1="medium", depth=2="small", depth=3="trivial".
- `critical_path`: longest path from root to deepest leaf (titles only).
- `leaf_tasks`: nodes with no children.
- `confidence` = 0.7 for heuristic decomposition.

**Tests (≥35):**
- root goal with "and" → 2 sub-goals at depth=1.
- root goal without conjunctions → 2 default sub-goals.
- depth_limit=0 → only root node returned.
- depth_limit=2 → max_depth_reached=2.
- leaf_tasks contains only depth-2 nodes when depth_limit=2.
- critical_path starts with root_goal title.
- total_nodes = sum of nodes across all depths.
- estimated_effort correct per depth.
- All hierarchy nodes have parent_index set (except root at 0).
- validate_outputs passes.

---

## Phase 66 — Social Memory & Team Dynamics Agent

### What / Why
The system stores individual memories but has no model of social dynamics: who
collaborates with whom, which teams have communication gaps, which individuals are
knowledge silos. Social memory surfaces team health signals for managers.

### Files to Create
- `app/agents/social_memory_agent.py`
- `app/models/social_graph_edge.py`
- `tests/test_social_memory_agent.py`

### Model: `SocialGraphEdge`
```python
# Table: social_graph_edges
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   actor_user_id: UUID FK
#   collaborator_user_id: UUID FK
#   interaction_type: str   ("co_authored", "reviewed", "referenced", "escalated_to")
#   interaction_count: int  default 1
#   last_interaction: datetime
#   strength: float   (EMA of interaction_count, α=0.2)
```

### Agent: `SocialMemoryAgent`
```python
# Inputs (enrichment dict):
#   - memories: list[dict]    — each: {user_id, tags, created_at, linked_user_ids: list}
#   - org_users: list[str]    — all user IDs in org
#
# Outputs:
#   - collaboration_edges: list[dict]   — {actor, collaborator, interaction_count, strength}
#   - knowledge_silos: list[str]        — user_ids with 0 collaborators
#   - most_connected: str | None        — user_id with most edges
#   - least_connected: str | None       — user_id with fewest edges (non-zero)
#   - team_cohesion_score: float        — actual_edges / max_possible_edges
#   - confidence: float
```

**Heuristic logic:**
1. For each memory with `linked_user_ids`, emit an edge between `user_id` and each
   `linked_user_id` (interaction_type="co_authored").
2. Count edges per (actor, collaborator) pair.
3. `strength` = interaction_count / max(1, total_interactions_for_actor) clamped [0,1].
4. `knowledge_silos` = org_users with no edges in or out.
5. `most_connected` = user with highest degree (unique collaborators).
6. `least_connected` = user with lowest non-zero degree.
7. `team_cohesion_score` = actual_edge_count / (n*(n-1)/2) where n=len(org_users).
8. `confidence` = 0.8.

**Tests (≥30):**
- Two memories with shared linked_user_ids → collaboration edge created.
- User with no linked memories in any memory → in knowledge_silos.
- most_connected = user appearing most in linked_user_ids.
- team_cohesion_score: all users connected → 1.0.
- team_cohesion_score: no connections → 0.0.
- strength bounded [0, 1].
- org_users=[] → graceful empty result.
- validate_outputs passes.

---

## Phase 67 — Episodic Future Simulation

### What / Why
Phase 51 (Counterfactual) reasons about past alternatives.  This phase adds forward
simulation: given a current state and a planned action, simulate what episode sequence
is likely to follow — enabling the agent to "rehearse" a plan before executing it.

### Files to Create
- `app/agents/episodic_future_simulation_agent.py`
- `tests/test_episodic_future_simulation_agent.py`

### Agent: `EpisodicFutureSimulationAgent`
```python
# Inputs:
#   - current_state: dict      — {entities: list[str], active_incidents: list[str],
#                                 open_goals: list[str], severity_level: str}
#   - planned_action: str      — action the agent intends to take
#   - historical_episodes: list[dict]  — past episodes to draw patterns from
#   - simulation_steps: int    — default 3 (how many future events to project)
#
# Outputs:
#   - simulated_episodes: list[dict]   — ordered projected future events:
#                                         {step, event_description, probability,
#                                          severity_change, entities_affected}
#   - success_probability: float       — P(planned_action achieves goal)
#   - risk_events: list[dict]          — simulated episodes with probability > 0.5
#                                         AND severity_change = "increase"
#   - recommended_precautions: list[str]
#   - confidence: float
```

**Heuristic logic:**
1. Find historical_episodes where content token-overlaps with planned_action >= 0.2.
2. For each matched historical episode, extract the next episode in sequence
   (if available — use index+1).  These are "likely sequels".
3. Deduplicate sequel templates by dominant tag.
4. For simulation_steps steps:
   - step 0: the planned_action itself, probability=0.9 (agent executes it).
   - step 1..N: pick top-matching sequel template; probability = base_prob * (0.8^step).
   - severity_change: "increase" if sequel contains severity-escalation tokens
     ["critical", "alert", "failure", "down", "error"], else "stable", else "decrease".
5. `success_probability` = probability of step 0 * (1 - max_risk_probability).
6. `risk_events` = simulated steps with severity_change="increase" AND probability > 0.5.
7. `recommended_precautions`: for each risk_event, "Monitor {entities_affected} closely
   before proceeding." Add "Roll back if {risk_event.event_description[:50]} occurs."
8. `confidence` = 0.6 (simulation is inherently uncertain).

**Tests (≥30):**
- No matching historical episodes → simulated_episodes has generic steps only.
- Historical match found → sequel template used for step 1.
- Probability decreases by 0.8^step each step.
- risk_events only includes severity_change="increase" AND probability>0.5.
- success_probability reduced by high-probability risk.
- recommended_precautions non-empty when risk_events non-empty.
- simulation_steps=1 → only step 0 in result.
- confidence=0.6 always.
- validate_outputs passes.

---

## Phase 68 — Error Recovery & Replan Agent

### What / Why
Phase 47 (AutonomousActionAgent) executes plans and retries on failure.  There is no
intelligent replan: "step 3 failed because the target service is down — skip and route
around it."  Error recovery reasoning enables graceful degradation rather than hard stops.

### Files to Create
- `app/agents/error_recovery_agent.py`
- `tests/test_error_recovery_agent.py`

### Agent: `ErrorRecoveryAgent`
```python
# Inputs:
#   - failed_step: dict    — {step_index, title, error_type, error_message, attempts}
#   - remaining_plan: list[dict]  — steps not yet executed
#   - completed_steps: list[dict] — steps already done
#   - available_tools: list[str]  — tools the executor has access to
#
# Outputs:
#   - recovery_strategy: str   — "retry" | "skip" | "substitute" | "replan" | "escalate"
#   - revised_plan: list[dict] — updated remaining steps after recovery decision
#   - substitute_step: dict | None   — replacement step if strategy="substitute"
#   - skip_justification: str | None
#   - escalation_reason: str | None
#   - confidence: float
```

**Recovery decision rules (in priority order):**
```
1. error_type = "transient" AND attempts < 3 → recovery_strategy = "retry"
   revised_plan = [failed_step (reset attempts)] + remaining_plan

2. error_type = "not_found" OR "permission_denied" AND attempts >= 1 →
   recovery_strategy = "skip"
   skip_justification = f"Step '{failed_step.title}' skipped: {error_type}"
   revised_plan = remaining_plan (unchanged)

3. error_type = "service_unavailable" →
   recovery_strategy = "substitute"
   substitute_step = {
     title: f"[SUBSTITUTE] {failed_step.title} via fallback",
     tool: fallback_tool (first available_tool not mentioned in failed_step.title),
     step_index: failed_step.step_index
   }
   revised_plan = [substitute_step] + remaining_plan

4. error_type = "data_corruption" OR attempts >= 5 →
   recovery_strategy = "replan"
   revised_plan = []  (caller must re-decompose goal)

5. Default (unknown error, attempts >= 3) →
   recovery_strategy = "escalate"
   escalation_reason = f"Unrecoverable: {error_type} after {attempts} attempts"
   revised_plan = remaining_plan
```

- `confidence`: retry=0.8, skip=0.75, substitute=0.65, replan=0.5, escalate=0.4.

**Tests (≥30):**
- transient error, attempts=1 → strategy="retry", failed_step back in revised_plan.
- transient error, attempts=3 → not retry (falls to later rule).
- not_found error → strategy="skip", original remaining_plan returned.
- service_unavailable → strategy="substitute", substitute_step present.
- data_corruption → strategy="replan", revised_plan empty.
- attempts >= 5, unknown error → strategy="escalate".
- confidence correct per strategy.
- substitute_step uses first available_tool.
- skip_justification contains error_type.
- validate_outputs passes.

---

## Phase 69 — Semantic Role Inference Agent

### What / Why
The system knows what happened but not who plays what role in the organisation.  Semantic
role inference learns from memory patterns: "Alice always deploys", "Bob owns incident
reviews", "team-infra handles database issues".  This enables smarter routing, assignment
suggestions, and accountability tracking.

### Files to Create
- `app/agents/semantic_role_inference_agent.py`
- `app/models/inferred_role.py`
- `tests/test_semantic_role_inference_agent.py`

### Model: `InferredRole`
```python
# Table: inferred_roles
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   entity_id: str   (user_id or team_id)
#   entity_type: str  ("user" | "team")
#   role_label: str   ("deployer", "reviewer", "incident_owner", "architect",
#                      "approver", "escalation_target")
#   evidence_count: int
#   confidence: float
#   last_updated: datetime
```

### Agent: `SemanticRoleInferenceAgent`
```python
# Inputs:
#   - memories: list[dict]   — each: {user_id, content, tags, action_type}
#   - existing_roles: list[dict]  — already inferred roles for this org
#
# Outputs:
#   - inferred_roles: list[dict]   — each: {entity_id, entity_type, role_label,
#                                     evidence_count, confidence}
#   - role_coverage: float         — fraction of users with at least 1 inferred role
#   - conflicts: list[dict]        — users with contradictory role signals
#   - confidence: float
```

**Role detection rules (keyword-based on memory.content + memory.tags):**
```python
_ROLE_SIGNALS = {
    "deployer":          ["deploy", "release", "rollout", "push", "ship"],
    "reviewer":          ["review", "approve", "lgtm", "merge", "PR"],
    "incident_owner":    ["incident", "postmortem", "RCA", "on-call", "pager"],
    "architect":         ["design", "architecture", "RFC", "proposal", "schema"],
    "approver":          ["approved", "sign-off", "authorized", "granted"],
    "escalation_target": ["escalate", "escalated to", "notify", "alert"],
}
```

For each memory:
1. Tokenize content. Check overlap with each role's signal list.
2. If overlap >= 1: record evidence for (memory.user_id, role_label).
3. `evidence_count` = number of memories supporting this role for this user.
4. `confidence` = min(0.95, 0.4 + 0.1 * evidence_count).
5. `conflicts` = users where evidence exists for both "deployer" AND "reviewer"
   with evidence_count >= 2 each (suggests ambiguous signal).
6. `role_coverage` = users_with_any_role / max(1, unique_users_in_memories).

**Tests (≥30):**
- Memory with "deploy" in content → user inferred as "deployer".
- 3 deploy memories → confidence = 0.7.
- User with "review" and "deploy" both >= 2 evidence → in conflicts.
- role_coverage = 0.5 when half users have inferred roles.
- Empty memories → inferred_roles=[], role_coverage=0.0.
- existing_roles used for evidence_count baseline.
- confidence clamped at 0.95.
- validate_outputs passes.

---

## Phase 70 — Confidence Ensemble Service

### What / Why
Multiple agents independently score the same memory (credibility, anomaly, uncertainty,
hypothesis).  There is no mechanism to combine these into a single calibrated confidence
score.  Ensembling reduces individual agent biases and produces a more reliable signal.

### Files to Create
- `app/services/confidence_ensemble_service.py`
- `tests/test_confidence_ensemble_service.py`

### Service: `ConfidenceEnsembleService`
```python
# Constants — weights per signal source (must sum to 1.0):
#   WEIGHTS = {
#       "credibility":    0.30,
#       "anomaly":        0.20,
#       "uncertainty":    0.25,
#       "hypothesis":     0.15,
#       "calibration":    0.10,
#   }
#
# Methods:
#   def ensemble(*, signals: dict[str, float]) -> float
#     — signals: e.g. {"credibility": 0.9, "anomaly": 0.1, "uncertainty": 0.3}
#     — for anomaly: inverted (anomaly=1.0 is BAD, contribute 0.0 to confidence)
#     — for uncertainty: inverted (high uncertainty = low confidence)
#     — credibility, hypothesis, calibration: used directly as confidence contribution
#     — missing signals default to 0.5 (neutral)
#     — formula:
#         score = sum(WEIGHTS[k] * adjusted_signal[k] for k in WEIGHTS)
#         return round(clamp(score, 0.0, 1.0), 4)
#
#   def ensemble_label(score: float) -> str
#     — >=0.8 → "high_confidence"
#     — >=0.6 → "moderate_confidence"
#     — >=0.4 → "low_confidence"
#     — <0.4  → "unreliable"
#
#   def missing_signal_impact(*, missing_key: str) -> float
#     — returns WEIGHTS[missing_key] — the maximum uncertainty from this missing signal
#
#   def dominant_signal(*, signals: dict[str, float]) -> str
#     — returns the signal key with the highest weighted contribution
```

**Tests (≥25):**
- All signals = 0.5 → ensemble ≈ 0.5.
- credibility=1.0, all others=0.5 → score > 0.5.
- anomaly=1.0 (fully anomalous) → anomaly contribution = 0 (inverted).
- uncertainty=1.0 (fully uncertain) → uncertainty contribution = 0.
- Missing signal → defaults to 0.5.
- ensemble_label: 0.85 → "high_confidence".
- ensemble_label: 0.35 → "unreliable".
- dominant_signal: highest weighted signal returned.
- missing_signal_impact("credibility") = 0.30.
- Score clamped between 0.0 and 1.0.

---

## Phase 71 — Memory Importance Ranker

### What / Why
Phase 58 (Attention-Weighted Retrieval) weights memories by attention context.
There is no global "importance" score that persists and is updated over time — telling
the system which memories are intrinsically valuable independent of the current query.

### Files to Create
- `app/services/memory_importance_ranker.py`
- `tests/test_memory_importance_ranker.py`

### Service: `MemoryImportanceRanker`
```python
# Importance components and weights:
#   REFERENCE_WEIGHT    = 0.30  — how often other memories reference this one
#   GOAL_LINK_WEIGHT    = 0.25  — how many goals this memory is linked to
#   RECENCY_WEIGHT      = 0.20  — recency decay (half-life=30 days)
#   CREDIBILITY_WEIGHT  = 0.15  — credibility score of the memory
#   ACTIVATION_WEIGHT   = 0.10  — current activation level (from Phase 61)
#
# Methods:
#   def score(
#       *,
#       memory: dict,             — {id, created_at, credibility_score, activation,
#                                    reference_count, goal_link_count}
#       now: datetime,
#   ) -> float
#     — recency = exp(-days_old / 30) where days_old = (now - created_at).days
#     — ref_score = min(1.0, reference_count / 10)   (saturates at 10 references)
#     — goal_score = min(1.0, goal_link_count / 5)   (saturates at 5 goals)
#     — credibility = memory.get("credibility_score", 0.7)
#     — activation = memory.get("activation", 0.5)
#     — total = (REFERENCE_WEIGHT * ref_score
#                + GOAL_LINK_WEIGHT * goal_score
#                + RECENCY_WEIGHT * recency
#                + CREDIBILITY_WEIGHT * credibility
#                + ACTIVATION_WEIGHT * activation)
#     — return round(clamp(total, 0.0, 1.0), 4)
#
#   def rank(
#       *,
#       memories: list[dict],
#       now: datetime,
#       limit: int = 10,
#   ) -> list[dict]
#     — attaches "_importance_score" to each memory, returns top `limit` sorted desc
#
#   def importance_tier(score: float) -> str
#     — >=0.8 → "critical", >=0.6 → "important", >=0.4 → "normal", <0.4 → "archivable"
```

**Tests (≥25):**
- reference_count=10 → ref_score=1.0 (saturated).
- reference_count=5 → ref_score=0.5.
- days_old=0 → recency=1.0; days_old=30 → recency≈0.37.
- goal_link_count=5 → goal_score=1.0.
- All components at max → score close to 1.0.
- All components at 0 → score close to 0.0.
- rank returns at most limit items.
- rank attaches "_importance_score".
- importance_tier: 0.85 → "critical"; 0.35 → "archivable".
- Score always in [0, 1].

---

## Phase 72 — Knowledge Graph Embedding Service

### What / Why
Phase 30 (Knowledge Graph API) exposes graph traversal.  Traversal finds paths but
cannot find structurally similar nodes ("which entities are in a similar position in the
graph as 'postgres-primary'?").  Graph embeddings encode structural position as a
vector, enabling similarity search over graph structure, not just content.

### Files to Create
- `app/services/knowledge_graph_embedding_service.py`
- `tests/test_knowledge_graph_embedding_service.py`

### Service: `KnowledgeGraphEmbeddingService`
```python
# Embedding approach: simplified DeepWalk-style random walk features.
# We do NOT use neural nets — pure Python, no GPU, no external ML libs.
#
# Methods:
#   def build_adjacency(
#       *,
#       edges: list[dict],   — each: {source_id, target_id, weight: float}
#   ) -> dict[str, list[tuple[str, float]]]
#     — {node_id: [(neighbor_id, weight), ...]}
#
#   def compute_degree_features(
#       *,
#       adjacency: dict,
#   ) -> dict[str, dict]
#     — For each node, compute:
#         in_degree, out_degree, total_degree,
#         avg_neighbor_weight, max_neighbor_weight,
#         is_hub (total_degree > mean_degree + std_degree),
#         is_leaf (total_degree == 1)
#
#   def structural_similarity(
#       *,
#       node_a: str,
#       node_b: str,
#       features: dict[str, dict],
#   ) -> float
#     — cosine similarity of feature vectors [in_degree, out_degree,
#       avg_neighbor_weight, max_neighbor_weight] (normalized by max in graph)
#
#   def find_similar_nodes(
#       *,
#       query_node: str,
#       features: dict[str, dict],
#       top_k: int = 5,
#   ) -> list[dict]
#     — returns top-k most structurally similar nodes: [{node_id, similarity}]
#     — excludes query_node from results
#
#   def identify_bridges(
#       *,
#       adjacency: dict,
#   ) -> list[str]
#     — returns node_ids with highest betweenness approximation:
#       nodes whose removal would disconnect the most pairs.
#       Approximation: nodes with in_degree >= 2 AND out_degree >= 2.
```

**Tests (≥25):**
- build_adjacency: 3 edges → correct neighbor lists.
- compute_degree_features: leaf node has total_degree=1.
- compute_degree_features: hub node flagged is_hub=True.
- structural_similarity: identical feature vectors → 1.0.
- structural_similarity: orthogonal vectors → 0.0.
- find_similar_nodes: returns at most top_k results.
- find_similar_nodes: excludes query_node.
- identify_bridges: node with in=3, out=3 is a bridge.
- identify_bridges: leaf node not a bridge.
- Empty edges → adjacency={}.

---

## Phase 73 — Cognitive Offload Scheduler

### What / Why
Not everything should be remembered.  Some information is better looked up on demand
(documentation URLs, transient debug logs).  Cognitive offloading — deciding what to
keep in memory vs. what to discard or mark "look up when needed" — prevents memory
bloat and improves retrieval signal-to-noise ratio.

### Files to Create
- `app/services/cognitive_offload_scheduler.py`
- `tests/test_cognitive_offload_scheduler.py`

### Service: `CognitiveOffloadScheduler`
```python
# Offload decision matrix:
#   Keep (store fully):
#     - content type: decision, incident_summary, goal, playbook, causal_insight
#     - importance_score >= 0.5 (from Phase 71)
#     - referenced by >= 2 other memories
#   Compress (store summary only):
#     - content type: log, event_stream, raw_data
#     - importance_score 0.2–0.5
#     - not referenced
#   Offload (store URL/pointer only, mark retrievable):
#     - content type: documentation, changelog, static_config
#     - importance_score < 0.2
#   Discard (do not store):
#     - content is a near-duplicate of existing memory (handled by noise gate)
#     - content type: "test", "debug_trace" with importance_score < 0.1
#
# Methods:
#   def decide(
#       *,
#       content_type: str,
#       importance_score: float,
#       reference_count: int,
#       is_near_duplicate: bool,
#   ) -> str   # returns "keep" | "compress" | "offload" | "discard"
#
#   def compress_content(content: str, max_chars: int = 200) -> str
#     — truncates to max_chars at sentence boundary; appends "[compressed]"
#
#   def offload_pointer(*, content: str, source_url: str | None) -> dict
#     — returns {"type": "pointer", "summary": content[:100],
#                "source_url": source_url, "retrievable": True}
#
#   def batch_decide(
#       *,
#       memories: list[dict],  — each has: content_type, importance_score,
#                                           reference_count, is_near_duplicate
#   ) -> dict[str, list[dict]]   # {"keep": [...], "compress": [...],
#                                #  "offload": [...], "discard": [...]}
```

**Tests (≥25):**
- content_type="decision", importance=0.8 → "keep".
- content_type="log", importance=0.3, not referenced → "compress".
- content_type="documentation", importance=0.1 → "offload".
- is_near_duplicate=True, importance=0.05 → "discard".
- content_type="debug_trace", importance=0.05 → "discard".
- compress_content truncates at sentence boundary.
- compress_content appends "[compressed]".
- offload_pointer returns correct dict shape.
- batch_decide: all 4 buckets populated from mixed input.
- reference_count >= 2 keeps even low-importance memory.

---

## Phase 74 — Meta-Learning Service (Learning to Learn)

### What / Why
Phase 2 (StrategyLearningService) learns which strategies work.  This is first-order
learning.  Meta-learning is second-order: learning which learning hyperparameters work
best for this organisation.  "For org-X, α=0.3 works better than α=0.25 for EMA
updates because their signal volume is higher."

### Files to Create
- `app/services/meta_learning_service.py`
- `app/models/meta_learning_config.py`
- `tests/test_meta_learning_service.py`

### Model: `MetaLearningConfig`
```python
# Table: meta_learning_configs
# Columns:
#   id: UUID PK
#   org_id: UUID FK (unique per org)
#   ema_alpha: float    default 0.25
#   noise_threshold: float   default 0.85
#   confidence_floor: float  default 0.4
#   decay_half_life_days: int  default 30
#   calibration_window: int   default 100  (samples before calibration kicks in)
#   last_tuned: datetime
#   tuning_iteration: int  default 0
```

### Service: `MetaLearningService`
```python
# Methods:
#   async def get_config(*, db, org_id) -> MetaLearningConfig
#     — returns existing config or creates default
#
#   async def tune(
#       *,
#       db,
#       org_id: str,
#       outcome_history: list[dict],  — {predicted_confidence, actual_outcome, timestamp}
#   ) -> MetaLearningConfig
#     — Computes calibration error: mean |predicted_confidence - actual_outcome|
#     — If error > 0.15 AND sample_count >= 20:
#         if error > 0.25: ema_alpha += 0.05 (learn faster)
#         else:            ema_alpha -= 0.02 (learn slower, over-tuned)
#         ema_alpha = clamp(ema_alpha, 0.05, 0.5)
#     — If noise_duplicate_rate > 0.3: noise_threshold -= 0.05 (tighten gate)
#     — If noise_duplicate_rate < 0.05: noise_threshold += 0.05 (loosen gate)
#     — noise_threshold = clamp(noise_threshold, 0.5, 0.99)
#     — increments tuning_iteration, updates last_tuned
#     — returns updated config
#
#   def calibration_error(outcome_history: list[dict]) -> float
#     — mean |predicted_confidence - actual_outcome| across history
#
#   def recommended_alpha(*, calibration_error: float, current_alpha: float) -> float
#     — if error > 0.25: return min(0.5, current_alpha + 0.05)
#     — if error < 0.05: return max(0.05, current_alpha - 0.02)
#     — else: return current_alpha  (within tolerance)
```

**Tests (≥25):**
- calibration_error: all predictions correct → 0.0.
- calibration_error: all predictions 0.5 off → 0.5.
- tune: high error (>0.25) increases ema_alpha.
- tune: low error stays at current_alpha.
- ema_alpha clamped at 0.5 maximum.
- ema_alpha clamped at 0.05 minimum.
- noise_threshold tightened when duplicate_rate > 0.3.
- noise_threshold loosened when duplicate_rate < 0.05.
- tuning_iteration increments on each call.
- < 20 samples → no tuning applied.

---

## Phase 75 — Multi-Agent Voting Engine

### What / Why
When multiple agents produce conflicting assessments of the same memory (one says
"anomaly=high", another says "anomaly=low"), there is no arbitration mechanism. A voting
engine aggregates conflicting signals into a consensus decision with weighted votes.

### Files to Create
- `app/services/multi_agent_voting_engine.py`
- `tests/test_multi_agent_voting_engine.py`

### Service: `MultiAgentVotingEngine`
```python
# Agent trust weights (prior credibility per agent type):
#   AGENT_WEIGHTS = {
#       "credibility_agent":           0.90,
#       "anomaly_detection_agent":     0.85,
#       "conflict_detection_agent":    0.85,
#       "uncertainty_reporting_agent": 0.80,
#       "hypothesis_service":          0.75,
#       "causal_reasoning_agent":      0.80,
#       "default":                     0.70,
#   }
#
# Methods:
#   def vote(
#       *,
#       ballots: list[dict],   — each: {agent_name, verdict: bool, confidence: float}
#   ) -> dict
#     — weighted_true  = sum(AGENT_WEIGHTS[b.agent_name] * b.confidence
#                            for b in ballots if b.verdict=True)
#     — weighted_false = sum(AGENT_WEIGHTS[b.agent_name] * b.confidence
#                            for b in ballots if b.verdict=False)
#     — consensus = True if weighted_true > weighted_false
#     — margin = abs(weighted_true - weighted_false) / (weighted_true + weighted_false)
#     — agreement_rate = votes_matching_consensus / total_votes
#     — returns: {consensus, weighted_true, weighted_false, margin, agreement_rate,
#                 dissenting_agents: list[str]}
#
#   def resolve_numeric(
#       *,
#       ballots: list[dict],   — each: {agent_name, value: float, confidence: float}
#   ) -> dict
#     — weighted average: sum(weight*conf*value) / sum(weight*conf)
#     — returns: {consensus_value, std_deviation, min_value, max_value}
#
#   def is_contested(*, margin: float) -> bool
#     — True if margin < 0.2 (close vote — should escalate)
```

**Tests (≥25):**
- All agents vote True → consensus=True, agreement_rate=1.0.
- 3 True, 1 False (low confidence) → consensus=True.
- Equal weighted votes → contested.
- is_contested: margin=0.1 → True; margin=0.5 → False.
- dissenting_agents: agents voting against consensus listed.
- resolve_numeric: weighted average computed correctly.
- resolve_numeric: single ballot → consensus_value = ballot.value.
- Unknown agent_name → AGENT_WEIGHTS["default"] used.
- Empty ballots → graceful result (consensus=False, margin=0).

---

## Phase 76 — Cross-Modal Reasoning Agent

### What / Why
Phase 43 (Multimodal Deep Memory) stores image/audio/video memories with searchable_tags.
Reasoning across modalities — "the screenshot shows the button is red, which correlates
with the alert that fired at the same time" — is not yet possible.  Cross-modal reasoning
links evidence across content types to form unified conclusions.

### Files to Create
- `app/agents/cross_modal_reasoning_agent.py`
- `tests/test_cross_modal_reasoning_agent.py`

### Agent: `CrossModalReasoningAgent`
```python
# Inputs:
#   - text_memories: list[dict]    — {id, content, tags, created_at}
#   - visual_memories: list[dict]  — {id, searchable_tags, modality, created_at}
#   - audio_memories: list[dict]   — {id, searchable_tags, modality, created_at}
#   - query: str                   — what to reason about across modalities
#   - time_window_minutes: int     — default 60 (co-occurrence window)
#
# Outputs:
#   - cross_modal_links: list[dict]  — {text_id, visual_id | audio_id,
#                                       link_type, shared_tags, temporal_gap_seconds,
#                                       correlation_score}
#   - unified_conclusion: str        — synthesised narrative from all modalities
#   - modalities_used: list[str]     — which modalities contributed
#   - evidence_strength: float       — how many cross-modal corroborations found
#   - confidence: float
```

**Heuristic logic:**
1. Tokenize query.
2. For each text_memory with tag overlap >= 0.2 with query tokens:
   - Find visual/audio memories within time_window_minutes AND shared_tags >= 1.
   - temporal_gap_seconds = abs((text.created_at - visual.created_at).total_seconds()).
   - correlation_score = tag_overlap * (1 - temporal_gap_seconds / (time_window_minutes*60)).
   - link_type = "temporal_co_occurrence" if gap < 300 else "thematic_co_occurrence".
3. Filter links where correlation_score > 0.1.
4. `modalities_used` = unique modalities in contributing memories.
5. `evidence_strength` = min(1.0, len(cross_modal_links) / 5).
6. `unified_conclusion` = template: "Analysis of {len(modalities_used)} modalities found
   {len(cross_modal_links)} correlated signals related to {query[:50]}."
7. `confidence` = 0.5 + 0.1 * len(cross_modal_links) clamped at 0.9.

**Tests (≥30):**
- Text and visual memory with shared tag + within window → link created.
- Gap > time_window_minutes → no link.
- correlation_score > 0 when tags overlap and gap is small.
- link_type="temporal_co_occurrence" when gap < 300 seconds.
- modalities_used contains "text" when text_memories match.
- evidence_strength saturates at 1.0 with >= 5 links.
- unified_conclusion contains query text.
- Empty visual_memories → cross_modal_links=[], modalities_used=["text"] only.
- validate_outputs passes.

---

## Phase 77 — Memory Provenance Graph

### What / Why
Phase C (ConsolidationLineageService) tracks merge provenance.  The broader question —
"where did this memory originally come from?" — is unanswered.  A provenance graph
traces the full lineage: raw ingest → enrichment → hypothesis → consolidation →
current memory state, enabling full auditability.

### Files to Create
- `app/services/memory_provenance_service.py`
- `app/models/provenance_edge.py`
- `tests/test_memory_provenance_service.py`

### Model: `ProvenanceEdge`
```python
# Table: provenance_edges
# Columns:
#   id: UUID PK
#   org_id: UUID FK
#   source_id: str    (memory_id, hypothesis_id, enrichment_run_id, or "ingest:{connector}")
#   target_id: str    (memory_id)
#   edge_type: str    ("ingest", "enrichment", "hypothesis", "consolidation", "writeback")
#   agent_name: str   (which agent/service created this edge)
#   created_at: datetime
#   metadata: dict    (JSONB — additional context)
```

### Service: `MemoryProvenanceService`
```python
# Methods:
#   async def record_edge(
#       *,
#       db,
#       org_id: str,
#       source_id: str,
#       target_id: str,
#       edge_type: str,
#       agent_name: str,
#       metadata: dict | None = None,
#   ) -> ProvenanceEdge
#
#   async def get_lineage(
#       *,
#       db,
#       org_id: str,
#       memory_id: str,
#       max_depth: int = 10,
#   ) -> dict
#     — BFS backwards from memory_id following source_id edges
#     — returns: {
#         root_sources: list[str],   — source_ids with no incoming edges
#         edges: list[dict],         — all edges in lineage
#         depth: int,                — max depth reached
#         agent_chain: list[str],    — ordered agent_names from root to target
#       }
#
#   async def get_descendants(
#       *,
#       db,
#       org_id: str,
#       source_id: str,
#   ) -> list[str]
#     — returns all target_ids reachable from source_id
#
#   def summarise_lineage(lineage: dict) -> str
#     — pure function returning human-readable provenance string
#     — e.g. "Ingested via connector → enriched by CredibilityAgent →
#             consolidated by MemoryConsolidationAgent"
```

**Tests (≥25):**
- record_edge: ProvenanceEdge created with correct fields (mock DB).
- get_lineage: returns root_sources for single-hop chain.
- get_lineage: max_depth limits traversal.
- agent_chain ordered correctly (root → target).
- get_descendants: returns all reachable targets.
- summarise_lineage: contains agent_names in order.
- Empty lineage (no edges): root_sources=[memory_id], depth=0.
- Cycle detection: max_depth prevents infinite loop.

---

## Phase 78 — Reward Signal Propagation

### What / Why
Phase 24 (FeedbackIntegrationAgent) learns from feedback.  Phase 50 (StrategyEvolution)
promotes/prunes strategies.  Neither propagates reward backwards through the causal
chain: "goal succeeded → credit the plan step that was decisive → credit the memory
that provided the key evidence."  Reward propagation closes the loop between outcomes
and the evidence/decisions that caused them.

### Files to Create
- `app/services/reward_propagation_service.py`
- `tests/test_reward_propagation_service.py`

### Service: `RewardPropagationService`
```python
# Constants:
#   PROPAGATION_DISCOUNT = 0.8   # reward attenuates with each step back
#   MIN_REWARD = 0.01
#
# Methods:
#   def propagate_backwards(
#       *,
#       outcome_reward: float,      — final reward signal: +1.0 (success), -1.0 (failure)
#       causal_chain: list[dict],   — ordered list of {step_id, step_type, contribution}
#                                     most recent step first
#   ) -> list[dict]
#     — For each step at index i:
#         discounted_reward = outcome_reward * (PROPAGATION_DISCOUNT ** i)
#         if abs(discounted_reward) < MIN_REWARD: stop
#         yield {step_id, step_type, reward_signal: round(discounted_reward, 4),
#                credit_type: "positive" if reward > 0 else "negative"}
#
#   def aggregate_credits(
#       *,
#       reward_records: list[dict],  — from multiple propagate_backwards calls
#                                      each record: {step_id, reward_signal}
#   ) -> dict[str, float]
#     — sum reward_signal per step_id, return {step_id: total_credit}
#
#   def top_credited_steps(
#       *,
#       credits: dict[str, float],
#       top_k: int = 5,
#   ) -> list[dict]
#     — returns top_k by abs(credit) descending: [{step_id, credit}]
#
#   def update_memory_importance(
#       *,
#       memory_id: str,
#       current_importance: float,
#       credit: float,
#       alpha: float = 0.1,
#   ) -> float
#     — EMA update: new_importance = (1-alpha)*current + alpha*(credit + 0.5)
#     — clamp result to [0.0, 1.0]
```

**Tests (≥25):**
- propagate_backwards: first step gets full reward, second gets *DISCOUNT.
- propagate_backwards: stops when abs(reward) < MIN_REWARD.
- Negative outcome_reward → credit_type="negative" for all steps.
- aggregate_credits: multiple records for same step_id summed.
- top_credited_steps: returns at most top_k results.
- top_credited_steps: sorted by abs(credit) descending.
- update_memory_importance: EMA formula correct.
- update_memory_importance: clamped at 1.0.
- update_memory_importance: clamped at 0.0.
- Empty causal_chain → returns [].

---

## Phase 79 — Adversarial Robustness Monitor

### What / Why
The system trusts inbound data from connectors and users.  Adversarial inputs —
prompt injection in memory content, anomalously high credibility scores, suspiciously
uniform confidence values from a compromised agent — are not detected.  A robustness
monitor flags inputs that show adversarial signatures.

### Files to Create
- `app/services/adversarial_robustness_monitor.py`
- `tests/test_adversarial_robustness_monitor.py`

### Service: `AdversarialRobustnessMonitor`
```python
# Detection checks (each returns a finding dict or None):
#
# 1. Prompt injection: content contains instruction-like patterns
#    _INJECTION_PATTERNS = [
#        r"ignore previous instructions",
#        r"disregard (all|the) (above|previous)",
#        r"you are now",
#        r"system prompt",
#        r"forget everything",
#        r"new instruction",
#        r"act as",
#    ]
#    → finding: {type: "prompt_injection", severity: "high", matched: str}
#
# 2. Score manipulation: credibility_score or confidence outside [0, 1]
#    OR credibility_score > 0.99 (suspiciously perfect)
#    → finding: {type: "score_manipulation", severity: "medium", value: float}
#
# 3. Statistical anomaly in confidence batch: if stddev of batch < 0.01 (all same)
#    AND batch_size > 5 (uniform batch is suspicious)
#    → finding: {type: "uniform_confidence_anomaly", severity: "medium"}
#
# 4. Encoding attack: content contains null bytes, control chars, or excessive
#    unicode direction overrides (U+202E)
#    → finding: {type: "encoding_attack", severity: "high"}
#
# Methods:
#   def check_content(*, content: str, metadata: dict) -> list[dict]
#     — runs checks 1, 2, 4 — returns list of findings (empty = clean)
#
#   def check_confidence_batch(*, confidence_values: list[float]) -> list[dict]
#     — runs check 3 — returns list of findings
#
#   def is_safe(*, findings: list[dict]) -> bool
#     — True if no findings with severity="high"
#
#   def risk_summary(*, findings: list[dict]) -> str
#     — returns comma-joined finding types, or "clean"
```

**Tests (≥30):**
- "ignore previous instructions" in content → prompt_injection finding.
- "act as" in content → prompt_injection finding.
- credibility_score=1.5 → score_manipulation finding.
- credibility_score=0.999 → score_manipulation finding.
- credibility_score=0.9 (normal) → no finding.
- Null byte in content → encoding_attack finding.
- U+202E in content → encoding_attack finding.
- Uniform confidence batch (all 0.7) with size 10 → uniform_confidence_anomaly.
- Varied confidence batch → no anomaly.
- is_safe: high-severity finding → False.
- is_safe: medium-severity only → True.
- risk_summary: "prompt_injection, encoding_attack" for two findings.
- risk_summary: "clean" for empty findings.

---

## Phase 80 — Cognitive State Checkpoint & Restore

### What / Why
The cognitive loop can fail mid-session (process crash, timeout, OOM).  There is no
way to resume from a consistent checkpoint — the loop restarts from scratch, losing all
in-flight reasoning.  Checkpoint & restore gives the loop crash tolerance.

### Files to Create
- `app/services/cognitive_checkpoint_service.py`
- `app/models/cognitive_state_checkpoint.py`
- `tests/test_cognitive_checkpoint_service.py`

### Model: `CognitiveStateCheckpoint`
```python
# Table: cognitive_state_checkpoints
# Columns:
#   id: UUID PK
#   session_id: UUID FK
#   org_id: UUID FK
#   checkpoint_seq: int    (incremented per checkpoint in session)
#   loop_iteration: int
#   active_goal: str
#   completed_step_indices: list[int]   (JSONB)
#   pending_step_indices: list[int]     (JSONB)
#   working_memory_snapshot: list[dict] (JSONB — from Phase 61 snapshot)
#   last_output: str
#   status: str   ("active" | "restored" | "completed" | "abandoned")
#   created_at: datetime
```

### Service: `CognitiveCheckpointService`
```python
# Methods:
#   async def save(
#       *,
#       db,
#       session_id: str,
#       org_id: str,
#       loop_iteration: int,
#       active_goal: str,
#       completed_steps: list[int],
#       pending_steps: list[int],
#       working_memory: list[dict],
#       last_output: str,
#   ) -> CognitiveStateCheckpoint
#     — writes new checkpoint row, increments checkpoint_seq
#     — returns new checkpoint
#
#   async def latest(*, db, session_id: str) -> CognitiveStateCheckpoint | None
#     — returns checkpoint with highest checkpoint_seq for this session
#
#   async def restore(*, db, session_id: str) -> dict | None
#     — loads latest checkpoint
#     — updates status="restored"
#     — returns state dict: {loop_iteration, active_goal, completed_steps,
#                             pending_steps, working_memory, last_output}
#     — returns None if no checkpoint exists
#
#   async def mark_completed(*, db, session_id: str) -> bool
#     — sets status="completed" on all checkpoints for session
#     — returns True if any rows updated
#
#   async def cleanup_old(*, db, org_id: str, keep_sessions: int = 100) -> int
#     — deletes checkpoints for sessions beyond the most recent keep_sessions
#     — returns deleted count
#
#   def diff_steps(
#       *,
#       completed: list[int],
#       full_plan: list[int],
#   ) -> list[int]
#     — pure function: returns full_plan steps not in completed
```

**Tests (≥30):**
- save: creates checkpoint with checkpoint_seq=1 for new session (mock DB).
- save: checkpoint_seq increments on second save.
- latest: returns highest checkpoint_seq (mock DB returns two rows).
- restore: returns correct state dict.
- restore: marks status="restored".
- restore: returns None when no checkpoint.
- mark_completed: sets status="completed".
- cleanup_old: deletes beyond keep_sessions limit.
- diff_steps: [0,1,2] full, [0,1] completed → [2] returned.
- diff_steps: all completed → [].
- working_memory_snapshot stored as list.
- validate_outputs on restored state (required keys present).

---

## Implementation Order & Dependencies (Phases 61–80)

```
Phase 61 (Working Memory)          — no deps; provides foundation for 67, 80
Phase 62 (Temporal Pattern Miner)  — no deps
Phase 63 (Active Knowledge Seeker) — no deps
Phase 64 (Uncertainty Propagation) — no deps
Phase 65 (Hierarchical Goal)       — depends on Phase 21 patterns
Phase 66 (Social Memory)           — no deps
Phase 67 (Episodic Future Sim)     — benefits from Phase 61 working memory
Phase 68 (Error Recovery Replan)   — no deps
Phase 69 (Semantic Role Inference) — no deps
Phase 70 (Confidence Ensemble)     — depends on Phases 19, 22, 25, 64 signals
Phase 71 (Memory Importance)       — depends on Phase 61 (activation field)
Phase 72 (Graph Embedding)         — depends on Phase 30 graph data
Phase 73 (Cognitive Offload)       — depends on Phase 71 importance_score
Phase 74 (Meta-Learning)           — depends on Phase 2 strategy learning patterns
Phase 75 (Multi-Agent Voting)      — depends on Phases 19, 22, 25 output shapes
Phase 76 (Cross-Modal Reasoning)   — depends on Phase 43 multimodal data
Phase 77 (Memory Provenance)       — depends on Gap C lineage patterns
Phase 78 (Reward Propagation)      — depends on Phase 50 outcome tracking
Phase 79 (Adversarial Robustness)  — no deps; should run early in pipeline
Phase 80 (Checkpoint & Restore)    — depends on Phase 61 working memory snapshot
```

Implement in order 61 → 80. Do not skip phases.

---

## Updated Git Workflow (same as AGI_ADVANCEMENT_PLAN.md)

```bash
git checkout main && git pull
git checkout -b agi/phase-{N}-{short-slug}

# implement phase

python -m pytest tests/test_{agent_or_service_file}.py -x -q
python -m pytest tests/ -x -q   # full suite must pass

git add app/agents/{file}.py app/services/{file}.py \
        app/models/{file}.py app/tasks/{file}.py \
        tests/test_{file}.py CLAUDE.md
git commit -m "feat: phase {N} — {phase name}"
git push -u origin agi/phase-{N}-{short-slug}
gh pr create --title "feat: phase {N} — {Phase Name}" \
  --body "..." --base main --head agi/phase-{N}-{short-slug}
gh pr merge --admin --squash --delete-branch
```

---

## CLAUDE.md Row Template (add after each phase merges)

```
| {N} | {Phase Name} | Done ({AgentOrServiceName}, {N_tests} tests, {total} total passing) |
```

---

## Celery Wiring (phases with Celery tasks)

For Phase 53 (Prospective Memory), add to `app/core/celery_app.py`:

```python
# In includes list:
"app.tasks.prospective_memory_pipeline",

# In task_routes:
"app.tasks.prospective_memory_pipeline.prospective_memory_scan_task": {"queue": "q.maintenance"},

# In beat_schedule:
"prospective-memory-scan": {
    "task": "app.tasks.prospective_memory_pipeline.prospective_memory_scan_task",
    "schedule": 300.0,   # every 5 minutes
    "args": (),
},
```

No other phases in 61–80 require new Celery tasks.

---

*Last updated: 2026-04-01*
*Covers phases 61–80. Prior phases 51–60 in AGI_ADVANCEMENT_PLAN.md.*
