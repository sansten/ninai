/**
 * API Type Definitions
 * ====================
 * 
 * TypeScript types matching backend schemas.
 */

// =============================================================================
// Common Types
// =============================================================================

export interface Timestamps {
  created_at: string;
  updated_at: string;
}

// =============================================================================
// User & Auth
// =============================================================================

export interface User {
  id: string;
  email: string;
  display_name: string;
  avatar_url?: string;
  is_active: boolean;
  created_at: string;
  role_names?: string[];
  role_assignment_ids?: Record<string, string>;
}

// User shape returned by auth endpoints (backend uses full_name + roles)
export interface AuthUser {
  id: string;
  email: string;
  full_name: string;
  avatar_url?: string;
  is_active: boolean;
  clearance_level?: number;
  created_at: string;
  last_login_at?: string;
  organization_id?: string;
  roles?: string[];
}

export interface UserRole {
  id: string;
  user_id: string;
  role_id: string;
  role_name: string;
  scope_type: 'global' | 'organization' | 'department' | 'team';
  scope_id?: string;
  granted_by?: string;
  granted_at: string;
  expires_at?: string;
  is_active: boolean;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
  user: AuthUser;
  organization: Organization;
}

// =============================================================================
// Organization
// =============================================================================

export interface Organization extends Timestamps {
  id: string;
  name: string;
  slug: string;
  description?: string;
  settings?: Record<string, unknown>;
  is_active?: boolean;
  parent_org_id?: string;
}

export interface OrganizationCreate {
  name: string;
  slug: string;
  tier?: string;
  settings?: Record<string, unknown>;
}

export interface HierarchyNode extends Timestamps {
  id: string;
  organization_id: string;
  name: string;
  node_type: 'root' | 'division' | 'department' | 'team' | 'project';
  parent_id?: string;
  path: string;
  depth: number;
  settings: Record<string, unknown>;
}

// =============================================================================
// Team
// =============================================================================

export interface Team extends Timestamps {
  id: string;
  organization_id: string;
  name: string;
  slug: string;
  description?: string;
  hierarchy_node_id?: string;
  settings: Record<string, unknown>;
  is_active: boolean;
}

export interface TeamCreate {
  name: string;
  slug: string;
  description?: string;
  hierarchy_node_id?: string;
  settings?: Record<string, unknown>;
}

export interface TeamMember {
  id: string;
  team_id: string;
  user_id: string;
  organization_id: string;
  role: 'member' | 'lead' | 'admin';
  joined_at: string;
  left_at?: string;
  is_active: boolean;
}

// =============================================================================
// Memory
// =============================================================================

export type MemoryType = 
  | 'episodic' 
  | 'semantic' 
  | 'procedural' 
  | 'working' 
  | 'strategic' 
  | 'context';

export type ContentFormat = 'text' | 'markdown' | 'json' | 'code';

export type VisibilityLevel = 'private' | 'team' | 'department' | 'organization';

export type AccessLevel = 'read' | 'write' | 'admin';

export interface Memory extends Timestamps {
  id: string;
  organization_id: string;
  agent_id?: string;
  owner_user_id?: string;
  owner_team_id?: string;
  memory_type: MemoryType;
  content_format: ContentFormat;
  title?: string;
  content: string;
  summary?: string;
  tags: string[];
  importance_score: number;
  access_count: number;
  last_accessed_at?: string;
  visibility_level: VisibilityLevel;
  hierarchy_scope_id?: string;
  is_archived: boolean;
  is_deleted: boolean;
  deleted_at?: string;
  retention_until?: string;
  legal_hold: boolean;
  source_metadata: Record<string, unknown>;
}

export interface MemoryCreate {
  memory_type: MemoryType;
  content_format?: ContentFormat;
  title?: string;
  content: string;
  summary?: string;
  tags?: string[];
  importance_score?: number;
  visibility_level?: VisibilityLevel;
  hierarchy_scope_id?: string;
  owner_team_id?: string;
  source_metadata?: Record<string, unknown>;
  vector?: number[];
}

export interface MemoryUpdate {
  title?: string;
  content?: string;
  summary?: string;
  tags?: string[];
  importance_score?: number;
  visibility_level?: VisibilityLevel;
  is_archived?: boolean;
  retention_until?: string;
  source_metadata?: Record<string, unknown>;
  vector?: number[];
}

export interface MemorySearchRequest {
  query: string;
  memory_types?: MemoryType[];
  tags?: string[];
  min_importance?: number;
  visibility_levels?: VisibilityLevel[];
  include_archived?: boolean;
  limit?: number;
  offset?: number;
  use_vector_search?: boolean;
}

export interface MemorySearchResult {
  memory: Memory;
  score: number;
  highlights?: string[];
}

export interface MemoryShareRequest {
  target_type: 'user' | 'team' | 'department' | 'organization';
  target_id: string;
  access_level: AccessLevel;
  expires_at?: string;
}

export interface AccessExplanation {
  has_access: boolean;
  access_level?: AccessLevel;
  reason: string;
  permission_path?: string;
  checked_at: string;
}

// =============================================================================
// Audit
// =============================================================================

export interface AuditEvent {
  id: string;
  organization_id: string;
  event_type: string;
  actor_id?: string;
  actor_type: 'user' | 'agent' | 'system';
  resource_type: string;
  resource_id?: string;
  action: string;
  details: Record<string, unknown>;
  ip_address?: string;
  user_agent?: string;
  request_id?: string;
  parent_event_id?: string;
  timestamp: string;
}

export interface AccessLog {
  id: string;
  memory_id: string;
  organization_id: string;
  user_id?: string;
  agent_id?: string;
  access_type: 'read' | 'update' | 'delete' | 'share' | 'search';
  access_granted: boolean;
  permission_path?: string;
  denial_reason?: string;
  ip_address?: string;
  request_id?: string;
  accessed_at: string;
}

export interface AuditStats {
  total_events: number;
  events_by_type: Record<string, number>;
  events_by_action: Record<string, number>;
  top_actors: Array<{
    actor_id: string;
    actor_type: string;
    count: number;
  }>;
  recent_denials: number;
  period_start: string;
  period_end: string;
}

// =============================================================================
// Operations / Admin
// =============================================================================

export interface AgentProcess {
  id: string;
  organization_id?: string;
  agent_name: string;
  status: string;
  session_id?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  metadata?: Record<string, unknown>;
}

export interface ApiKeySummary {
  id: string;
  name: string;
  prefix: string;
  user_id?: string;
  created_at: string;
  last_used_at?: string;
  revoked_at?: string;
}

export interface PolicyVersion {
  id: string;
  policy_name: string;
  policy_type: string;
  version: number;
  rollout_status: string;
  rollout_percentage: number;
  canary_group_ids: string[];
  activated_at?: string | null;
  superseded_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  change_notes?: string | null;
  success_count?: number | null;
  failure_count?: number | null;
  error_rate?: number | null;
}

export interface ResourceBudget {
  id: string;
  period: string;
  period_start: string;
  period_end: string;
  token_budget: number;
  tokens_used: number;
  tokens_reserved: number;
  storage_budget_mb: number;
  storage_used_mb: number;
  request_budget: number;
  requests_used: number;
  admission_blocked: boolean;
  degraded_mode: boolean;
  throttle_rate: number;
  token_utilization: number;
  storage_utilization: number;
  request_utilization: number;
}

export interface Snapshot {
  id: string;
  snapshot_name: string;
  snapshot_type: string;
  status: string;
  snapshot_size_bytes: number;
  memory_count: number;
  embedding_count: number;
  storage_location?: string | null;
  compression_format?: string | null;
  parent_snapshot_id?: string | null;
  retention_days: number;
  expires_at?: string | null;
  started_at?: string | null;
  completed_at?: string | null;
  failed_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  verified: boolean;
  replicated: boolean;
  replication_targets: unknown[];
  snapshot_metadata: Record<string, unknown>;
}

export interface AlertRule {
  id: string;
  name: string;
  severity: string;
  route: string;
  channel: string;
  target: string;
  enabled: boolean;
  created_at: string;
}
// =============================================================================
// Pipeline Tasks
// =============================================================================

export interface PipelineTask {
  id: string;
  organization_id: string;
  task_type: string;
  status: string;
  priority: number;
  
  // SLA fields
  sla_deadline?: string | null;
  sla_category?: string | null;
  sla_remaining_ms?: number | null;
  sla_breached: boolean;
  
  // Resource tracking
  estimated_tokens?: number | null;
  actual_tokens?: number | null;
  estimated_latency_ms?: number | null;
  duration_ms?: number | null;
  
  // Backpressure
  blocks_on_task_id?: string | null;
  blocked_by_quota: boolean;
  
  // Retry tracking
  attempts: number;
  max_attempts: number;
  last_error?: string | null;
  
  // Metadata
  metadata?: Record<string, unknown> | null;
  trace_id?: string | null;
  
  // Timestamps
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  updated_at: string;
}

export interface PipelineTaskCreate {
  task_type: string;
  priority?: number;
  sla_category?: string | null;
  sla_deadline_minutes?: number | null;
  estimated_tokens?: number | null;
  estimated_latency_ms?: number | null;
  metadata?: Record<string, unknown> | null;
  blocks_on_task_id?: string | null;
}

export interface PipelineStats {
  total_tasks: number;
  queued_tasks: number;
  running_tasks: number;
  blocked_tasks: number;
  succeeded_tasks_last_hour: number;
  failed_tasks_last_hour: number;
  
  // SLA metrics
  sla_breached_count: number;
  sla_compliance_rate: number;
  avg_queue_time_ms?: number | null;
  avg_execution_time_ms?: number | null;
  
  // Resource utilization
  total_tokens_consumed_last_hour: number;
  avg_tokens_per_task?: number | null;
  
  // Queue depth by type
  queue_depth_by_type: Record<string, number>;
  
  // SLA breach by category
  sla_breach_by_category: Record<string, number>;
}