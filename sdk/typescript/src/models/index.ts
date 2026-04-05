export type JsonObject = Record<string, unknown>;

export interface DecideResult {
  decision?: string;
  rationale?: string;
  [key: string]: unknown;
}

export interface PlanResult {
  plan?: string;
  [key: string]: unknown;
}

export interface MemoryResult {
  id?: string;
  content?: string;
  [key: string]: unknown;
}

export interface StreamEvent {
  event?: string;
  data?: unknown;
  [key: string]: unknown;
}

export interface PlanOptions {
  depth?: number;
  context?: JsonObject;
}
