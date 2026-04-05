import type { DecideResult, MemoryResult, PlanOptions, PlanResult, StreamEvent } from "./models/index.js";
import { withRetries } from "./utils/retry.js";
import { parseSseChunk } from "./utils/stream.js";

export class CognitiveGateway {
  private readonly base: string;
  private readonly headers: Record<string, string>;
  private readonly maxRetries: number;

  constructor(base: string, headers: Record<string, string>, maxRetries = 3) {
    this.base = base;
    this.headers = headers;
    this.maxRetries = maxRetries;
  }

  async decide(query: string, context: Record<string, unknown> = {}): Promise<DecideResult> {
    return withRetries(async () => {
      const response = await fetch(`${this.base}/cognitive/gateway/decide`, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({ query, context }),
      });
      if (!response.ok) throw new Error(`Decide failed: ${response.status}`);
      return (await response.json()) as DecideResult;
    }, this.maxRetries);
  }

  async plan(goal: string, options: PlanOptions = {}): Promise<PlanResult> {
    return withRetries(async () => {
      const response = await fetch(`${this.base}/cognitive/gateway/plan`, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({ goal, ...options }),
      });
      if (!response.ok) throw new Error(`Plan failed: ${response.status}`);
      return (await response.json()) as PlanResult;
    }, this.maxRetries);
  }

  async read(query: string, limit = 10): Promise<MemoryResult[]> {
    return withRetries(async () => {
      const response = await fetch(`${this.base}/cognitive/gateway/read`, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({ query, limit }),
      });
      if (!response.ok) throw new Error(`Read failed: ${response.status}`);
      const body = (await response.json()) as { items?: MemoryResult[] };
      return body.items ?? [];
    }, this.maxRetries);
  }

  async *streamRead(query: string, limit = 10): AsyncGenerator<StreamEvent> {
    const response = await fetch(`${this.base}/cognitive/gateway/stream/read`, {
      method: "POST",
      headers: { ...this.headers, Accept: "text/event-stream" },
      body: JSON.stringify({ query, limit }),
    });

    const body = response.body;
    if (!body) return;

    const reader = body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      const parsed = parseSseChunk(parts.join("\n\n"));
      for (const event of parsed) {
        yield event as StreamEvent;
      }
    }
  }
}
