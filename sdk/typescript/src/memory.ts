import type { MemoryResult } from "./models/index.js";
import { withRetries } from "./utils/retry.js";

export class MemoryResource {
  private readonly base: string;
  private readonly headers: Record<string, string>;
  private readonly maxRetries: number;

  constructor(base: string, headers: Record<string, string>, maxRetries = 3) {
    this.base = base;
    this.headers = headers;
    this.maxRetries = maxRetries;
  }

  async search(query: string, limit = 10): Promise<MemoryResult[]> {
    return withRetries(async () => {
      const response = await fetch(`${this.base}/memories/search`, {
        method: "POST",
        headers: this.headers,
        body: JSON.stringify({ query, limit }),
      });
      if (!response.ok) return [];
      const body = (await response.json()) as { items?: MemoryResult[] };
      return body.items ?? [];
    }, this.maxRetries);
  }
}
