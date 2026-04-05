import { withRetries } from "./utils/retry.js";

export interface Goal {
  id?: string;
  title: string;
  status?: string;
}

export class GoalResource {
  private readonly base: string;
  private readonly headers: Record<string, string>;
  private readonly maxRetries: number;

  constructor(base: string, headers: Record<string, string>, maxRetries = 3) {
    this.base = base;
    this.headers = headers;
    this.maxRetries = maxRetries;
  }

  async list(): Promise<Goal[]> {
    return withRetries(async () => {
      const response = await fetch(`${this.base}/goals`, {
        method: "GET",
        headers: this.headers,
      });
      if (!response.ok) return [];
      return (await response.json()) as Goal[];
    }, this.maxRetries);
  }
}
