export type NinaiEventType =
  | "memory.created"
  | "memory.updated"
  | "memory.decayed"
  | "conflict.detected"
  | "anomaly.detected"
  | "goal.completed"
  | "heartbeat"
  | "insight.generated";

export class NinaiEventStream {
  private readonly base: string;
  private readonly headers: Record<string, string>;

  constructor(base: string, headers: Record<string, string>) {
    this.base = base;
    this.headers = headers;
  }

  streamUrl(topic = "all"): string {
    const encoded = encodeURIComponent(topic);
    return `${this.base}/memory/stream?topic=${encoded}`;
  }

  get authHeaders(): Record<string, string> {
    return { ...this.headers };
  }
}
