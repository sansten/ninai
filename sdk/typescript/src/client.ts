import { CognitiveGateway } from "./cognitive.js";
import { MemoryResource } from "./memory.js";
import { NinaiEventStream } from "./events.js";
import { GoalResource } from "./goals.js";
import { WebhookResource } from "./webhooks.js";

export interface NinaiClientOptions {
  apiKey: string;
  baseUrl?: string;
  orgId?: string;
  timeout?: number;
  maxRetries?: number;
}

export class NinaiClient {
  readonly cognitive: CognitiveGateway;
  readonly memory: MemoryResource;
  readonly goals: GoalResource;
  readonly events: NinaiEventStream;
  readonly webhooks: WebhookResource;

  constructor(options: NinaiClientOptions) {
    const base = options.baseUrl ?? "https://api.ninai.ai/api/v1";
    const headers = {
      Authorization: `Bearer ${options.apiKey}`,
      "X-Org-Id": options.orgId ?? "",
      "Content-Type": "application/json",
    };
    const maxRetries = options.maxRetries ?? 3;

    this.cognitive = new CognitiveGateway(base, headers, maxRetries);
    this.memory = new MemoryResource(base, headers, maxRetries);
    this.goals = new GoalResource(base, headers, maxRetries);
    this.events = new NinaiEventStream(base, headers);
    this.webhooks = new WebhookResource(base, headers);
  }
}
