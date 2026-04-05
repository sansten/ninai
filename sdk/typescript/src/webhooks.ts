export interface WebhookRegistration {
  event_type: string;
  endpoint_url: string;
  secret_key?: string;
}

export class WebhookResource {
  private readonly base: string;
  private readonly headers: Record<string, string>;

  constructor(base: string, headers: Record<string, string>) {
    this.base = base;
    this.headers = headers;
  }

  async register(registration: WebhookRegistration): Promise<unknown> {
    const response = await fetch(`${this.base}/webhooks`, {
      method: "POST",
      headers: this.headers,
      body: JSON.stringify(registration),
    });
    if (!response.ok) {
      throw new Error(`Webhook registration failed: ${response.status}`);
    }
    return response.json();
  }
}
