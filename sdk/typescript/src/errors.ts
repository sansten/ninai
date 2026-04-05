export class NinaiError extends Error {
  readonly status?: number;

  constructor(message: string, status?: number) {
    super(message);
    this.name = "NinaiError";
    this.status = status;
  }
}

export class AuthError extends NinaiError {
  constructor(message = "Authentication failed", status?: number) {
    super(message, status);
    this.name = "AuthError";
  }
}

export class RateLimitError extends NinaiError {
  constructor(message = "Rate limit exceeded", status?: number) {
    super(message, status);
    this.name = "RateLimitError";
  }
}
