export async function withRetries<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  delayMs = 200
): Promise<T> {
  let lastError: unknown;
  for (let i = 0; i <= maxRetries; i += 1) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      if (i === maxRetries) break;
      await new Promise((resolve) => setTimeout(resolve, delayMs * (i + 1)));
    }
  }
  throw lastError;
}
