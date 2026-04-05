export function parseSseChunk(chunk: string): unknown[] {
  const events: unknown[] = [];
  const parts = chunk.split("\n\n");
  for (const part of parts) {
    const line = part
      .split("\n")
      .find((l) => l.startsWith("data: "));
    if (!line) continue;
    const payload = line.slice(6).trim();
    if (!payload) continue;
    try {
      events.push(JSON.parse(payload));
    } catch {
      // ignore malformed event payloads
    }
  }
  return events;
}
