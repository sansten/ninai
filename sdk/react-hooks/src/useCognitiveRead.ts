import { useCallback, useState } from "react";
import type { MemoryResult, NinaiClient } from "@ninai/sdk";

export function useCognitiveRead(client: NinaiClient) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<MemoryResult[]>([]);

  const run = useCallback(async (query: string, limit = 10) => {
    setLoading(true);
    setError(null);
    try {
      const results = await client.cognitive.read(query, limit);
      setData(results);
      return results;
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed cognitive read");
      throw err;
    } finally {
      setLoading(false);
    }
  }, [client]);

  return { run, loading, error, data };
}
