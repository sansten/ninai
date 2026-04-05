import { useCallback, useState } from "react";
import type { NinaiClient, StreamEvent } from "@ninai/sdk";

export function useCognitiveStream(client: NinaiClient) {
  const [streaming, setStreaming] = useState(false);
  const [events, setEvents] = useState<StreamEvent[]>([]);

  const run = useCallback(async (query: string, limit = 10) => {
    setStreaming(true);
    setEvents([]);
    try {
      for await (const event of client.cognitive.streamRead(query, limit)) {
        setEvents((prev) => [...prev, event]);
      }
    } finally {
      setStreaming(false);
    }
  }, [client]);

  return { run, streaming, events };
}
